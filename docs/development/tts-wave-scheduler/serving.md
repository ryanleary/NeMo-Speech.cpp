# Serving many requests through one wave

Status as of 2026-09-12, branch `perf/continuous-batching`. The decode engine
serves concurrent requests and the throughput claim is measured; what is missing
is audio output for more than one of them at a time. This is a handoff: what is
done, what the numbers are, and the one decision that gates the rest.

Read [continuous-batching.md](continuous-batching.md) first — this builds
directly on the lane model it describes.

## The problem

Every request from every frontend serializes on `std::lock_guard(mutex_)` at
`runtime.cpp:162`. One `Synthesizer` is shared by gRPC, HTTP and the speech
translator (`engine_registry.cpp:93`); there is no pool. A paragraph request is
5 chunks / 25 s of audio at **55x realtime**, because 5 chunks means 4 lanes and
the fixed cost is amortised over 25 s. Under load the Nth request waits N x
0.45 s.

The decode path was never the obstacle. Lanes are independent and chunks are
admitted mid-run; the scheduler simply sat inside a per-request function.

## What is done

| commit | |
|---|---|
| `aeed9d3` | every item of a batched round gets its own voice, cfg, temperature, top-k, seed, frame index |
| `a8bfe52` | `WaveSession` — one request's chunks, conditioning, cursors, sinks |
| `d1115bd` | `WaveEngine` — lanes, runtime, guidance pair, RNG position |
| `d5c11c7` | multi-session decode probe (`MAGPIE_MULTI_SESSION=N`) |
| `2c7a6e4` | `plan_session_admission` — which request fills each idle lane |

A lane holds a `(session, item)` pair and the engine reads **no** request
parameters directly: it asks the lane's owner which voice to open with, which
settings to sample with, and who gets the frame. That is the whole multi-tenant
contract. `grep -c "params\."` inside `struct WaveEngine` returns only
`hparams` hits.

## The measurement that matters

`MAGPIE_MULTI_SESSION=N` runs N independent sessions through one engine, round
robin, counting frames and dropping audio. Each session gets the whole input, so
N is N identical requests arriving at once. Paragraph-sized sessions (5 chunks),
decode only, GB300:

| sessions | 32 lanes | occ | 128 lanes | occ |
|---|---|---|---|---|
| 1 | 42.9x | 13% | | |
| 4 | 149.2x | 49% | | |
| 8 | 176.9x | 60% | | |
| 16 | 222.5x | 79% | | |
| 32 | **260.8x** | 98% | 300.3x | 60% |
| 64 | 253.7x | 94% | 348.4x | 74% |
| 128 | | | **376.5x** | 83% |

**Throughput tracks occupancy almost exactly.** That is what says the engine is
indifferent to whose chunks it carries: one session leaves 87% of the lanes idle
and gets 43x; filling the same lanes with 32 sessions gets 261x. Against the 55x
a paragraph gets through the serialized path today, that is the payoff.

```bash
MAGPIE_MULTI_SESSION=32 nemo-speech synthesize "<paragraph>" --device cuda \
  --tts.batch-size 32 --tts.longform-history-tokens 20 --tts.chunk-frames 32 --verbose
```

## The decision that gates the rest

**How the codec serves more than one session.** `codec_stream_state` and
`codec_stream_graph` are borrowed by reference from the workspace
(`magpietts.cpp`, the `codec_stream_worker` constructor), so two sessions would
corrupt each other's convolution state immediately. Nothing can stream audio for
two requests until this is settled.

Spiked 2026-09-12 — codec throughput against chunk size, 150 chunks:

| chunk_frames | codec | e2e | TTFA |
|---|---|---|---|
| 16 | 245.7x | 122.1x | 189 ms |
| 32 | 338.3x | 139.9x | 262 ms |
| 64 | 419.2x | 152.4x | 400 ms |
| 128 | 474.6x | 159.1x | 707 ms |
| 256 | 494.7x | 160.4x | 823 ms |

The codec is **per-call-overhead bound**, saturating ~495x — the same shape the
decoder's prefill had. So:

1. One codec thread can outrun the decoder (495x vs 376x) *if* chunks are large.
2. Large chunks destroy latency (823 ms TTFA at 256), so that ceiling is only
   reachable in a configuration nobody serving interactively would pick.
3. Batching B sessions into one call amortises the same overhead **without**
   enlarging any session's chunk: cf=16 latency at cf=256 efficiency, worth
   about **2x**.

Three ways out, in increasing cost:

- **One worker, N stream states.** Safe — keeps today's GPU concurrency of one
  codec thread beside the decode thread. Caps the engine near 345x with the
  small chunks a latency-sensitive server wants.
- **N worker threads.** Needs the ggml backend's thread-safety for concurrent
  graph compute actually established, not assumed. Note one request already runs
  two threads against the device (request thread on magpie, worker on NanoCodec)
  — that is the current bound, and this raises it to N+1.
- **Batched codec.** `decodeStream(state, graph, frames, threads, audio)` is one
  state, one graph, one stream, and there is no batch axis anywhere in the
  NanoCodec model. Adding one means threading a lane dimension through the conv
  stack with per-lane conv caches — the same shape of work the wave decoder's
  lane axis took, which was many commits. Worth ~2x.

## What remains

1. **Engine thread and submission queue.** The policy exists
   (`plan_session_admission`); what is missing is the thread and sessions
   arriving asynchronously. The engine thread must never block on a session:
   today `drain_item` calls `codec_worker.write_frame`, which waits on a
   condition variable, from inside the step loop — one backpressured session
   would stall every lane. Invert it so the codec pulls.
2. **Delete the serialization.** `runtime.cpp:162`'s `lock_guard`, and the
   per-request reset in `MagpieStreamingWorkspace::beginRequest`, which clears
   KV caches, codec stream state and resizes the CUDA sampler.
3. **Cancellation and errors.** Pre-existing and load-critical.
   `SynthesisResult::cancelled` is dead code — `synthesizer.cpp:218` sets it but
   `runtime.cpp:214` throws first, so `speech_translator.cpp:108`'s branch is
   unreachable. gRPC maps a client cancel to `INTERNAL` because `map_exception`
   fires before the `IsCancelled()` checks. The codec worker's real failure
   reason goes to stderr and only a bool survives `join()`. Model the fix on
   `MicroBatcher`'s `promise.set_exception` fan-out in `src/asr/batching.h`.
4. **Sizing the wave to demand.** See the TTFA note below.
5. **Concurrency tests.** A deterministic single-threaded drive mode — `step()`
   called by the test, no engine thread — is what makes multi-session behaviour
   testable without timing flakiness.

Follow `src/asr/batching.h` for config style (`BatchingConfig::Register`,
`asr.batching.*` → a `tts.batching.*` block) and for exception fan-out. Do **not**
reuse `MicroBatcher` itself: it is request-to-single-result and synthesis is a
long-lived generator. It is the right shape for one decode step, which is how
ASR uses it.

## Open regression: TTFA

Dropping the solo chunk-0 pass in `d1115bd` cost first-audio latency:

| | before | after |
|---|---|---|
| paragraph, 32 lanes | 71 ms | 118 ms |
| 150 chunks, 128 lanes | 63 ms | 452 ms |

It had to go — an engine outliving a request cannot tear its runtime down — and
its original rationale had already expired, since the reorder buffer streams the
head chunk's frames as they are produced. But the cost is structural: first audio
needs `chunk_frames` decode steps and a step costs time proportional to lane
count.

`chunk_frames` trades one for the other and is already per-session (a paragraph
at 32 lanes gives 118 ms / 60x at 32 frames, 52 ms / 44x at 4) but does not close
the gap at 128 lanes, where the wide prefill and wide steps dominate. **Sizing
the wave to demand** — start narrow, widen as load arrives — is the real answer
and belongs with the engine thread. Note the lane cap already refuses to make a
wave wider than half the pending chunks, for a related reason.

## Gotchas

- `scripts/tts/greedy-hashes.sh` runs the **sequential** path and passes no
  `--tts.batch-size`, so it cannot see a wave regression. Use
  `scripts/tts/wave-hashes.sh` for that.
- Byte-identity is not available for anything that reassigns lanes: the batched
  arithmetic depends on which lane a chunk occupies. Gate on reproducibility,
  duration, per-chunk step counts and the sequential hashes.
- `nemo-speech` is a thin CLI over `libnemo_speech_tts.so.1`. Prove the built arm
  by symbol, not by which directory you are in.
- The engine's `idle_codes` exists so a lane no chunk has reached yet still has
  in-range tokens; the graph decodes every lane either way.
