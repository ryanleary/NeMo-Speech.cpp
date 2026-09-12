# Serving many requests through one wave

Status as of 2026-09-12, branch `perf/continuous-batching`. Concurrent requests
share one wave and one codec thread, and the serialization they used to queue on
is gone. This is where that stands, what it measures, and what is left.

Read [continuous-batching.md](continuous-batching.md) first — this builds
directly on the lane model it describes.

## What the shape is now

A request thread tokenizes, builds a `WaveSession`, submits it, and waits. It
touches no device state. Everything the session needs — encoding its chunks,
prefilling them into lanes, stepping, handing frames to the codec — happens on
the engine thread, and the codec decodes for every session from one more. So
the MagpieTTS backend sees one caller and NanoCodec sees one caller, regardless
of how many requests are in flight, which is the same concurrency a single
request has always run at.

| | |
|---|---|
| `codec_channel` | one session's convolution state, the graph around it, its queue and its audio sink. Pooled in the workspace, because a graph costs a backend capture |
| `codec_stream_worker` | one thread, round-robin over the channels with a chunk ready |
| `MagpieWaveService` | the wave, the thread driving it, the queue of sessions waiting for lanes |
| `MagpieStreamingWorkspace::gate` | shared for requests that can share the wave, exclusive for anything that drives the decoder from its own thread |

The gate is what replaces `runtime.cpp`'s lock. Holding it exclusively also
parks the engine, because a session only exists while its request holds the
shared side — so the sequential path and the one-time setup are safe without
either of them knowing the engine exists.

## What it measures

Identical paragraph requests (5 chunks, 25 s of audio), submitted at the same
moment through one synthesizer, end to end on a GB300:

| requests | 32 lanes | TTFA median | 128 lanes | TTFA median |
|---|---|---|---|---|
| 1 | 60.9x | 71 ms | | |
| 4 | 108.0x | 261 ms | | |
| 16 | 169.5x | 388 ms | | |
| 32 | **180.8x** | 862 ms | 176.8x | 1105 ms |
| 64 | | | 197.0x | 1346 ms |
| 128 | | | **207.4x** | 1826 ms |

Against **55x** through the serialized path, where the Nth caller also waited
N × 0.45 s to start at all.

```bash
scripts/tts/serving-load.sh . 32
CONCURRENCY="32 64 128" scripts/tts/serving-load.sh . 128
```

Each request still produces its own audio, not a share of someone else's: mean
duration per session is 25.05-25.19 s against the 25.17 s a solo run produces,
across every row above. The spread is the lane-dependent arithmetic continuous
batching already documents -- greedy sampling moves a code, and a chunk ends a
frame earlier or later.

Chunk size barely moves it at 128 requests — 167x at 16 frames, 207x at 32,
219x at 64, 203x at 128 — so the codec is no longer what bounds this. The
decode-only probe reaches 376x at 128 sessions, so roughly half the gap between
that and 207x is everything else in the end-to-end path.

## What is left

1. **Shrinking the wave.** It only grows. It is sized from everything queued and
   running when it opens, and growing means rebuilding the captured graph, which
   needs the wave empty — so a burst that arrives while a narrow wave is busy
   waits for it to drain, once. A wave that has widened for load never narrows
   again, which costs a later lone request the wide steps it does not need.
   This is the other half of "size the wave to demand".
2. **TTFA under load.** 71 ms alone against 862 ms at 32 concurrent requests.
   Two causes, both structural: a step costs time proportional to lane count, so
   the `chunk_frames` steps before first audio cost more in a wide wave; and 32
   requests are ~160 chunks queueing for 32 lanes. Admission already serves
   sessions with nothing in flight first (`plan_session_admission`), which is
   what keeps the minimum at ~340 ms rather than the median.
3. **Cancellation.** Still pre-existing and load-critical.
   `SynthesisResult::cancelled` is dead code — `synthesizer.cpp:218` sets it but
   `runtime.cpp:214` throws first, so `speech_translator.cpp:108`'s branch is
   unreachable. gRPC maps a client cancel to `INTERNAL` because `map_exception`
   fires before the `IsCancelled()` checks. A session now carries a
   `fail_reason`, which is the seam to hang a proper terminal state on; model
   the fan-out on `MicroBatcher`'s `promise.set_exception` in
   `src/asr/batching.h`.
4. **`TtsPreemptionCoordinator`** (`http_server.cpp:326`) exists only because
   synthesis was serialized — its own comment says so. It should be
   reconsidered now that it is not.
5. **Deterministic concurrency tests.** `MagpieWaveService::drive_once` is
   already the single-threaded drive mode this needs; what is missing is a test
   that can run it without a model.

## An intermittent failure, now reportable

8 of 128 concurrent requests failed once at `chunk-frames 16`, with no
diagnostic beyond "MagpieTTS synthesis failed": the engine thread is shared, so
the thread that hits an error is never the one that reports it. A session now
carries the reason and the request prints it. Not reproduced in five runs since.
If it returns it will say which of the two ways a session can leave the engine
it took.

## Gotchas

- `scripts/tts/greedy-hashes.sh` runs the **sequential** path and passes no
  `--tts.batch-size`, so it cannot see a wave regression. Use
  `scripts/tts/wave-hashes.sh` for that. Both still hold exactly: the whole of
  this work is gated on producing the same bits from a lane table that outlives
  the request.
- Byte-identity is not available for anything that reassigns lanes: the batched
  arithmetic depends on which lane a chunk occupies. Two requests sharing a wave
  is exactly that. Gate concurrent behaviour on reproducibility, per-session
  duration and chunk counts, and the sequential hashes.
- Sizing the wave one session at a time looks right and is not: it fits the
  first arrival and then never grows, because every later session finds the wave
  busy and waits. It measures identically to the lock it replaced — flat
  throughput at every concurrency, with first-audio latency climbing linearly.
  Size it from the whole queue.
- `nemo-speech` is a thin CLI over `libnemo_speech_tts.so.1`. Prove the built
  arm by symbol, not by which directory you are in.
- The PCM callback runs on the codec thread, not the request thread. With N
  sessions that is N callbacks from one thread, one per channel, and each
  request's `Pcm16Resampler` state stays its own.
