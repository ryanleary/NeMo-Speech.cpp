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

The HTTP server's `TtsPreemptionCoordinator` is gone with it. It admitted one
request at a time and cancelled the older one to let a newer start, for the
reason its own comment gave -- that the runtime serialized synthesis. `tts.preempt`
is still accepted so existing configurations load, and says once that it is
ignored.

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

Those requests all arrive at once, which is the worst arrival pattern there is
and not what a server sees. At a rate, 32 requests of the same size, 32 lanes:

| arrival | aggregate | TTFA median | TTFA min/max |
|---|---|---|---|
| all at once | 174.6x | 504 ms | 339/1403 ms |
| 50 ms apart | 167.5x | 571 ms | 127/942 ms |
| 150 ms apart | 136.4x | 243 ms | 120/517 ms |
| 400 ms apart | 63.0x | 153 ms | 113/726 ms |

So first audio is 150-250 ms at a sustainable offered load, not the 862 ms the
thundering herd suggests, and the curve is the ordinary one: the fuller the
wave, the better it amortises and the longer a new arrival waits for a lane.
The last row is not a ceiling -- 32 arrivals 400 ms apart span 12.4 s on their
own, so the engine is simply not being offered enough work.

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
2. **TTFA under load.** 71 ms alone against 243 ms at a sustained rate.
   Two causes, both structural: a step costs time proportional to lane count, so
   the `chunk_frames` steps before first audio cost more in a wide wave; and 32
   requests are ~160 chunks queueing for 32 lanes. Admission already serves
   sessions with nothing in flight first (`plan_session_admission`), which is
   what keeps the minimum at ~340 ms rather than the median.
3. **First audio in a wide wave.** 718 ms at 128 lanes against 243 ms at 32,
   because a step costs time proportional to lane count. Narrowing the wave
   when demand falls is the same lever as (1).

4. **Cancellation beyond the callback.** A client hanging up is handled: the
   run returns its audio so far with `cancelled` set, gRPC answers CANCELLED
   and HTTP 499, and the session releases its lanes without disturbing its
   neighbours. What is not handled is a request that wants to stop while it is
   still queued is handled too, through `should_cancel`. Shutdown drains rather
   than failing what it is serving.
5. **Deterministic concurrency tests.** `MagpieWaveService::drive_once` is
   already the single-threaded drive mode this needs; what is missing is a test
   that can run it without a model.

## Sizing the wave, and what it costs first audio

Lanes, text capacity and the position budget are baked into the captured graph,
so growing the wave means rebuilding it, which needs the lanes empty. The
barrier used to wait for every admitted session to *finish*. A burst landing
just after the engine had sized itself for one request was therefore held for
that request's entire synthesis -- visible as a first-audio tail of 1449 ms
against a 495 ms median for the same burst.

It now waits for the chunks in the lanes instead. A chunk mid-decode cannot be
un-admitted, but a session can stop being fed: those already admitted keep their
place and carry on once the wider wave is up. Worst case fell to 979 ms with
median and throughput unchanged.

The rest of the tail was not the barrier either. The first request to arrive
woke the engine and was admitted alone into every idle lane -- and since
`plan_session_admission` gives a session one lane before round-robining the
rest, that one request's *continuation* chunks took the other 31. Its
neighbours, milliseconds behind, found nothing free and waited a whole chunk.

So the engine now waits ~2 ms from a standing start for the rest of the burst,
extending while the queue is still growing and capped at 20 ms; and while anyone
is waiting on their first lane, the burst threshold drops to one. Steady state,
same burst:

| width / requests | before | after | throughput |
|---|---|---|---|
| 32 | 245/351/1439 ms | 243/348/445 ms | 180 -> 189x |
| 128 | 736/2170/4145 ms | 718/1238/1548 ms | 213 -> 214x |

Throughput is not paid for it: admitting a full burst at once is fewer prefills,
not more. `--rounds` runs a discarded warm-up burst first, so the per-channel
codec graph builds (~145 ms on the first burst's median, once per process
because channels are pooled) do not land on a measured round.

What remains at 128 lanes is the floor, not a queue: a step costs time
proportional to lane count, so the `chunk_frames` steps before first audio cost
more in a wide wave. 718 ms is the minimum any request sees there, and the
median is within 2x of it.

Smaller codec chunks cut the minimum but not the median, which is consistent
with the tail being elsewhere:

| chunk_frames | aggregate | TTFA min/median/max |
|---|---|---|
| 4 | 78.8x | 169/304/1107 ms |
| 8 | 115.0x | 190/775/906 ms |
| 16 | 155.1x | 202/790/929 ms |
| 32 | 179.7x | 331/872/1359 ms |

Note the obvious lever is closed: NanoCodec zero-pads a short chunk to the
graph's width and the caches are refreshed from the padded input, so a short
**non-final** read would carry that padding into the next chunk's convolution
state. Short reads are safe only at the end of a stream, which is why the codec
does them only there. Giving a session a small opening chunk needs the state to
survive a graph rebuild, which today it does not -- `nc_stream_decode_graph_init`
frees the caches.

## Cancellation

The PCM callback returning false used to travel down exactly the path a decode
error does, so the runtime threw before `SynthesisResult::cancelled` could be
returned. That one throw was why gRPC answered a client cancel with `INTERNAL`,
why the speech translator's cancelled branch was unreachable, and why the C
API's two checks were dead code -- four apparently separate bugs with one cause.

It is recorded now where it happens, in `stream_audio_outputs::write_audio`,
which is the only place that knows the callback said stop. Gate:

```bash
# a quarter of 32 concurrent requests hang up mid-stream
nemo-speech synthesize "<paragraph>" --concurrency 32 --cancel-after-ms 300 ...
```

8 cancelled, 0 failed, and the 24 survivors produce 24.4-25.9 s of audio each
against the 25.2 s they produce alone. A cancelled session releases its lanes
and its neighbours do not notice. `should_cancel` covers the case the callback
cannot: cancelling at 30 ms, before any audio exists, returns cleanly with zero
samples.

One survivor in one run produced 32.5 s rather than ~25 s. That is the
straggler behaviour long-form decoding already has -- a degenerate chunk runs to
its step budget -- not something cancellation introduced, but it has not been
run down.

## The failure that was silent

8 of 128 concurrent requests failed at `chunk-frames 16` with no diagnostic at
all. The cause was not in the sessions. The codec worker and the wave service
are built on first use, from a request thread holding only the shared side of
the gate, so the first burst all found the pointer null, all built one, and the
losing assignment destroyed the winner's -- whose `stop()` failed every session
already inside it. Both accessors are guarded now, and a service that is serving
refuses to be rebuilt rather than being stopped under the requests in it.

Two lessons worth keeping. Lazy initialisation is a writer, so it belongs on the
exclusive side of a gate or behind its own lock; the shared side is exactly
where it looks safe and is not. And an error with no reason cost far more to
find than the fix was worth -- a session now carries a `fail_reason`, because
the thread that hits an error on a shared engine is never the one reporting it.

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
