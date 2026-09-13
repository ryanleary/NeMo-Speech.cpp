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

### Picking a width

Width is not a throughput dial; it is a capacity dial with a latency price. A
step costs time proportional to lane count, so the `chunk_frames` steps before
first audio cost more in a wide wave -- and a wave narrower than the offered
concurrency makes requests queue for lanes, which costs far more than either.
Paragraph requests, one burst, first audio in ms:

| width | 32 requests | 64 requests | 128 requests |
|---|---|---|---|
| 32 | **188.8x**, 354/439/453 | 188.6x, 4530/4713/4726 | 191.7x, 8736/12980/13011 |
| 64 | 193.2x, 522/624/639 | **201.4x**, 639/779/791 | 213.3x, 7814/8200/8225 |
| 128 | 184.3x, 871/1007/1028 | 198.6x, 1051/1216/1226 | **206.7x**, 1218/1519/1542 |

(p50/p95/p99.) Read down a column, not across a row. At every offered load the
best width is the smallest one that still holds the concurrency: 64 requests get
201x at 639 ms from a 64-lane wave against 199x at 1051 ms from a 128-lane one
-- more throughput *and* 40% less latency. Go under, and latency collapses
entirely: 64 requests into 32 lanes is 4.5 s to first audio for the same 189x,
because half of them are waiting for a lane rather than decoding.

So 128 lanes is the right answer only when 128 requests are actually in flight.
For a realtime operating point, size the wave to the concurrency the service is
provisioned for and no wider.

### Herds and backlogs

A node coming online against a full queue sees a herd, and keeps seeing one
until the backlog drains. Mixed-size requests (1-40 sentences), all submitted at
once, first audio p50/p95/p99:

| lanes | 64 requests | 128 requests |
|---|---|---|
| 32 | 181.0x, 788/9073/9130 | 177.7x, 9705/39380/41066 |
| 48 | 205.9x, 549/2438/2452 | 202.1x, 2691/26304/27513 |
| 64 | **216.5x**, 659/814/824 | 210.1x, 1356/17767/23434 |

Throughput holds up under a herd -- 210-216x, at or above what a matched,
non-backlogged load gets. What does not hold up is first audio for requests
beyond the wave's width, and that is arithmetic rather than scheduling: 128
requests averaging ~74 s of audio is 9540 s of work, which at 210x is 45 s no
matter how it is ordered, so the back of the queue waits ~23 s to be heard. The
scheduler's job there is to not make it worse, and the p50 says it does not.

The one real choice is width. 32 lanes is undersized for a backlog -- it loses
both ways, 177.7x with a 9.7 s p50, because it cannot hold enough work to
amortise a step. 48 and 64 both hold; 64 is better on every measure at these
sizes. Above the width, extra concurrency costs queue wait and nothing else.

So a node draining a backlog wants its wave sized to the concurrency it intends
to hold, and should let the rest queue: widening past that trades first audio
for nothing, and narrowing below it gives up throughput as well as latency.

### How many streams it holds

The question for TTS is not how fast a request finishes but how many streams
stay ahead of playback, and the ceiling on that is aggregate throughput: 216x
means ~216 streams each consuming one audio-second per second.
`--stream-realtime` measures it -- each stream is consumed at playback rate,
accepting no more than `--jitter-ms` ahead, and a chunk arriving after the
moment it was needed counts as an underrun.

64-lane wave, 25 s streams, all arriving at once:

| streams | underruns | first audio p50 |
|---|---|---|
| 128 | 0 | 3660 ms |
| **192** | **0** | 5258 ms |
| 256 | 66 of 256 | 7751 ms |

192 concurrent realtime streams, against the one-at-a-time the branch started
from. The first-audio figures there are a 3x oversubscribed herd -- 192 arrivals
into 64 lanes -- so most of that number is queueing. Let them arrive at a rate
instead, still 192 streams:

| arrivals | underruns | first audio p50/p95/p99 |
|---|---|---|
| all at once | 0 | 5077/9361/9465 ms |
| Poisson, 30 ms mean | 9 of 192 | 1807/2919/3035 ms |
| Poisson, 100 ms mean | 0 | **410/661/733 ms** |

So the operating point is ~192 concurrent streams at 410 ms to first audio,
with 30 ms arrivals marking the edge where the queue stops absorbing bursts.

Three things had to be true together to get there, and none of them worked
alone -- the middle one made things worse by itself:

1. **The codec worker has to wake up.** It skips a channel whose caller is
   behind, but nothing signals it when that caller catches up: the delivery
   thread belongs to the request and knows nothing about the codec. Once every
   channel was backlogged it slept until some unrelated event.
2. **A session already several seconds ahead must not hold a lane.**
   `lane_demand` returns zero while its caller is backlogged. Alone this traded
   underruns for latency, because a session that yields has to win a lane back
   inside its buffer and under contention could not.
3. **Lanes go out least-buffered first.** A session with nothing queued is about
   to fall silent; one five seconds ahead can wait. A brand-new request is
   buffered at zero and so sorts to the front, which is the same rule that used
   to be spelled "unoccupied sessions first".

### Capping it

`tts.max-sessions` is how many streams the engine will carry; past it a request
waits for a slot. `tts.max-queued-sessions` refuses past a further point, for
when the wait itself is the problem. Neither is set by default.

The cap matters because an oversubscribed wave does not fail -- it delivers
stuttering audio to everyone in it, including streams that were fine before the
last arrival. 320 streams offered to a 64-lane engine:

| cap | underruns | first audio p50 |
|---|---|---|
| none | 112 of 320 | 10247 ms |
| 224 | 27 of 320 | 10249 ms |
| 192 | 11 of 320 | 10338 ms |
| **160** | **0 of 320** | 15376 ms |

Some callers wait rather than all callers stutter, and the wait is visible in
the p50. Note 224 was clean as a *static* population (below) and is not clean
under churn: a slot freed by a finishing stream is taken by a new one that needs
lanes for its opening chunk, which the population it joins has already spent.
Size the cap against the load's arrival pattern, not against the static number.

`tts.max-queued-sessions` then chooses between waiting and shedding. 0 means
nobody waits -- a full engine refuses at once -- and -1, the default, queues
without limit. Same engine, same 320 streams:

| queue | admitted | refused | underruns | first audio p50 |
|---|---|---|---|---|
| unbounded | 320 | 0 | 0 | 15376 ms |
| 0 | 160 | 160 | 0 | **3713 ms** |

Queue everyone and they all wait; turn half away and the half you took get first
audio in a fifth of the time. Which is right depends on whether a router can
place the refused request somewhere else.

### What actually sets each number

Two quantities, two different bounds, and conflating them wastes time:

**Capacity is throughput.** A stream consumes 1x, so the engine holds about as
many as its aggregate realtime factor, ~200. It is not bounded by buffer depth:
at 320 streams, delivery buffers of 5 / 15 / 30 s give 109 / 114 / 118
underruns, because 320 streams want 320x from a ~200x engine and slack cannot
invent the difference. Note the corollary -- at cap 160 the engine reports only
127x, so a cap set for safety leaves real throughput unused. 192 buys that back
at about 3% of streams underrunning.

**First audio is admitted-streams-per-lane.** A lane is held for a whole chunk,
so a stream arriving when every lane is busy waits a chunk-decode, and one
arriving two cohorts deep waits two. At 64 lanes with 320 offered:

| cap | streams per lane | aggregate | first audio p50 |
|---|---|---|---|
| 64 | 1.0 | 60.8x | 1258 ms |
| 96 | 1.5 | 82.7x | 1610 ms |
| 128 | 2.0 | 105.4x | 3683 ms |
| 160 | 2.5 | 127.3x | 3705 ms |

That is the whole trade, and it is why the two cannot both be maximised by
tuning: throughput wants many streams per lane, first audio wants one.

Widening the wave does not escape it. More lanes means each lane advances
slower -- per-lane rate is aggregate/lanes -- and below about 2x realtime per
lane streams start falling behind. At 128 lanes with 160 streams: 80 underruns
and 70x, against 0 and 127x at 64 lanes. Useful lanes top out near half the
aggregate realtime factor.

What *would* break the trade is a shorter opening chunk. Lane hold time is chunk
decode time, so first audio in a herd scales with it directly, and only the
first chunk of a request need be short -- the rest can stay full size, leaving
throughput alone. Chunks come from sentence splitting in `Synthesizer::prepare`
today, with no size control, so this is a code change rather than a setting. It
is the one lever identified that would put herd first-audio near the ~600 ms a
request sees when a lane is free for it.

**Discovering the cap did not work.** A buffer-level signal was tried in place
of a count -- hold new requests while any established stream has less than N ms
buffered -- on the theory that falling buffers are the symptom a count is a
proxy for. It measured far worse than a fixed count, 78 underruns against 11 on
the same load, and capping unproven starts alongside it did not rescue it. The
buffer is a cliff rather than a gradient: streams sit at the delivery cap until
the engine saturates and then fall together, so a threshold sees nothing until
it is too late. Discovering the count needs a signal that leads rather than
lags -- a slow-start ramp against observed underruns would, since it probes
upward rather than waiting to be told.

### A slow client stalls every stream

The right question for TTS is not how fast a request finishes but how many
streams can be held at 1x. `--stream-realtime` asks it: each stream is consumed
at playback rate, accepting no more than `--jitter-ms` ahead, and underruns are
counted. The first run found something else before it could produce a capacity number.

64 streams into a 64-lane wave, consumed at 1x: **25.5x aggregate, 63 of 64
streams underran, first audio p50 32 s** -- against 216x and 659 ms for the same
load read as fast as possible.

The cause is the PCM callback's thread. It runs on the codec worker, and there
is one codec worker for every session, so a consumer that paces -- which is what
every real client does, through TCP flow control, a blocking `stream->Write`, or
a jitter buffer -- blocks audio for every other session while it waits. The
benchmark's sleep is only the most obvious version of it.

Fixed by decoupling delivery from decoding: each request has a buffer and a
thread of its own, the codec hands audio over and moves on, and a slow caller
becomes an idle lane -- a scheduling decision the engine can act on -- rather
than a stopped thread, which it cannot. That inversion is what made the capacity
numbers above measurable at all.

Note the concurrency tables earlier in this document still describe a client
that reads as fast as the engine can produce. They are not wrong, but they
answer "how fast do requests finish", not "how many streams can it hold".

### The admission window

`tts.admission-window-ms` (default 2) and `tts.admission-window-max-ms`
(default 20) are how long the engine waits, from idle, for a burst to finish
arriving. 64 requests into a 64-lane wave:

| window | aggregate | p50/p95/p99 |
|---|---|---|
| 0 ms | 203.4x | 664/1371/2249 ms |
| 2 ms | 201.0x | 645/786/799 ms |
| 10 ms | 200.3x | 662/804/815 ms |
| 40 ms | 199.1x | 717/858/869 ms |

It is a tail knob, not a throughput knob: it takes p99 from 2249 ms to 799 ms
and costs ~1% throughput, and past 2 ms it buys nothing and starts adding to the
median. Turning it off does not buy throughput back in any useful amount. The
knob is there because the right value depends on how requests arrive, not
because there is a latency/throughput curve worth riding.

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
