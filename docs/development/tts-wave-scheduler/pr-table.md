## Results

NVIDIA GB300 (sm_103), CUDA 13.2, aarch64. Greedy (`--top-k 1`), median of
five runs, GPU idle-gated, via `scripts/tts/bench_magpietts.py`. RTF is
seconds of compute per second of audio, so lower is better. Baseline is
stock `main` at `a5b6953`, built from the same tree with the same flags.

### At the shipped defaults

| Case | Stock `a5b6953` | This branch | Change | Speed-up |
|---|---|---|---|---|
| line (1 sentence) | 0.0431 | 0.0393 | −8.8% | 1.10× |
| paragraph (5 sentences) | 0.0401 | 0.0226 | −43.6% | 1.77× |
| script (20 sentences) | 0.0402 | **0.0172** | **−57.2%** | **2.34×** |

Decoder inter-token latency on the script case: **1.74 → 0.61 ms**.
Throughput: about 25× real time → **56× real time**.

The line case is a single chunk, so no wave forms and only the kernel-level
work applies. It is the control: it should move a little, and does.

### Across the codec chunk width

`--tts.chunk-frames` (codec frames per decode call, default 4) is a
pre-existing knob. It matters here because the two branches are bound by
different things: stock is decoder-bound, so widening the codec barely helps
it; this branch has removed most of the decoder cost, which exposes the codec
and lets the knob pay. **Script case:**

| chunk-frames | Stock | This branch | Speed-up |
|---|---|---|---|
| 3 | 0.0409 | 0.0209 | 1.96× |
| **4 (default)** | **0.0402** | **0.0177** | **2.27×** |
| 8 | 0.0394 | 0.0135 | 2.92× |
| 16 | 0.0389 | 0.0113 | 3.44× |
| 32 | 0.0388 | 0.0103 | 3.77× |

**Paragraph case:**

| chunk-frames | Stock | This branch | Speed-up |
|---|---|---|---|
| 3 | 0.0410 | 0.0252 | 1.63× |
| **4 (default)** | **0.0401** | **0.0220** | **1.82×** |
| 8 | 0.0394 | 0.0180 | 2.19× |
| 16 | 0.0387 | 0.0157 | 2.46× |
| 32 | 0.0386 | 0.0146 | 2.64× |

Stock moves 3.5% across the whole sweep; this branch moves 42%. Best against
best, that is **0.0388 → 0.0103, a 3.8× speed-up**.

Widening the codec is not free: the decoded audio is not byte-identical
across chunk widths (−49 dB SNR, max sample delta 132 of 32768 — boundary
effects in the chunked convolution, below audibility but enough to break the
reproducibility check). Time-to-first-audio is unaffected on this branch
(2.1 → 4.3 ms of codec latency, against a ~490 ms decoder-bound TTFA).

### Latency

A wave only emits once its slowest member finishes, so the scheduler
decodes chunk 0 alone and streams the chunk at the head of each group frame
by frame rather than buffering it. Without that, first audio waits for a
whole group.

| Case | Stock | This branch | Wave, if the head did not stream |
|---|---|---|---|
| paragraph | 29.9 ms | 37.4 ms | 288.2 ms |
| script | 30.3 ms | 66.3 ms | 492.5 ms |

The sequential path is unchanged at 29.7 ms. The residual gap on the script
case is the encoder pre-pass, which plans all twenty chunks before decoding
starts.

### Tuning the two knobs

They do different jobs, which makes them easy to set:

- **Wave width** (`--tts.batch-size`) buys throughput at almost no latency
  cost -- across widths 4 to 32 TTFA moves ~1 ms while RTF falls a quarter,
  because chunk 0 never waits on a wave. Set it as wide as the input allows;
  it costs memory, not latency.
- **Codec chunk width** (`--tts.chunk-frames`) buys throughput and costs
  latency linearly: every doubling from 4 to 32 adds ~38 ms of TTFA at any
  wave width.

| Use | Setting | Script RTF | TTFA |
|---|---|---|---|
| Minimum latency | width 1, cf 4 | 0.0356 | 30 ms |
| Streaming | width 32, cf 4 | 0.0158 | 66 ms |
| Balanced | width 32, cf 8 | 0.0119 | 72 ms |
| Throughput | width 32, cf 32 | **0.0088** | 105 ms |

Against stock's 0.0402 / 30 ms, the throughput point is **4.6x** at 3.5x the
latency, and the streaming point is 2.5x at 2.2x. Full grid in
`Z-joint-tuning.md`.

### Where the script case's gain comes from

Each row is the full stack up to that point, at the default chunk width.

| Stage | Script RTF | vs previous |
|---|---|---|
| Stock `a5b6953` | 0.0402 | — |
| + half_snake aliasing guard, no-op copy elision, two ggml patches, device-resident codec state, flash-attention in the persistent decoder | 0.0357 | −11.2% |
| + wave scheduler (long-form chunks decoded in lockstep) | 0.0323 | −9.5% |
| + batched local transformer | 0.0229 | −29.1% |
| + batched cross-attention | **0.0177** | −22.7% |

### Correctness

- The sequential path is **bit-identical** to before at every stage:
  line `37b7f9808f3da46f`, paragraph `0cd06cc3307442f5`,
  script `5c5d4ebc9e3b9d72`. Batching is opt-in via `--tts.batch-size`.
- Wave output holds its length: script 101.5 s against the sequential
  101.7 s (−0.2%). Paragraph comes out 24.8 s against 26.0 s (−4.6%) —
  every chunk still reaches the end of its text, so no words are dropped,
  but chunk ends land a few frames earlier.
- A wave requires a pinned long-form history
  (`--tts.longform-history-tokens >= 0`). The adaptive default derives chunk
  N's text window from chunk N-1's decode, which a wave has not run; the gate
  refuses rather than silently changing behaviour.

### What is left

At the default chunk width the decoder producer (1014 ms) and the codec
consumer (1019 ms) are within 0.5% of each other and run concurrently, so
the wall clock is whichever is slower. Inside the producer, the largest
remaining item is the per-chunk opening prefill: 197 ms over 20 chunks,
19.4% of the thread, still running one chunk at a time through the
single-item path. Batching it only pays once the codec is out of the way
(`--tts.chunk-frames` 16 or more), where it would be worth roughly another
10-15%.
