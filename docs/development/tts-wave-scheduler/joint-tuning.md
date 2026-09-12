# Joint tuning: wave width x codec chunk width

GB300, greedy, median of 3, GPU idle. Script = 20 sentences (20 chunks),
paragraph = 5 sentences (5 chunks). Stock `a5b6953` at its defaults is
RTF 0.0402 / TTFA 30.3 ms on the script.

## Script case

| wave width | cf=4 | cf=8 | cf=16 | cf=32 |
|---|---|---|---|---|
| **1 (no wave)** | 0.0356 / 30 ms | 0.0345 / 36 ms | 0.0342 / 46 ms | 0.0343 / 69 ms |
| **4** | 0.0212 / 65 ms | 0.0180 / 72 ms | 0.0163 / 82 ms | 0.0155 / 104 ms |
| **8** | 0.0184 / 67 ms | 0.0151 / 74 ms | 0.0131 / 84 ms | 0.0122 / 105 ms |
| **16** | 0.0173 / 67 ms | 0.0136 / 72 ms | 0.0116 / 84 ms | 0.0107 / 105 ms |
| **32** | 0.0158 / 66 ms | 0.0119 / 72 ms | **0.0098 / 83 ms** | **0.0088 / 105 ms** |

(RTF / end-to-end time-to-first-audio. Lower is better on both.)

## Paragraph case (only 5 chunks, so width saturates at 5)

| wave width | cf=4 | cf=8 | cf=16 | cf=32 |
|---|---|---|---|---|
| **1 (no wave)** | 0.0354 / 30 ms | 0.0344 / 35 ms | 0.0341 / 46 ms | 0.0343 / 68 ms |
| **8** | 0.0226 / 37 ms | 0.0196 / 44 ms | 0.0180 / 54 ms | 0.0174 / 77 ms |

## The two knobs do different jobs

**Wave width buys throughput and costs almost no latency.** Across
widths 4 to 32 at a fixed chunk width, TTFA moves by about 1 ms while RTF
falls by a quarter. That is the payoff from decoding chunk 0 alone: the
opening audio never waits on a wave, so widening it is close to free.
Wider is simply better, up to the number of chunks available -- past that
it does nothing, which is why the paragraph saturates at 5.

**Codec chunk width buys throughput and costs latency, linearly.**
Every step from 4 to 32 adds roughly 38 ms of TTFA at any wave width,
because the first codec call cannot return until it has that many frames.

So: **set the wave as wide as the input allows, then choose the chunk
width from the latency budget.** The wave width is not a latency
trade-off; the chunk width is the only real dial.

## Operating points

| Use | Setting | RTF | TTFA | vs stock |
|---|---|---|---|---|
| Minimum latency | width 1, cf 4 | 0.0356 | 30 ms | 1.13x throughput, same TTFA |
| Streaming | width 32, cf 4 | 0.0158 | 66 ms | 2.5x throughput, 2.2x TTFA |
| Balanced | width 32, cf 8 | 0.0119 | 72 ms | 3.4x throughput, 2.4x TTFA |
| Throughput | width 32, cf 32 | 0.0088 | 105 ms | **4.6x throughput**, 3.5x TTFA |

Everything at widths 4, 8 and 16 is Pareto-dominated by width 32 on this
input: same TTFA, lower RTF. Width costs memory (the K/V ring and the
padded cross-attention arena both scale with it), which is the real
reason to cap it.

## Caveat

"Width 32" on a 20-chunk script means one chunk alone and then all 19
remaining in a single wave. The numbers do not extrapolate to inputs with
many more chunks than the width; there the last group's stragglers start
to matter, since a wave runs until its slowest member finishes.

## GB10 (sm_121), deployed 2026-09-10

Same twenty-sentence script, greedy. Stock a5b6953 built on the box.

| Setting | RTF | vs stock |
|---|---|---|
| stock, cf4 | 0.0669 | - |
| ours w1, cf4 (kernels only) | 0.0601 | 1.11x |
| ours w8, cf16 | 0.0342 | 1.96x |
| ours w32, cf16 | 0.0293 | 2.28x |

GB10 is ~1.66x slower than GB300 at stock. The wave is worth less here
(2.3x against 4.5x) because the codec binds sooner -- stock codec RTFx is
23 against 69 -- and chunk width saturates at 16 rather than 32.

Decoder-bound at the best setting: decoder RTFx 48.7 against codec 99.2.

### Why the composed local-transformer graph cannot be sub-chained

Wave widths above 8 failed outright until the sticky-error fix. The cause
is not a resource limit:

  width 8    child graph n=24   t0=24                  composes
  width 12   child graph n=42   t0=36 t10=3 t11=3      cudaErrorNotSupported

Types 10 and 11 are MemAlloc and MemFree. CUDA does not allow a child
graph containing memory allocation nodes to be embedded with
cudaGraphAddChildGraphNode.

It is NOT the batch threshold. The same widths on GB300:

  GB300 width 8    n=24  t0=24     composes
  GB300 width 12   n=24  t0=24     composes
  GB10  width 8    n=24  t0=24     composes
  GB10  width 12   n=42  t0=36 t10=3 t11=3   fails

Same source, same batch, same ggml. GB300's graph does not change at all
across that boundary; GB10's gains twelve kernels and three alloc/free
pairs. So ggml is selecting a different matmul path per architecture --
ggml_cuda_should_use_mmq takes the compute capability, and the two boxes
also carry different cuBLAS versions (CUDA 13.2 against 13.0) whose
heuristics pick algorithms with different workspace needs. Only the
sm_121 path allocates, and only allocation makes the graph unembeddable.

The *first* add fails, so splitting the chain into sub-chains cannot help:
any sub-chain containing that round carries the same alloc nodes.
GGML_CUDA_NO_VMM=1 does not remove them either (verified: still n=42 with
t10/t11).

Removing them would mean patching ggml to pre-allocate that workspace
outside capture. Low priority: eager execution at width 12 (0.0316)
already beats composed execution at width 8 (0.0342), so the wider batch
more than pays for losing the composed graph.
