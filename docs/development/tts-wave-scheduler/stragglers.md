# Straggler chunks, and what continuous batching would buy

Written 2026-09-11 against `perf/upstream-best` @ `6e52dd3` (PR #38). Measured
on grandteton, GB300 sm_103, CUDA 13.2.

## The short version

A wave decodes its whole group in lockstep and pays for its slowest member.
On the inlined benchmark that costs almost nothing, because every chunk runs a
similar number of steps. On real English prose it is the largest remaining
cost in the decoder.

Two ways to stop paying it. **Narrowing** -- retiring finished items and
re-forming the group smaller -- is worth 3-6%. **Refilling** -- admitting the
next pending chunk into the freed lane, which is continuous batching -- is
worth ~60% of decode time and ~37% end to end. That is a 10x difference, and
it inverts the plan we had been working to.

## How the two input distributions differ

`bench_magpietts.py` repeats five sentences, all eleven words or longer.
`bench_longform_prose.py` samples ~12.3k real sentences from four Gutenberg
texts: mean 17.2 words, median 15, p10 about 6, p90 33.

Per-chunk decode steps, measured at width 32:

| input | chunks | mean steps | max | max/mean |
|---|---|---|---|---|
| inlined benchmark, 20 sentences | 20 | 114 | 143 | 1.26 |
| prose, 16 sentences | 17 | 119 | 213 | 1.79 |
| prose, 32 sentences | 38 | 129 | 244 | 1.90 |
| prose, 64 sentences | 75 | 129 | 426 | 3.29 |
| prose, 128 sentences | 150 | 126 | 426 | 3.37 |

The group runs until its longest member finishes, so that ratio is roughly the
factor by which the group overpays.

This is also why the branch measures slower on prose than the PR's headline:
0.0083 on the inlined script against 0.0115 on 16 sentences of prose, same
settings, same binary. The PR number reproduces exactly; the gap is entirely
this effect.

## Where the stragglers come from

They are pre-existing model behaviour, not a scheduler bug. A chunk whose
attention does not advance through its text keeps decoding: ~291 steps where a
normal chunk finishes in 115-145, at roughly one text position per
`attention_prior_advance_threshold` steps. Which chunk it lands on moves with
any change to the arithmetic -- a different GPU, a different kernel, a
different graph shape -- so it is not reproducible per-chunk, only
statistically.

Sequential decoding pays a straggler once. A wave pays it `group_width` times,
because lockstep holds every other item until it finishes. That multiplication
is the whole problem.

**This was settled, and the answer was a port error rather than a threshold.**
The prior held at the last attended position when its lookahead window emptied
within three tokens of the text end, where the reference jumps to the final
token. See "After the empty-window parity fix" below, and PR #44. The shipped
sink thresholds (8 and 10) remain unexplained -- our conversion script invented
them for keys that exist in no NeMo config, and NeMo 3.1.0 hardcodes 4 at both
sites -- but with the missing branch restored they no longer drive the
pathology.

## Cost model

Measured on GB300: a wave step costs

    step(B) = 0.764 + 0.0496 * B  ms

for `B` live items. **76% of it is fixed** regardless of width, which is the
key number: shedding lanes buys much less than the item-count reduction
suggests. Re-forming a runtime costs ~12 ms -- construction plus the first
step's graph capture -- which is about 25 steps' worth.

Sanity check: the model predicts 3.46 s of wave-step arithmetic for the
150-chunk case, against 7.46 s of measured producer wall time, so decode steps
are ~46% of the producer. Profiles put that fraction at 42-58%, so the model is
right to about the right fraction. Everything below is a projection on measured
step counts, not an end-to-end measurement.

## Projection

Four policies on the same real step counts, width 32:

| chunks | max/mean | current | narrow (1 re-form) | perfect retire | refill |
|---|---|---|---|---|---|
| 38 | 1.90 | 0.87 s | +2.7% | +22.5% | **+58.7%** |
| 75 | 3.29 | 1.94 s | +4.5% | +35.3% | **+63.2%** |
| 150 | 3.37 | 3.46 s | +5.7% | +36.7% | **+59.8%** |

End to end, with the codec floor respected -- it runs concurrently with the
producer, so it caps the win:

| chunks | audio | now | projected | xRT now | xRT projected |
|---|---|---|---|---|---|
| 38 | 215 s | 2.02 s | 1.51 s | 106 | **142** |
| 75 | 427 s | 4.23 s | 3.00 s | 101 | **142** |
| 150 | 830 s | 7.59 s | 5.52 s | 109 | **150** |

### Why narrowing loses

It only helps the tail. One re-form per group, fired when the waste already
covers its cost, leaves the group at full width through most of its life; and
with 76% of a step fixed, the narrowed remainder is not much cheaper per step.
Even *perfect* retirement -- re-forming on every finish -- tops out at +37%,
and would pay ~12 ms each time to get there.

### Why refilling wins

It changes what the cost is proportional to. Currently a group costs
`max(steps) * step(W)`; with refilling it costs `sum(steps) / W * step(W)`.
With max/mean at 3.3, that ratio is most of the waste. It also gets better as
inputs get longer, because there is always another chunk waiting to fill a
freed lane.

## What refilling requires

The wave's economy rests on one invariant, stated in `decoder.cpp`:

> what they share is the step index, which is what lets a single ring head and
> a single mask serve them all

Every item is at the same step. That is what makes one `ring_head_`, one
`fa_mask`, and one `write_rows` serve the whole batch. A newly admitted chunk
starts at step 0 while its neighbours are at step 140, so the shared ring head
no longer describes it.

Refilling therefore needs **per-item ring positions and per-item masks**. That
is a decoder redesign, not an addition. Narrowing needs neither -- every
survivor is still at the same step, so the arena copy preserves slot positions
and the mask recomputes unchanged -- which is exactly why narrowing is cheap to
build and not worth building.

Not yet scoped: what per-item ring positions do to the captured CUDA graph.
The graph's whole value is that shapes are fixed while the ring rotates; per
-item heads may or may not preserve that. That question should be answered
before anything is written, because if it forces a graph rebuild per admission
the projection above is void.

## After the empty-window parity fix (PR #44)

The prior held at the last attended position when its lookahead window ran out
within three tokens of the text end, where the reference jumps to the final
position. Every chunk therefore paid a crawl at its end. Fixing it drops mean
steps per chunk ~14% and leaves `max` almost untouched, so the ratio this design
exploits gets *larger*, not smaller:

| chunks | ratio before | ratio after | refill saving before | after |
|---|---|---|---|---|
| 38 | 1.90 | 2.02 | +58.7% | **+60.8%** |
| 75 | 3.29 | 3.61 | +63.2% | **+65.8%** |
| 150 | 3.37 | 3.72 | +59.8% | **+62.8%** |

End to end, post-fix, with the codec floor:

| chunks | audio | now | projected | xRT now | xRT projected |
|---|---|---|---|---|---|
| 38 | 215 s | 1.92 s | 1.43 s | 112 | **150** |
| 75 | 426 s | 4.01 s | 2.80 s | 106 | **152** |
| 150 | 829 s | 7.36 s | 5.31 s | 113 | **156** |

The two changes compound: the parity fix lowers the floor every chunk paid,
continuous batching removes the tail one chunk imposes on its whole group.
Post-fix step counts are in `steps_prose_*_postfix.txt`.

## Status

- Narrowing: analysed, **not built**, and should not be. Task #18.
- Refilling: **not scoped**. The ring question above is the first thing to
  settle.
- The straggler cost is documented in PR #38's Known gaps with these numbers.

## Reproducing

    scripts/tts/bench_longform_prose.py \
      --bin build/cuda-speech/bin/nemo-speech \
      --baseline /path/to/stock/nemo-speech \
      --magpie MAGPIE.gguf --codec CODEC.gguf --tokenizer TOKENIZER_DIR

Per-chunk step counts, from any verbose run:

    grep -oE 'chunk=[0-9]+ attention-prior step=[0-9]+' run.log \
      | sed 's/chunk=//;s/ attention-prior step=/ /' \
      | awk '{if($2>m[$1])m[$1]=$2} END{for(c in m) print c, m[c]}' \
      | sort -k2 -n | tail
