# Continuous batching for the MagpieTTS wave scheduler

Design, 2026-09-11, against `perf/upstream-best` @ `6e52dd3`. Payoff and the
measurements behind it are in STRAGGLERS.md: ~60% of decode time, ~37% end to
end, because cost stops tracking `max(steps)` per group and starts tracking
`sum(steps)`.

## What this replaces

Today a wave decodes a fixed group of chunks in lockstep and retires the whole
group when its slowest member finishes. Continuous batching keeps the lanes
full instead: when an item finishes, the next pending chunk takes its lane.

## Prior art, and what is and is not reusable

**Not reusable: vLLM, SGLang, TensorRT-LLM.** All three provide continuous
batching, but they provide it for models structurally isomorphic to a text LLM.
GEPARD names the three features that make a TTS model unusable on a stock vLLM
engine: a layer-wise depth transformer over codec codes, cross-attention to the
audio interface in intermediate layers, and two-pass classifier-free guidance
at each decode step. MagpieTTS has all three -- our local transformer, our
cross-attention, our CFG pair. Fish Audio S2 gets SGLang's continuous batching
free precisely because their architecture lacks them. We cannot buy this.

**Reusable: llama.cpp, vendored in this repo, on our ggml.**
`llama.cpp/src/llama-kv-cache.cpp` implements the pattern with the same
primitives and the same CUDA graph machinery:

- `n_stream` -- independent sequences sharing one batch
- per-token `pos[i]` and `seq_id[i]` in the ubatch
- `set_input_kq_mask_impl` -- a `[n_kv, n_tokens]` mask, skipping a cell when
  it is empty, belongs to another sequence, or is causally in the future
- `n_pad` on the cache, keeping `n_kv` shape-stable as composition changes

The useful confirmation is the last one: fixed shape, changing contents, is how
they keep captured graphs valid under rolling admission. That is what we
concluded independently from ggml's flash-attention asserts.

**Our problem is simpler than theirs.** llama.cpp shares one cell pool across
sequences, so it needs cells tagged by `seq_id`, defragmentation, and paging.
We statically partition: lane `l` owns slab `l * cache_len_`, one token per item
per step, and the opening prefill is already a separate graph. The lane *is* the
sequence. We need no cells, no `seq_id`, no paging, no defrag -- only per-lane
positions. The cost of that simplicity is that memory is `lanes * cache_len_`
regardless of utilisation, which is the 3.4 GiB already measured at width 32.

## What changes

### 1. Per-lane ring head -- trivial

`write_step_state()` already writes a row index per lane:

    write_rows[lane] = lane * cache_len_ + ring_head_;

It becomes `ring_head_[item_of(lane)]`. Same I64 tensor, same shape, different
contents. No graph implication.

### 2. Per-lane attention mask -- supported natively

`fa_mask` is `[cache_len_, kMagpieKqMaskPad]` today, one mask shared by every
lane because every lane has the same ring head and the same valid length.

It becomes `[cache_len_, kMagpieKqMaskPad, 1, lanes]`. `ggml_flash_attn_ext`
asserts only

    q->ne[2] % mask->ne[2] == 0
    q->ne[3] % mask->ne[3] == 0

and q is `[d_head, n_q, n_head, lanes]`, so `ne[2]=1` broadcasts over heads
while `ne[3]=lanes` gives each lane its own mask. Shape is fixed; contents
change per step, exactly as now.

Memory: `cache_len_ * 64 * lanes * 2` bytes -- about 12 MB at `cache_len_ 1500`,
`lanes 64`. The incremental upload becomes `lanes * 2` bytes per step instead of
2, which is still nothing against the 78 KB it replaced.

### 3. Per-item position -- shape change, fixed size

`decoder.cpp:676` reads "Items step in lockstep, so one position row broadcasts
across them" and the input is `{1}`. It becomes `{items}`, and
`ggml_get_rows(pos_emb, position)` yields a per-item row. Fixed at `items`, so
still graph-stable.

### 4. The local transformer needs no change

`local_transformer_forward_cached_fixed_pos` takes `n_past` as the *codebook
round within a frame*, not the decoder step, and its position embedding resets
every frame. Every item is at the same round of every step regardless of which
decoder step it occupies. The LT is step-agnostic and its composed CUDA graph is
unaffected.

### 5. Sampler seed

Currently `round * width + item`, which is stable only because a slot means the
same chunk for a group's lifetime. Under rolling admission it must key off the
item's own frame counter: `seed_for(item) = hash(item_frame_index, round)`.
Greedy output is unaffected; sampled output changes, which needs stating but is
not a regression.

## The three real problems

### A. Cross-K/V arena sizing

The arena is `[n_layers * text_len_ * cross_dim, items]` where `text_len_` is
the *group's widest* text. Under rolling admission a newly admitted chunk may
have longer text than the arena was built for.

Options, in order of preference:

1. **Size to a configured maximum chunk text length**, not the group's widest.
   Chunking already bounds text length; a 256-token cap costs about
   `12 * 256 * cross_dim * 4 * 2` bytes per item, roughly 19 MB at
   `cross_dim 768`, or 600 MB at 32 lanes. Acceptable, and it makes the arena
   admission-independent.
2. Refuse admission for a chunk longer than the arena and let the lane idle
   until the runtime is re-formed. Simple, correct, occasionally wasteful.

Sizing to `n_ctx` is not an option: 2048 tokens would be ~4.8 GB at 32 lanes.

`fill_wave_cross` must also learn to rewrite **one item's slice** rather than
the whole arena. It is already a per-item loop over layers, so this is a
narrowing of its bounds, not a rewrite.

### B. In-order streaming

The codec worker is a single in-order stream over a serial convolution state.
Today chunk order is guaranteed because groups retire in order. Under rolling
admission chunk N+1 can finish before chunk N.

This needs a reorder buffer: completed chunks hold their frames until every
earlier chunk has drained. Bounded by the number of lanes, so memory is
bounded. The head-of-line chunk still streams frame by frame, which is what
keeps first audio early -- that property survives, but its implementation moves
from "the head of the group" to "the oldest live chunk".

### C. Prefill batching regresses

The opening prefill is currently one graph over a whole group, which is what
took it from 195 ms over 20 chunks to 30 ms over 3. Under rolling admission
lanes free one at a time, so prefills would be one at a time -- losing most of
that win.

Mitigation: accumulate freed lanes and prefill in batches when several are
available, admitting in bursts rather than singly. This trades a little lane
idleness for prefill batching, and the right batch size is an empirical
question. **This is the one place where continuous batching makes something
worse, and it should be measured before the design is called done.**

## Work estimate

| piece | where | lines |
|---|---|---|
| Per-lane ring head, mask, position | `decoder.cpp` module + runtime | ~120 |
| Cross arena: fixed max text, per-item refill | `decoder.cpp` | ~90 |
| ~~Survivor history copy, arena to arena~~ | done, `bf21435` | ~85 |
| Subset prefill: per-lane arena writes, offset cross view | `decoder.cpp` | ~70 |
| Burst re-form: retirement, admission, threshold | `magpietts.cpp` | ~150 |
| Reorder buffer for in-order streaming | `magpietts.cpp` | ~80 |
| Batched admission to keep prefill batched | `magpietts.cpp` | ~60 |
| Tests | `tests/cpp/tts` | ~80 |

About **580 lines**, two to three days including measurement. Higher than the
275 that narrowing would have cost, for roughly ten times the payoff.

## Where this lives

**Branch `perf/continuous-batching`, not PR #38.** #38 is closed to new work: it
is out of draft, under review, and its scope is the wave scheduler as it stands.
Groundwork for a feature that ships later does not belong in it -- a reviewer
would rightly ask why the mask grew a lane axis for a feature that is not there.

Step 1 is already on that branch (`34aa8a3`): per-lane ring head, per-lane mask,
per-item position. It is behaviour-preserving -- the three sequential hashes and
the wave's output are byte-identical, and RTF is unchanged at 0.0083 for width 32
/ chunk-frames 32 -- and it answers the question that gated this whole design:
**a captured CUDA graph survives per-lane ring state**, because every shape stays
fixed and only contents move.

One implementation note from that step. Turning on one slot per lane is a strided
scatter; doing it as one `ggml_backend_tensor_set` per lane costs 4.8% of
end-to-end at width 32, sixty-four small synchronous transfers a step.
`ggml_backend_tensor_set_2d` does it in one call and restores the number exactly.
`cudaMemcpy2D` rejects a zero source pitch, so the source must be one value per
lane rather than one value re-read.

The branch rebases onto #38 as that merges.

## Admission: re-form in bursts, do not prefill into a live runtime

The obvious reading of "admit a chunk into a freed lane" is to prefill that one
chunk directly into lane `i` of a runtime that is mid-flight. That is possible
but surgical: the prefill graph would need to write two non-adjacent lane slabs
(`i` and `items + i`), read the cross arena at item `i`'s slice rather than
slice 0, and write one column of the hidden pair. Every one of those is an
offset the current graph takes for granted.

There is a much cheaper route that reuses everything already built. Re-form the
runtime when **enough** lanes are idle, not when one is:

- a re-form costs ~12 ms, measured. Admitting singly on a 150-chunk input is
  ~118 admissions, or 1.4 s -- which eats the entire ~2 s saving.
- admitting in bursts once a quarter of the lanes are free is ~19 re-forms, or
  ~230 ms against a ~2 s saving. That is affordable.

And a burst re-form needs no new graph: it is `prefillWave` over the newly
admitted chunks, plus an arena-to-arena copy carrying the survivors' history
into the new runtime. `cache_len_` is identical between the two, so whole lane
slabs copy with no index arithmetic -- the same ~60 lines the abandoned
narrowing design needed, which is the one piece of that work worth keeping.

So the shape is: run the group until a burst threshold of lanes is idle, re-form
with survivors plus new chunks, continue. Lanes idle briefly between bursts,
which is the price of not writing a single-item prefill path.

This also fixes the prefill-batching regression on its own: admissions arrive in
batches, so the batched prefill stays batched. Problem C above largely
dissolves.

## The remaining obstacle: prefilling a subset of lanes

A burst re-form prefills only the newly admitted chunks, because the survivors
already have history and it is copied across. So the prefill graph has to write
*some* lanes, not all -- and that runs into two offsets it currently takes for
granted.

1. **Arena writes.** The prefill writes K and V as one `ggml_cpy` per plane with
   a `[n_embd, total_len, lanes]` view. A non-contiguous lane subset cannot be
   expressed that way. Splitting it into one copy per lane fixes it -- 2 planes x
   2k lanes x n_layers copies, about 384 at k=8, which is nothing for a one-shot
   graph.
2. **Cross-attention.** `cross_attention_wave` reads item `i`'s slice of the
   padded arena through the `ne3` axis. A scattered subset has no single view.

Problem 2 is the real one, and the fix is scheduling rather than graph surgery:
**place newly admitted chunks in a contiguous item range.** The mapping from old
lanes to new is ours to choose, so put survivors first and admissions last. Item
range `[a, a+k)` becomes lane runs `[a, a+k)` and `[items+a, items+a+k)` -- two
contiguous runs, and the cross arena view is then just an offset of
`a * item_stride` with `ne3 = k`.

Prefilling *all* lanes and overwriting survivors afterwards does not work, even
as a slow first cut: the prefill requires every item to be at the same audio
length, and survivors are mid-decode by definition.

## Order of work

1. ~~Per-lane ring head, mask and position, with the group still fixed.~~ Done,
   `34aa8a3`. Gates byte-identical, RTF unchanged, and the CUDA-graph question
   is answered: fixed shapes with moving contents keep the capture valid.
2. Cross arena to a fixed maximum text length, still per group. Again no
   behaviour change.
3. Admission and retirement, single-chunk. First real behaviour change;
   expect the straggler win to appear here.
4. Reorder buffer. Required for correctness of 3 in the general case.
5. Batched admission, measured against 3.

Steps 1 and 2 are safe refactors that can land before any scheduler change, and
they carry the CUDA-graph risk. If they hold the gates, the rest is scheduling.

## Gates

Unchanged from the rest of this branch: the three sequential hashes, wave audio
duration, every chunk reaching the end of its text. Plus, new for this work:

- `bench_longform_prose.py` at 32/64/128 sentences, where the payoff lives
- per-chunk step counts, to confirm the group no longer pays for its slowest
  member: the current-versus-projected numbers are in STRAGGLERS.md

## Post-fix numbers

The open question below is now settled, and the answer strengthens this design
rather than weakening it. The straggler's *source* turned out to be a port
error, not the sink thresholds: the prior held position when its lookahead
window emptied near the text end, where the reference jumps to the final token.
PR #44 fixes it. Mean steps per chunk fall ~14% while `max` barely moves, so
`max/mean` rises from 1.90/3.29/3.37 to 2.02/3.61/3.72 and the refill saving
rises with it, to 60.8/65.8/62.8% of decode. Projected end to end: 150x, 152x,
156x realtime against 112x, 106x, 113x today.

Build this against post-fix step counts (`steps_prose_*_postfix.txt`), not the
originals.

## Settled question

Fixing the straggler at its source came first, and it was worth doing: see
PR #44. It was a missing branch, not a threshold. The sink thresholds we ship
(8 and 10) remain unexplained -- they are defaults our conversion script
invented for keys that exist in no NeMo config, and the installed NeMo 3.1.0
hardcodes 4 at both sites -- but with the empty-window branch restored they no
longer drive the pathology, so changing them is no longer urgent. Worth
revisiting on its own.

---

# Results

Measured 2026-09-12 on grandteton (GB300, aarch64, CUDA 13.2, compute 10.3),
GPU idle, greedy (`--top-k 1`), `--tts.chunk-frames 32`, medians of 3 unless
noted. Every arm was verified by symbol before it was measured -- `nemo-speech`
is a thin CLI over `libnemo_speech_tts.so.1`, so a stale library is invisible
otherwise.

Ten commits on `perf/continuous-batching`, from `4d468a2`:

| | |
|---|---|
| `81c98c7` | prefill a chosen set of lanes into a live wave runtime |
| `bbdfe76` | forbid EOS per item in the batched sampler |
| `ae6f478` | hold one wave runtime for the whole run |
| `4b1f62e` | admit chunks as lanes free |
| `670af88` | admit at the wave's ring head, and tune the burst |
| `c8e2ac6` | cover the wave admission policy |
| `9232810` | report wave occupancy, not just idle lane-steps |
| `24b22c5` | prefill only the lanes it opens |
| `e3bcaea` | cap wave lanes below the chunk count |

## Headline

Three configurations, each at its own best width. `baseline` is pre-#38
(`a5b6953`, the merge-base with origin/main) running stock -- no batching flags,
adaptive history -- which is how `bench_longform_prose.py --baseline` measures.
`wave` is `4d468a2`. `CB` is the tip. Inputs are the three inlined cases the
branch has always gated on, plus book-corpus prose sampled by
`bench_longform_prose.py`'s seeded pool.

| input | chunks | baseline | wave (best) | vs baseline | CB (best) | vs wave | total |
|---|---|---|---|---|---|---|---|
| line | 1 | 23.7x | 26.3x (w32) | +11% | 25.8x | -2% † | 1.09x |
| paragraph | 5 | 25.9x | 56.6x (w32) | +118% | 55.6x | -2% † | 2.15x |
| script | 20 | 25.9x | 120.8x (w32) | +367% | 119.2x | -1% † | 4.61x |
| prose 16 | 17 | 25.9x | 86.6x (w32) | +234% | 85.3x | -1% † | 3.29x |
| prose 32 | 38 | 25.7x | 106.7x (w32) | +315% | **121.9x** (w128) | **+14%** | 4.74x |
| prose 64 | 75 | 25.7x | 113.4x (w128) | +342% | **122.0x** (w128) | **+8%** | 4.75x |
| prose 128 | 150 | 25.6x | 118.9x (w128) | +364% | **153.2x** (w128) | **+29%** | 5.98x |
| prose 256 | 291 | 25.5x | 113.5x (w128) | +345% | **163.3x** (w128) | **+44%** | 6.41x |
| prose 2048 | 2377 | 25.6x | 138.0x (w128) | +440% | **196.9x** (w128) | **+43%** | **7.71x** |

† The noise floor. `line` forms no wave in either arm -- it is the same code --
and still moves 2% between runs.

Where continuous batching engages (>= 38 chunks): **+27% over the wave, 5.9x
over baseline**, reaching 7.7x at 2377 chunks.

Compare against the wave's *best* width, not a fixed w32: the wave gains from
wider batches too (88.3x -> 113.4x at 75 chunks). Comparing CB's best against
wave-at-32 would report +36% instead of +27%.

The baseline column produces ~6% more audio for the same text, because stock
uses adaptive history while both wave arms pin it at 20. xRT normalises for that.
It also cannot run prose at all without one backported fix (below).

## Scaling

At width 128, the plateau is ~198x and is within 3% of it by ~1200 chunks.

| chunks | audio | CB w32 | CB w128 | occupancy w128 | RSS |
|---|---|---|---|---|---|
| 291 | 26 min | 154.6x | 164.0x | 54.2% | 1.3 GB |
| 593 | 54 min | 160.1x | 181.6x | 70.2% | 1.7 GB |
| 1190 | 1h 47m | 161.9x | 191.8x | 81.5% | 2.9 GB |
| 2377 | 3h 32m | 165.4x | 196.6x | 87.6% | 5.8 GB |
| 4759 | 7h 04m | 153.3x | 198.2x | 91.0% | 13.2 GB |

e2e tracks `decoder_rtfx` to within a percent at every size once lanes are full,
and the codec sits at ~345x throughout -- more than 2x headroom -- so **the
decoder is the limiter**, not the codec and not the host. The w32 figure at 4759
chunks is a single rep and its codec number drops with it (327x against 345x);
treat it as suspect rather than as a regression.

Host memory is O(chunks) at ~2.7 MB each -- `plan` holds every chunk's state for
the life of the request. That is pre-existing and CB makes it substantially
better, not worse: the pre-#38 sequential path costs ~8.1 MB per chunk, 5.8 GB
at 594 chunks against CB's 1.7 GB. Still worth revisiting for a server, where
7 hours of audio in one request is 13 GB.

## Width, and why lanes are capped

A wave sized at one lane per chunk admits everything at once and never refills --
exactly the single-cohort behaviour continuous batching exists to remove, paying
`max(steps)` over the whole run. So asking for more lanes than the input has
chunks made things sharply *worse*, and `--tts.batch-size 128` -- the best
setting on a long script -- was a 30% regression on a short one.

Sweeping lanes against a fixed input (at `24b22c5`, before the cap) shows the
shape: more lanes always win on throughput, because a step's cost is mostly
fixed and widening amortises it, until so few chunks are left over that the last
arrivals have nothing to hide behind.

| lanes | 4 | 8 | 12 | 16 | 24 | 32 | 48 | 64 | 128 |
|---|---|---|---|---|---|---|---|---|---|
| 37 chunks | 69 | 95 | 86 | 89 | 111 | **122** | 86 | 86 | 86 |
| 74 chunks | 71 | 101 | 108 | 115 | 121 | **121** | 119 | 111 | 113 |
| 149 chunks | 72 | 106 | 117 | 128 | 140 | 147 | 153 | **155** | 133 |

`e3bcaea` caps lanes at half the chunks above a floor of 32, and never above the
chunk count itself. Half rather than less because parallelism is worth more than
refill at small counts -- 37 chunks prefer 32 lanes to 18 by a wide margin. The
result is that width 128 is now never worse than width 32:

| chunks | w128 before cap | after |
|---|---|---|
| 37 | 86.1x | **121.9x** |
| 74 | 112.7x | **122.0x** |
| 149 | 132.5x | **153.4x** |
| 291 | 163.3x | 163.7x |

**Recommended: `--tts.batch-size 128`.** It is now safe at every input size and
wins from ~150 chunks.

## Tuning

**Burst threshold.** An eighth of the lanes, floored at two. Swept over 2377
chunks at the tip: at 32 lanes, a threshold of 2 gives 162x, 4 gives 165x, 8
gives 163x; at 128 lanes, 2 gives 190x, 4 and 8 give 194x, 16 gives 196x, 32
falls back to 190x. The curve is shallow now -- it spanned 117x to 183x when
every prefill ran the runtime's full width.

**Occupancy is not the objective.** Higher occupancy is reliably *worse*: at
width 128, a threshold of two reaches 92% occupancy and 117x, while a threshold
of thirty-two sits at 77% and reaches 183x. Admission costs a prefill, and a
prefill costs more than the idle lanes it saves. The verbose line reports
occupancy so this is visible rather than assumed.

**Prefill cost** was ~0.55 ms per lane *computed* and nothing per lane *opened*,
because the graph ran the runtime's full width and wrote back only the lanes it
was opening. `24b22c5` narrows it to the admitted set:

| | mean admit | total | share of run | xRT |
|---|---|---|---|---|
| width 32, before | 17.7 ms | 10.1 s | 12.2% | 153.6x |
| width 32, after | 7.7 ms | 4.4 s | 5.7% | **165.1x** |
| width 128, before | 64.3 ms | 8.9 s | 12.7% | 181.2x |
| width 128, after | 24.6 ms | 3.4 s | 5.2% | **196.4x** |

GPU peak at width 128 *fell* 15.0 -> 12.6 GiB despite the second cross arena,
because the prefill's compute buffer now scales with what it opens.

**The per-step mask upload is not worth moving into the graph.** Measured
differentially -- repeat an idempotent upload N times, output byte-identical,
take the slope -- one upload costs 0.0125 s on a 5.85 s run, **0.21%**. An
in-graph version would still need its own index upload, and would put several
writes into a tensor a later node reads, which is the aliasing shape that cost
days on the NanoCodec work. (A first attempt at measuring this said 11%; it was
wrong, because skipping the upload breaks attention and the run produced 55%
more audio to amortise the fixed cost over.)

## Gates

`scripts/tts/greedy-hashes.sh` runs **sequentially** -- it passes no
`--tts.batch-size` -- so it cannot see a wave regression at all. It is still the
right gate for the sequential path, and it did not move across any of this work.
`scripts/tts/wave-hashes.sh` covers the wave: three inputs at two widths,
including a `strict` case whose chunk count is an exact multiple of the width
plus one, so every cohort is full.

**Byte-identity held for the first three commits and cannot survive continuous
batching.** The batched arithmetic depends on which lane a chunk occupies:
permuting only the chunk-to-lane mapping, with schedule, width and ring placement
all unchanged, moves the hash (`3d013abb67935a6e` -> `0f0a1b00197974fc`).
Admission assigns whichever lane is free, so it reassigns lanes by construction.
A chunk admitted mid-flight is bit-identical for its first dozen steps and then
drifts; lanes admitted at open stay bit-identical, so an admission does not
disturb the chunks already decoding.

Gate instead on: reproducibility (same input, same output, every run -- this
holds), audio duration, per-chunk step counts, and the sequential hashes. Where a
change should be identity-preserving, construct the case that makes it so --
`strict` at a full-width multiple, or forcing the packed cross path on a run that
is contiguous, which reads the same numbers from the other place and reproduced
all six hashes exactly.

## What this design doc got wrong

**The burst re-form was unnecessary.** The decode graph reads cross-K/V from the
runtime's *own* padded arena -- `fill_wave_cross` gathers into it -- not from the
chunks' `DecoderCrossKvCache`es, which are only a staging source. So one runtime
lives for the whole run and takes admissions in place: no second runtime, no
~12 ms re-form, no graph recapture, no LT recomposition, and peak KV memory stays
put instead of doubling while both runtimes are alive. `carryHistoryFrom`
(`bf21435`) is consequently dead code.

**Admissions need not be contiguous**, so "place newly admitted chunks in a
contiguous item range" and the survivors-first ordering both go away. What
replaced it changed twice: first computing the full width and narrowing only the
write-back, then (`24b22c5`) narrowing the compute too, with contiguous runs
reading the arena through an offset and scattered ones reading an
admission-ordered copy gathered alongside the real one.

This is also what finally fixed two things `4d468a2` claimed but did not do: the
self-attention shapes still said `lanes_` where the projection said `lanes`, and
the cross arena and mask were passed whole rather than offset. Both were
unreachable while the graph always ran full width.

**Problem C dissolved**, but not because re-forms batch admissions -- because the
threshold does. **Problem B was nearly free**: chunks already buffered their own
frames, so the reorder buffer is the existing flush index promoted from
group-local to run-global.

## Bugs found

- **Pre-#38 cannot synthesise ordinary prose.** `longform history context cache
  is too short: need 20, have 16` -- the history is spliced from the previous
  chunk's encoder output, so a short sentence makes a short chunk and the next
  chunk asks for more than it left behind. Fixed on the branch by `31fb2ce`; the
  baseline column above required backporting it (it does not cherry-pick, as
  `plan_text_chunk` did not exist yet).
- **A chunk that spends its whole position budget failed the run.** The ring
  holds one opening plus the budget and no more, so the decoder refuses the step
  after it; the group loop enforced that bound as its own `step <
  max_decoder_positions` and so never had to say it. Introduced by `ae6f478`,
  fixed by `4b1f62e` -- so `ae6f478` alone is a bisect hazard. It takes a
  degenerate chunk to hit, which is why 64 sentences of prose failed while 128
  passed.
- **Lanes exceeding the chunk count** left lanes with no state for a step to
  read. Caught by the gate sweep before it landed.

## Reproducing

```bash
cmake --build build/cuda-speech -j 20 --target nemo_speech_cli
# prove the arm before trusting a number: nemo-speech is a thin CLI over
# libnemo_speech_tts.so.1, so a stale library is invisible
nm -DC --defined-only build/cuda-speech/bin/libnemo_speech_tts.so.1 | grep -c plan_wave_admission

bash scripts/tts/wave-hashes.sh   . after        # wave gate
bash scripts/tts/greedy-hashes.sh . after        # sequential gate
ctest --test-dir build/cuda-speech -R magpietts

bash scripts/tts/make-prose-corpora.sh           # /tmp/prose<N>.txt, seeded
WIDTH=128 bash scripts/tts/size-table.sh . CB 3 1   # the table above
```

Model and tokenizer locations come from `scripts/tts/bench-env.sh` and are
overridable (`NEMO_SPEECH_MAGPIE`, `NEMO_SPEECH_CODEC`, `NEMO_SPEECH_TOKENIZER`).

The baseline arm lives in a separate worktree, because the branch's ggml patch
series drops `ggml_fused_attn_cached` and pre-#38 will not build against it:
`.claude/worktrees/pre38`, detached at `a5b6953`, with the history clamp
backported but uncommitted.
