# MagpieTTS wave scheduler — state and how to resume

Last updated 2026-09-10. Branch `perf/upstream-best` @ `be9eadc`, PR
NVIDIA/NeMo-Speech.cpp#38 (draft, 36 commits), based on stock `a5b6953`.

## What the branch does

Decodes long-form text chunks in lockstep "waves" instead of one at a time,
with five things batched: the opening prefill (one graph per group), the
persistent decoder (item axis), the local transformer (batch axis,
round-major codes), cross-attention (one padded K/V arena plus an
additive mask), and the sampler. Opt-in through
`--tts.batch-size N --tts.longform-history-tokens M` (both required
together). Default `batch-size 1` leaves the sequential path bit-identical.

Reference hashes for the sequential path, current as of `a905119`:

    line       0afc4e3a87b69463
    paragraph  b108af49283f35d8
    script     2dbd3bfb55596e25

These are a change-detector against our own top-of-tree, not a reference
oracle. They were re-baselined twice on purpose: once when the decoder took
flash attention, once when the local transformer did.

## Results

Twenty-sentence script, greedy, median of five, GPU idle-gated.

| machine | stock | ours, best | speed-up |
|---|---|---|---|
| grandteton, GB300 sm_103, CUDA 13.2 | 0.0402 | 0.0083 (w32 cf32) | 4.8x |
| sparky, GB10 sm_121, CUDA 13.0 | 0.0668 | 0.0263 (w16 cf16) | 2.54x |

GB10 before/after the batched prefill, twenty-sentence script, median of
five, both arms built and benched in one session:

| config | before | after |
|---|---|---|
| w16 cf4 | 0.0365 | 0.0310 |
| w16 cf16 | 0.0317 | **0.0263** |
| w32 cf16 | 0.0294 | 0.0319 |
| w32 cf32 | 0.0295 | 0.0322 |

The w32 rows are **not** a width cliff -- see the straggler section below.
GB10's optimum is width 16; GB300's is still width 32.

Time to first audio matches stock on both and does not vary with wave
width.

Full grandteton grid, twenty-sentence script, RTF / end-to-end TTFA:

| wave width | cf=4 | cf=8 | cf=16 | cf=32 |
|---|---|---|---|---|
| 1 (no wave) | 0.0359 / 30 ms | 0.0348 / 36 ms | 0.0344 / 47 ms | 0.0346 / 69 ms |
| 4 | 0.0211 / 30 ms | 0.0180 / 36 ms | 0.0162 / 47 ms | 0.0154 / 69 ms |
| 8 | 0.0180 / 30 ms | 0.0144 / 36 ms | 0.0126 / 47 ms | 0.0117 / 69 ms |
| 16 | 0.0167 / 30 ms | 0.0129 / 36 ms | 0.0109 / 47 ms | 0.0100 / 69 ms |
| 32 | 0.0151 / 30 ms | 0.0114 / 36 ms | 0.0093 / 47 ms | **0.0083 / 69 ms** |

Two hundred sentences (1013 s of audio), chunk-frames 32, peak device
memory net of other tenants:

| width | RTF | real time | peak |
|---|---|---|---|
| 32 | 0.0066 | 152x | 3.4 GiB |
| 64 | 0.0058 | 172x | 4.8 GiB |
| 128 | 0.0054 | 186x | 7.8 GiB |
| 200 | 0.0051 | 196x | 11.1 GiB |

The wave is worth less on GB10 because the codec binds sooner (stock codec
RTFx 23 against 69) and chunk width saturates at 16 rather than 32. GB10 is
decoder-bound at its best setting: decoder RTFx 48.7 against codec 99.2.

## Running the benchmark

Same driver script on both machines; it inlines its own text, so a run is
reproducible from the script plus a binary and the models.

    # grandteton (this box)
    cd ~/devel/NeMo-Speech.cpp/.claude/worktrees/upstream-main
    M=~/.cache/nemo-speech/models/nvidia
    python3 scripts/tts/bench_magpietts.py \
      --bin build/cuda-speech/bin/nemo-speech \
      --magpie $M/magpie_tts_multilingual_357m/452ef560f972c38d5fc16476259aac9456453547/magpie_tts_multilingual_357m.v2602.f16.gguf \
      --codec  $M/nemo-nano-codec-22khz-1.89kbps-21.5fps/fc00890b604aa2de298d2641ffc6c5f6caf8c4d7/nemo_nano_codec_22khz_1.89kbps_21.5fps.decoder.f16.gguf \
      --tokenizer ~/nemo-bench-assets/tokenizer-v2602 \
      --greedy --reps 5 \
      --extra --tts.batch-size 16 --tts.longform-history-tokens 20

Two traps when A/B-ing two builds on this box:

- **The Bash cwd resets to the main repo between calls.** A bare
  `git checkout <ref> -- src/...` then reverts the wrong tree and the
  relative `cmake --build` finds nothing, so the run silently uses the
  binary already there. Start every command with an explicit `cd` to the
  worktree.
- **`bin/nemo-speech` is a thin CLI over `libnemo_speech_tts.so`**, so
  copying it aside does not snapshot an arm. Prove which arm is live:
  `nm -D --defined-only build/cuda-speech/bin/libnemo_speech_tts.so.1 | grep -c prefillWave`
  (1 = with the batched prefill, 0 = without).

Drop `--extra ...` for the sequential arm. `--greedy --check` replaces
timing with sha256 reproducibility hashes; that is the gate to run after
every change. `--device auto` is needed off CUDA (it also skips the
nvidia-smi idle gate, which only runs when the device starts with "cuda").

On sparky the binaries live elsewhere and the preset differs:

    ssh ryan@sparky
    cd ~/devel/ours            # our branch
    python3 scripts/tts/bench_magpietts.py \
      --bin ~/devel/ours/build/cuda-tts/bin/nemo-speech ...        # ours
      --bin ~/devel/NeMo-Speech.cpp/build/cuda-tts/bin/nemo-speech # stock

Helper scripts already on sparky, written during the deploy:
`~/gb10w.sh` (wave width sweep), `~/gb10c.sh` (stock vs ours across chunk
widths), `~/gb10f.sh` (width sweep reporting decoder/codec RTFx),
`~/agg.py` (median aggregator the others pipe into), `~/diag.sh` (CUDA
graph composition diagnostics). They embed the model paths.

## Sparky state — ready to use, nothing to redo

- `~/devel/NeMo-Speech.cpp` at stock `a5b6953`, built: `build/cuda-tts/bin/nemo-speech`
- `~/devel/ours` worktree at `d5ba9e3`, ggml patch series applied, built:
  `build/cuda-tts/bin/nemo-speech`. Tree clean.
- Models at `~/.cache/nemo-speech/models/nvidia/` (magpie 448M, codec 76M).
- Tokenizer at `~/nemo-bench-assets/tokenizer-v2602` (9.3M; the 1.2G
  `model_weights.ckpt` was excluded and is not needed).
- `/tmp/script.txt` holds the twenty-sentence benchmark text.

Five environment quirks, all already worked around. Repeat them on a fresh
checkout:

1. **git-lfs is not installed.** The global gitconfig sets
   `filter.lfs.process`, which overrides smudge/clean. Set the local
   override: `git config --local filter.lfs.process ""` plus
   `filter.lfs.smudge cat`, `filter.lfs.clean cat`,
   `filter.lfs.required false`. Without this every checkout dies with
   "the remote end hung up unexpectedly".
2. **Untracked files survive a `git checkout`.** The first clone landed on
   a later commit and left `ggml-patches/0002-nvfp4-warp-quantizer.patch`
   behind; the patch script then applied it and failed. `git clean -fd`
   after changing refs.
3. **No ninja**, and the `base` preset requires it. `python3 -m pip` is
   PEP-668 blocked, so: `python3 -m venv ~/.venv-build && ~/.venv-build/bin/pip install ninja`,
   then put `~/.venv-build/bin` on PATH.
4. **nvcc is not on PATH.** Pass `-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc`.
5. **Use the `cuda-tts` preset, not `cuda-speech`.** The latter pulls in ASR
   and needs SentencePiece, which is not built there. TTS does not.

Full configure line:

    export PATH=$HOME/.venv-build/bin:/usr/local/cuda/bin:$PATH
    bash scripts/configure.sh cuda-tts \
      -DCMAKE_CUDA_ARCHITECTURES=121 \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
    cmake --build build/cuda-tts -j 20

`CMAKE_CUDA_ARCHITECTURES=121` is GB10; grandteton uses 103. **Binaries are
not portable between the two** — different SASS, no fallback PTX, and the
kernels simply will not launch. Build on each box.

`tests/cpp/tts/test_magpietts_asr.cpp` fails to compile under the TTS
presets (missing `recognizer.h`). Pre-existing, predates this branch, and
only bites with `-DNEMO_SPEECH_BUILD_TESTS=ON`. Build specific targets or
leave tests off.

## Finding: why the local transformer's CUDA graph will not compose on GB10

Wave widths above 8 used to fail the whole run on GB10. Two separate things,
and only the first is architectural:

    GB300 width 8    child graph n=24  t0=24                  composes
    GB300 width 12   child graph n=24  t0=24                  composes
    GB10  width 8    child graph n=24  t0=24                  composes
    GB10  width 12   child graph n=42  t0=36 t10=3 t11=3      cudaErrorNotSupported

Node types 10 and 11 are `MemAlloc` and `MemFree`. CUDA does not allow a
child graph containing memory allocation nodes to be embedded with
`cudaGraphAddChildGraphNode`.

It is **not** the batch threshold. Same source, same batch, same ggml:
GB300's graph does not change across that boundary at all, while GB10's
gains twelve kernels and three alloc/free pairs. ggml selects a different
matmul path per architecture — `ggml_cuda_should_use_mmq` takes the compute
capability — and the two boxes also carry different cuBLAS versions (13.2
against 13.0) whose heuristics pick algorithms with different workspace
needs. Only the sm_121 path allocates, and only allocation makes the graph
unembeddable. Worth retesting if sparky ever moves to CUDA 13.2.

**Sub-chaining cannot fix this.** The *first* add fails, so it is not a
cumulative limit; any sub-chain containing that round carries the same alloc
nodes. `GGML_CUDA_NO_VMM=1` does not remove them either (verified: still
n=42 with t10/t11).

The second thing was our bug and is fixed in `d5ba9e3`: the failed graph
call left a sticky CUDA error, so the eager fallback inherited it and failed
too, turning a missing optimisation into a dead run. Clearing it makes the
fallback test itself. Not worth chasing further — eager at width 12 (0.0316)
already beats composed at width 8 (0.0342), so the wider batch more than
pays for losing the composed graph.

## Done: the opening prefill is batched (`72d8cd2`)

Every chunk used to open through the single-item path -- `evalCachedPair`
into a fresh pair of full-size K/V caches, a host round trip to gather
each column's opening hidden state, then `seed()` copying those caches
into the wave arena. Measured 195 ms over 20 chunks, 19% of the producer.

Now one graph opens a whole group. Every chunk's baked context is the
same length, so the prefill's sequence is uniform across the batch; only
the text differs, and the padded cross arena already covered that. The
self-K/V are written straight into the ring the steps append to.

Three things fell out rather than being replaced: the per-chunk staging
caches (288 MiB a chunk for the guidance pair, which is why the scheduler
had to release them mid-run), the seed graph for waves, and the host
round trip.

Measured on grandteton, both arms built and benched in one session:

    opening cost, 20 chunks     195 ms / 20 calls -> 30 ms / 3 calls
    script, w16 cf16            0.0117 -> 0.0109
    script, w32 cf32            0.0093 -> 0.0083
    200 sentences, w32 cf32     0.0075 -> 0.0066
    peak device memory, ditto   11.2 GiB -> 3.4 GiB

Memory is the interesting one. Wave width used to be capped by it; width
now costs a third of what it did, so width 128 fits in less memory than
width 32 used to and runs 27% faster.

Things worth knowing if this is revisited:

- The prefill's attention reads the F32 projection it just produced, not
  the F16 ring, which is what the single-item prefill did. Only the ring
  write converts.
- The padded cross arena is now a property of being a wave rather than of
  width, so a one-chunk group (chunk 0) takes the same path as a wide one.
- Both the prefill and the persistent decoder assume `dec_kernel == 1`:
  `causal_conv1d` shifts along the axis those graphs use for the batch.
  Every shipped checkpoint has kernel 1; `be9eadc` makes the gate say so.
- Per-chunk cross-K/V construction is still serial, and is not worth
  chasing: 10 ms over 21 calls.

## Finding: a degenerate chunk costs `extra_steps x group_width`

Magpie occasionally produces a chunk whose attention does not reach its
text end for a long time: ~291 steps where a normal chunk finishes in
115-145. This is **pre-existing** -- it shows up in both arms -- and which
chunk it lands on moves with any change to the arithmetic.

Sequential decoding pays that once. A wave pays it times the group width,
because lockstep holds every other item until the straggler finishes.

Measured on GB10, batched-prefill arm, 20-sentence script, width 32
(groups of 1 and 19): chunk 7 ran to step 291 while every other chunk
finished by 142. Total steps 218 -> 361, which is the whole of the 9%
"regression" at that width. Per-chunk maxima:

| arm | input | width | mean steps | max |
|---|---|---|---|---|
| before | 20 sentences | 16 | 115 | 142 |
| before | 20 sentences | 32 | 116 | 146 |
| before | 24 sentences | 16 | 121 | **291** (chunk 17) |
| before | 24 sentences | 32 | 115 | 144 |
| after | 20 sentences | 32 | -- | **291** (chunk 7) |
| after | 24 sentences | 16 | 122 | **291** (chunk 22) |
| after | 24 sentences | 32 | 120 | **264** (chunk 2) |

Both arms produce stragglers; neither does so systematically. On the
24-sentence input the batched-prefill arm shows no width-32 penalty at all
(0.0285 at w16 against 0.0290 at w32).

**Never quote a width recommendation from one input.** Check per-chunk
step counts first:

    grep -oE 'chunk=[0-9]+ attention-prior step=[0-9]+' run.log \
      | sed 's/chunk=//;s/ attention-prior step=/ /' \
      | awk '{if($2>m[$1])m[$1]=$2} END{for(c in m) print c, m[c]}' \
      | sort -k2 -n | tail

## Listening pair

`docs/development/tts-wave-scheduler/data/prefill-audio/` holds the twenty-sentence script
from both arms:

    script-before-w16.wav   100.82 s
    script-after-w16.wav     99.52 s
    script-before-w1.wav      99.85 s   sequential reference
    script-after-w1.wav       identical to the line above, byte for byte

The width-1 pair being byte-identical is the strongest statement that the
sequential path is untouched. The w16 pair is the one to listen to; the
batched arm now lands closer to the sequential reference in duration than
the serial-prefill arm did.

## Straggler chunks and continuous batching

See STRAGGLERS.md. Short version: a wave pays for its slowest member, which
costs almost nothing on the inlined benchmark (max/mean 1.26) and a great deal
on real prose (1.9 to 3.4). Retiring finished items is worth 3-6% and is not
worth building; admitting the next pending chunk into the freed lane is worth
~60% of decode and ~37% end to end, but needs per-item ring positions and
per-item masks. The projection script and the measured step counts are beside
that doc.

## Other known gaps, ranked

1. 35 lines of unreachable per-item alignment stacking in `decoder.cpp`
   (~820-856) — the `items > 1` branch returns early, so the loop, the
   `ggml_pad` and the `ggml_concat` can only ever run with one item.
2. `stream_magpie_to_audio` is ~1039 lines; the wave block within it is a
   self-contained ~445 that would lift cleanly into its own function.
3. No test for `plan_text_chunk` (pure, trivially testable, and its two
   copies had already silently diverged before they were merged) or for the
   local transformer's chain-invalidation triple, which three bugs came from.
4. The sequential path silently falls back to the non-persistent decoder on
   the last step of the position budget; the ring is one slot short of it.
   A perf cliff, not a correctness bug. The wave gets the extra slot because
   it has no fallback.
5. Commit history still needs squashing: `ab01f17` describes a broken
   intermediate ("decodes wrong"), `a905119` carries a claim that `ed14876`
   corrects, and there are four `[pre-commit.ci]` commits.
