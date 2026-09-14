# Porting zero-shot voice cloning onto the serving branch

Standalone by convention: zero-shot support stays in its own commits, and the
general development notes do not link here. Nothing in
`docs/development/README.md` should point at this file.

Status: **landed**, except for one piece. Zero-shot checkpoints run, and clone
from a reference WAV, on the sequential path. The wave refuses them with a
diagnostic rather than running them, because its conditioning is still
decoder-wide. See "What is left" at the end.

## What to port

Two commits on `tts/zero-shot-cloning`, which branched from `4f96762` and has
since diverged by ~104 commits:

| commit | |
|---|---|
| `61c8ddb` | `feat(tts): support zero-shot MagpieTTS checkpoints` — the conditioning path, 28 files, +2053/-165 |
| `bf3126c` | `feat(tts): clone a voice from a reference WAV` — the codec encoder wired in, 9 files, +193/-47 |

The other ~100 commits on that branch are general perf and docs work and are
**not** wanted here.

## What is already present

Frame stacking landed on this branch separately:
`stacked_audio_codebooks()`, `magpietts_frames_to_emit`, and
`tests/cpp/tts/test_magpietts_frame_stacking.cpp` all exist.

Missing: `context.cpp` / `context.h`, `emit_codebooks`, `context_prefix`,
`context_duration_max`, `has_context_encoder`, and the conversion changes.

## What the merge actually costs

`git cherry-pick -n 61c8ddb` gives 18 files clean and 10 conflicted, 53 markers.
The taxonomy matters more than the count:

- **~46 markers are one mechanical rename.** Their `h.emit_codebooks` against
  our `h.stacked_audio_codebooks()`. Both mean codebooks x stacking; theirs is
  read from the weights and is authoritative for a zero-shot checkpoint, ours is
  computed. Take theirs, and make the loader fall back to
  `audio_codebooks * frame_stacking_factor` when the GGUF lacks the key, so
  existing GGUFs keep working. A script that normalises the two spellings and
  takes theirs where that is the only difference resolves 18 of the 24 in
  `decoder.cpp` unattended.
- **4 markers are the real integration**, and they are exactly where the
  voice-cloning notes said they would be: our `h.baked_context_length` against
  their `context_prefix_length(h, prefix)` plus a `context_prefix_valid` guard.
  Keep our `audio_len` (ours is post-stacking, theirs is not) and take their
  prefix call.
- **1 marker** is a forward declaration against a signature change; keep both.

## The part that is not a cherry-pick

`61c8ddb` touches **zero** wave entry points — `git show 61c8ddb --
src/tts/magpietts/decoder.cpp | grep -c 'prefillWave\|evalWave'` is 0. Its
design puts the context prefix on `MagpieDecoder` because it "is constant for a
request".

That is false on this branch. Lanes carry chunks from unrelated requests, so two
sessions can want different reference voices in the same wave, and a
decoder-wide prefix would give one of them the other's voice. The prefix has to
become per-item, threaded through `prefillWave` the way `speaker` already is.

The seam exists and is one line: `magpietts.cpp` sets `slot.speaker =
owner.speaker()` per admission, next to `WaveSession::speaker()`. A context
prefix goes beside it as `owner.context_prefix()`, with the cross-attention
arena machinery already proving that per-item conditioning slices work.

Without this, zero-shot runs only on the sequential path — which is to say, not
with batching, which is the whole of this branch.

## Assets for verification

All present on this machine, under `~/devel/magpie-tts-server`:

- `magpie_tts_weights/Magpie-TTS--val_cer_gt=0.3605-step=1200.ckpt` — the
  zero-shot checkpoint
- `magpie_tts_weights/21fps_causal_codecmodel.nemo` — convert with
  `--with-codec-encoder`; the codec GGUF in the model cache is decoder-only and
  cannot clone from a WAV
- `magpie_tts_weights/config_v3_nostress_fixed.local.yaml` — the sidecar config
  `61c8ddb` takes via `--config-yaml`
- `voices/` — reference audio
- a working Python implementation to compare against

Note the tokenizer warning in `notes/voice-cloning-status.md` on the source
branch: `--tokenizer-dir` must hold **this checkpoint's** dictionaries, which
share filenames with the public checkpoint's but differ in content.

## Verification

- Baked path unchanged: the six wave hashes and three sequential hashes, which
  is what proves the `emit_codebooks` rename did not move any arithmetic.
- Zero-shot prefix against the reference: `scripts/tts/compare-context-prefix.py`
  on the source branch checks the conditioning prefix built from natively
  encoded audio against the Python implementation's, and reports cosine. The
  original port recorded 0.99969.
- Per-session voices: two concurrent requests with different reference WAVs in
  one wave, each matching its own solo run. This is the case the original
  commits never had to consider and the one most likely to be wrong.


## What is left

`MagpieWavePrefillItem` carries `speaker` per item, and its comment says why
that is enough: "A step never re-supplies it -- the baked context lands in this
lane's K/V ring here and stays -- so voices only have to be separable at
admission." The same is true of a computed prefix, which makes this a
prefill-only change rather than a step-loop one.

What the wave prefill does today is read conditioning straight out of the baked
table, `ggml_get_rows(model_.baked_context, speaker_in)`, one row per lane. A
context-encoder checkpoint has no such table -- `model.baked_context` is null --
and its conditioning is a computed prefix held on `MagpieDecoder`, one for the
whole engine. Lanes from different requests would all get whichever prefix was
set last.

So the wave now refuses a context-encoder checkpoint and says so, rather than
faulting on the null table or, worse, handing one request another's voice.

To lift it:

1. Give `MagpieWavePrefillItem` a `const magpietts_context_prefix* context`
   beside `speaker`.
2. In the wave prefill graph, when the checkpoint conditions on a context
   encoder, build the `[n_embd, ctx_len, items]` conditioning block from each
   item's prefix instead of the baked row lookup. Every item in one prefill
   burst already opens at the same length, which is the constraint this needs.
3. `WaveSession` gains `context_prefix()` the way it has `speaker()`, and
   `WaveEngine::admit` sets `slot.context = owner.context_prefix()`.
4. Drop `wave_conditioning_ok`.

The test that matters is the one neither original commit had to consider: two
concurrent requests with different reference WAVs in one wave, each matching its
own solo run.


## Validation, and where it actually stands

Reference captured from the Python implementation with
`scripts/tts/capture-magpie-reference.py` (42 text tokens, 215 context frames,
prefix `[217, 768]`), and compared against ours with
`scripts/tts/compare-context-prefix.py`:

| conditioning route | cosine | verdict |
|---|---|---|
| reference's own context codes | **0.99999851** | context encoder is exact |
| reference WAV through our codec encoder | 0.632 | codec encoder diverges |

**Both numbers are correct and nothing is broken.** The 0.632 row compares
different audio, not different implementations.

NeMo's dataset takes a **random window** of the context audio per item --
`context_duration_min` and `_max` are both 10.0, so the length is fixed and the
offset is not. `magpie_serve` even has `_share_context()` to force one window
across a batch, with a comment saying so. The captured reference is therefore a
random 10 s window of `default.wav`, while the runtime takes the *leading*
`context_duration_max` seconds, which `bf3126c` chose deliberately.

Encoding the very same file with NeMo's own codec gives frame 0 =
`[551 1106 212 1270 250 832 1801 743]`, which is byte-identical to ours. The
captured reference's frame 0 is `[287 1223 566 408 1157 1148 446 929]`: a
different excerpt of the same speaker. Hence valid codes on both sides, 0.1%
agreement at every alignment, and a cloned voice that still sounds right.

The correct validation is the codes route, which supplies the reference's own
window: **cosine 0.99999851**. The port is verified.

Note what this cost to find. Listening said the cloned voice was clearly
conditioned on the reference; the metric said 0.632. Both were true -- FSQ code
indices differ enough to move the prefix a long way while the speaker identity
survives decoding. An earlier reading of the same numbers concluded the opposite,
that the context encoder was at fault, because the two routes were measured
either side of the padding fix rather than together. Measure both arms after
every change, or the comparison says nothing.

## Running a zero-shot checkpoint from text

The runtime needs a tokenizer directory matching the checkpoint's profile. One
for the pre-release zero-shot weights is built at `~/nemo-bench-assets/tokenizer-v2607`
from `magpie_tts_weights/config_v3_nostress_fixed.local.yaml` plus
`assets/tts_dataset_files/`. Six things it has to get right:

1. Asset paths rewritten to `nemo:<basename>`, the files copied in beside the config
2. `language_to_tokenizer_mapping` added -- the checkpoint omits it, NeMo defaults it
3. ...written list-valued, `en: [english_phoneme]`, which the parser requires
4. Block-style YAML; flow style silently collapses 8 of the 15 tokenizers
5. Three `charset_version: 1` fields the checkpoint omits
6. Hindi's two-entry `phoneme_dict` list collapsed to a scalar

Two deviations are recorded in `LIMITATIONS.txt` beside it, neither reachable by
English: pt-BR `locale_specific_punct` pinned false, and Hindi losing its
cmudict fallback. Only fields the checkpoint *omits* were filled; none it sets
were overridden.

Embedding the tokenizer in the GGUF, as the ASR models do, would remove this
whole directory and the mismatch class with it.

## Batching zero-shot: attempted, reverted

Per-item conditioning was plumbed end to end -- `MagpieWavePrefillItem::context`,
a `[n_embd, len, items]` conditioning block built from each lane's prefix
instead of the baked table's row lookup, `WaveSession::context_prefix()`, and
`context_len_` carried on the runtime so the ring arithmetic uses the real
length. Two further gaps surfaced and were fixed: this checkpoint has
`apply_attention_prior = false`, so both the wave prefill and the wave step fed
a prior tensor the graph never built.

It still failed before the first decode step, and informatively: by the time
`MagpieDecoder::evalWave` runs, `wave_runtime_` is already null, so the first
failure is upstream of the step -- most likely `sample()`, since this checkpoint
stacks 2 frames and emits 16 codebooks where every wave measured so far emitted
8.

The work was reverted rather than committed: it moved the **sequential** baked
hashes (`0afc4e3a...` to `60ec3ca8...`), so something in it changes the
non-wave path too. That regression is the thing to find first when picking this
up -- the wave hashes were unaffected, so it is narrow.

Until then zero-shot falls back to sequential decode, which on the 12.5 h book
is roughly 5x slower than the wave: the fallback is a real cost, not a
formality.
