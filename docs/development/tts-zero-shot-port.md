# Porting zero-shot voice cloning onto the serving branch

Standalone by convention: zero-shot support stays in its own commits, and the
general development notes do not link here. Nothing in
`docs/development/README.md` should point at this file.

Status: **landed**. Zero-shot checkpoints run, clone from a reference WAV, and
batch through the wave at the same rate the baked checkpoint manages -- the
whole novel in three minutes. What remains is packaging, not capability: see
"What is still missing" at the end.

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

That is what landed, and the seam was the one line it looked like. See
"Batching zero-shot through the wave" below for what had to follow the prefix
through, and what it is gated on.

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
  commits never had to consider and the one most likely to be wrong. **Still not
  run**, and not because of the wave -- nothing can yet name a voice per
  request. See Gap 2.

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

## The checkpoint carries no inference parameters

This one bit cost a whole render. The checkpoint's config has no
`inference_parameters` block at all: `magpie_serve/engine.py` builds
`ModelInferenceParameters` in code. Conversion therefore fell back to defaults,
and the default for `apply_attention_prior` is **false**, with no
`apply_prior_to_layers` and no `estimate_alignment_from_layers` written either.

The prior is what keeps cross-attention walking the text forward. Without it the
decoder loops: it reads the first minute correctly and then repeats itself, and
the repeats are exact -- identical code frames, not merely similar audio. On the
opening 1,200 characters of the novel, with `--top-k 1`:

| | frames | in a repeat of >= 100 frames | audio |
|---|---|---|---|
| baked, sequential | 1,843 | 0% | 85.7 s |
| baked, wave w=16 | 1,799 | 0% | 83.7 s |
| zero-shot, no prior, sequential | 2,248 | **14.3%** | 104.6 s |
| zero-shot, no prior, wave w=16 | 2,826 | **21.7%** | 131.4 s |
| zero-shot, prior on, sequential | 1,604 | 0% | 74.6 s |
| zero-shot, prior on, wave w=16 | 1,610 | 0% | 74.9 s |

Two things to take from the table. The looping was never the wave's -- the
sequential path had it too -- but the wave made it worse, so a bug that had been
tolerable at one lane became obvious at sixteen. And the tell was in plain sight
before anyone listened: the same text took 26% longer through the wave than
through the sequential path, where the baked checkpoint differs by 2%. **A large
duration difference between two decode paths on identical text is a correctness
signal, not a voice-pacing one.** It was dismissed as the cloned voice speaking
more slowly. It was not.

`convert_model.py --inference-yaml` supplies what the checkpoint omits;
`docs/development/magpie-zero-shot-inference.yaml` holds this checkpoint's
values, copied from the server. Conversion now warns when it is about to write a
checkpoint with no prior.

## Batching zero-shot through the wave

Conditioning is now per lane, which is what `speaker` always was and for the
same reason: a step never re-supplies it, so it only has to be separable at
admission. `MagpieWavePrefillItem` carries the lane's prefix,
`WaveSession::context_prefix()` answers with the request's, and the prefill
builds an `[n_embd, context_len, items]` block from them where it used to do
`ggml_get_rows(baked_context, speaker_in)`.

Three things had to follow the prefix through:

- **The ring is sized and stepped from the prefix length**, not from
  `hparams.baked_context_length`. Those are the same number on a baked
  checkpoint and are not on a zero-shot one: the context encoder pads to a
  duration-derived length, 217 positions here against the table's 110. The
  runtime carries `context_len_` and the three `total_len` computations read it.
  The sequential path takes the same argument, which is why it can now hold a
  persistent runtime at all -- its length check had been failing every step and
  silently falling back to the rebuilt-graph path.
- **`apply_attention_prior` is false on this checkpoint.** The wave prefill
  declared the prior as a graph input unconditionally; nothing read it, ggml
  pruned it, and `compute_graph` then failed to bind it by name. Declare it only
  where a layer applies it. (The step needed no guard: its prior is a session
  tensor, which exists whether or not the graph reads it.)
- **Lanes that disagree on conditioning length are refused**, since one ring
  serves them all. In practice they cannot disagree -- the length is the
  checkpoint's, not the request's -- but the runtime is sized to it, so the
  check belongs where the runtime is opened.

### What gates it

The six wave hashes and the three sequential hashes are unchanged, and
`ctest -R magpietts` passes. That is the baked path; for the zero-shot path
byte-identity against sequential is unavailable for the usual reason (batched
matmul numerics, and the wave's lane assignment), so the check is **step 0**:

```
sequential  216 148 386 813 232 1088 1041 789
wave w=8    216 148 386 813 232 1088 1041 789
```

Frame 0 is sampled straight off the prefill, so it is exactly the frame that
says whether the lane's conditioning arrived intact. It matches byte for byte,
and frames 1 onward diverge -- which is what the baked checkpoint does too, on
the same test.

### The book, in a cloned voice

Pride and Prejudice (Gutenberg 1342), same text as the baked run, one request,
128 lanes, `--tts.context-audio` pointing at the server's cached `default.wav`:

| | baked | zero-shot |
|---|---|---|
| conditioning positions | 110 (table row) | 217 (context encoder) |
| chunks | -- | 7,072 |
| output | 12 h 31 m | **11 h 04 m** (877,853,696 samples) |
| wall clock | 3 m 29.7 s | **3 m 06.4 s** |
| realtime factor | 216.3x | **213.5x** |
| decoder / codec alone | -- | 337.1x / 262.5x |
| first audio | -- | 878 ms |
| peak RSS | 24.9 GB | 24.7 GB |
| FLAC | 873 MB (44%) | 890 MB (53%) |

Within 1.3% of the baked checkpoint's rate, on a voice that reads the book an
hour and a half faster (183 words a minute against 162). Nothing about the
conditioning route costs throughput; the 5x gap was entirely the wave being
switched off. The decoder's own 337x is below the 451x the same run reached
without an attention prior, which is what the prior costs: it is applied in nine
layers and alignment is estimated from four more.

Levels sampled at 60 s, 15,000 s, 30,000 s and 39,000 s: mean -20.5 to -21.8 dB,
peak -2.6 to -3.9 dB. Consistent across the whole book, no silence, no clipping.

Run to run the sample count moves by a fraction of a percent. That is the
documented lane dependence: which lane a chunk lands in depends on admission
timing, and the arithmetic depends on the lane.

---

# What is still missing

The demo works: `synthesize -i pride.txt --tts.context-audio <voice>.wav`
renders the novel in a cloned voice at 213.5x. Three things stand between that
and a checkpoint anyone can run.

## Gap 1 -- the tokenizer has to be hand-built

`~/nemo-bench-assets/tokenizer-v2607` exists and works, but a user cannot be
asked to construct it. Two ways out, in order of preference:

1. **Embed the tokenizer in the GGUF**, as the ASR models do (base64 blobs).
   Self-describing checkpoints, and the whole mismatch class disappears --
   including `NEMO_SPEECH_TOKENIZER_PROFILE`, which exists only because the
   profile is a proxy for "which dictionaries". Note the TTS side is 15
   tokenizers with IPA dictionaries, not one SentencePiece model, so decide
   whether to embed all of them or only those the checkpoint's
   `language_to_tokenizer_mapping` reaches.
2. **Register a profile** for this tokenizer layout, so no override is needed.
   Cheaper, but leaves the assets out of band.

This is now the only thing keeping the demo from being reproducible by someone
who did not build the assets.

## Gap 2 -- the voice is per process, not per request

The wave carries conditioning per lane, and two lanes may hold different
voices. Nothing can currently supply two: `context_audio_file` lives on
`MagpieRuntimeConfig`, is copied into every request's params, and
`MagpieSynthesisOptions` has no context field at all. So one server serves one
cloned voice, and the per-lane plumbing is exercised only with every lane
holding the same prefix.

What it needs: a reference on the request (audio or codes, or a handle to a
voice the server has already encoded), a cache keyed on it so a repeated voice
is encoded once rather than per request, and then the test this whole design
exists for -- **two concurrent requests with different reference WAVs in one
wave, each matching its own solo run**. That test has never been run, because
until this it could not be.

## Gap 3 -- conversion needs an override

`NEMO_SPEECH_TOKENIZER_PROFILE=v2607` is required, because this checkpoint's
pt-BR tokenizer omits `locale_specific_punct` (NeMo defaults it true; the public
v2607 sets it false). The difference is real and Portuguese-only. Gap 1
dissolves this.

## Gap 4 -- context window selection

The runtime takes the **leading** `context_duration_max` seconds of the
reference. NeMo takes a random window. For a 38 s file the leading 10 s may be
an intro or near-silence, and the model gets one shot at the speaker from it.

Not a correctness bug, and deliberately out of scope so far. Cheapest
improvement is picking the window by energy; `--context-offset SECONDS` is the
smaller version. Note `context_duration_min == max == 10.0`, so *more* reference
audio cannot help without retraining -- only a better ten seconds can.

## Reproducing what exists

```bash
# 1. convert (both need the server's weights)
W=~/devel/magpie-tts-server/magpie_tts_weights
NEMO_SPEECH_TOKENIZER_PROFILE=v2607 python convert_model.py \
    $W/Magpie-TTS--val_cer_gt=0.3605-step=1200.ckpt \
    --config-yaml $W/config_v3_nostress_fixed.local.yaml \
    --inference-yaml <repo>/docs/development/magpie-zero-shot-inference.yaml \
    --outfile magpie-zs.f16.gguf --outtype f16
python convert_model.py $W/21fps_causal_codecmodel.nemo \
    --with-codec-encoder --outfile nanocodec-enc.f16.gguf --outtype f16

# 2. the reference must be mono at the codec rate; the runtime refuses to
#    resample, because a resampler that differs from NeMo's changes the codes
#    silently. The server's own cached conversion is the safest source:
#    ~/devel/magpie-tts-server/var/ctx_cache/default.wav

# 3. synthesize
nemo-speech synthesize -i pride.txt --device cuda --no-warmup \
    -o - --format pcm \
    --tts.magpie-model magpie-zs.f16.gguf \
    --tts.codec-model nanocodec-enc.f16.gguf \
    --tts.tokenizer-model-dir ~/nemo-bench-assets/tokenizer-v2607 \
    --tts.context-audio ~/devel/magpie-tts-server/var/ctx_cache/default.wav \
    --tts.batch-size 128 --tts.longform-history-tokens 20 --tts.chunk-frames 32 \
  | ffmpeg -f s16le -ar 22050 -ac 1 -i - -c:a flac -y pride-zeroshot.flac
```

`-o -` does not stream: the CLI still accumulates the whole output before
writing it, so the pipe saves the 2.5 GB WAV on disk but not the 26 GB of RSS.

## Verifying conditioning, if you touch it

```bash
# capture NeMo's own prefix and codes
cd ~/devel/magpie-tts-server && ./.venv/bin/python3 \
    <repo>/scripts/tts/capture-magpie-reference.py --out /tmp/ref

# ours, from the reference's own window
MAGPIETTS_CONTEXT_DUMP=/tmp/cpp.bin nemo-speech synthesize \
    --tokens-file /tmp/ref/text-tokens.txt --tts.context-codes /tmp/ref/context-codes.txt ...
<repo>/scripts/tts/compare-context-prefix.py /tmp/ref/prefix.npy /tmp/cpp.bin
# expect cosine 0.99999851
```

Compare via `--context-codes`, never `--context-audio`: NeMo's capture used a
random window of the reference, so the audio route legitimately disagrees. That
mistake cost most of a session. `MAGPIETTS_CONTEXT_CODES_DUMP` dumps our codes
if you need to compare those instead.

## Traps

- Measure both arms of any comparison **on the same build**. Comparing the codes
  route before a fix against the audio route after it produced a confident and
  entirely wrong diagnosis.
- Listening and the metric can both be right. A cloned voice sounded correct
  while the prefix cosine was 0.63: different windows of the same speaker.
- A duration difference between two decode paths on the same text is a bug until
  proved otherwise. 26% was explained away as a slower voice; it was the decoder
  looping.
- A full `cmake --build` that fails late can leave a stale `nemo-speech` behind,
  and the next benchmark silently measures the previous build. Fixed for the ASR
  test, but check the binary's timestamp when a result surprises you.
