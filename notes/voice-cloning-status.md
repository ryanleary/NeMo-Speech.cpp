# Zero-shot voice cloning: what landed

Implementation of [voice-cloning-plan.md](voice-cloning-plan.md), phases 0-2
(1 and 2 merged, as the plan warned they would have to be).

**Status: working, with no Python in the loop.** Plain text in,
`--context-audio reference.wav` for the voice, correct speech out.

```bash
# convert the token generator and a codec that includes the audio encoder
python convert_model.py magpie_tts_weights/Magpie-TTS--val_cer_gt=0.3605-step=1200.ckpt \
    --config-yaml magpie_tts_weights/config_v3_nostress_fixed.local.yaml \
    --outfile magpie-zs.f16.gguf --outtype f16
python convert_model.py magpie_tts_weights/21fps_causal_codecmodel.nemo \
    --with-codec-encoder --outfile nanocodec.f16.gguf --outtype f16

nemo-speech synthesize "Magpie is a text to speech model." \
    --magpie-model magpie-zs.f16.gguf \
    --codec-model nanocodec.f16.gguf \
    --tokenizer-dir <public magpie tokenizer dir> \
    --context-audio reference.wav \
    --device cuda -o out.wav
```

The reference WAV must be mono at the codec's sample rate (22050 Hz); the
runtime refuses anything else rather than resampling, because a resampler that
differs from the reference implementation's changes the codes silently. Only the
leading `context_duration_max` seconds (10 s here) are used, which is the window
the model was trained on.

`--context-codes` still accepts codes directly, which is how the encoder itself
is validated.

## Verification

| Check | Result |
|---|---|
| Context-encoder prefix vs NeMo | **cosine 0.99999863**, mean abs diff 0.0008 |
| First decoder step (2 frames, 16 codebooks) vs NeMo greedy | **bit-identical** |
| ASR round-trip, 3 sentences | correct (see below) |
| Public baked checkpoint | no regression; transcribes correctly |
| Frame-stacking unit test | passes |

ASR round-trip through `nemo-speech transcribe`:

```
want: Magpie is a text to speech model.
got : Magpie is a text to speech model.
want: The quick brown fox jumps over the lazy dog.
got : The quick brown fox jumps over the lazy dot.
want: Zero shot voice cloning now runs natively in C plus plus.
got : Zero shot voice cloning now runs natively in C Plus Plus plus.
```

### On the code-level divergence

Greedy codes match NeMo for the whole first decoder step - both frames, all 16
codebooks, bit-exact - then diverge at frame 2 in one codebook of eight.

That divergence is numerical, not algorithmic, and the cleanest way to see it is
to ask how stable *NeMo* is with itself:

| Comparison | First divergent frame |
|---|---|
| NeMo TF32 on vs NeMo TF32 off | **0** (3 of 8 codebooks) |
| NeMo vs NeMo-Speech.cpp, f16 weights | 2 |
| NeMo vs NeMo-Speech.cpp, f32 weights | 2 |

**NeMo disagrees with itself at frame 0** when you flip a single precision
setting, each differing code being one FSQ level step. So the native runtime
agrees with NeMo *longer than NeMo agrees with itself* across a precision
change. Greedy code-exactness past the first step is not achievable here and is
not a meaningful target; the first complete step matching bit-exactly is the
structural check that matters.

Neither f32 weights nor disabling the KV cache moves the divergence point, which
rules out weight storage and the incremental-decode path as causes.

### On the codec encoder

Encoding the same 10 s reference natively and with NeMo gives the same 216
frames and 98.84% identical codes; every one of the 20 differences is a single
FSQ level step, the smallest possible. The resulting conditioning prefixes agree
at cosine 0.99969.

Two things had to match exactly to get there, and neither is guessable from the
module definition:

* **Audio is zero-padded so the last frame is full** before encoding
  (`AudioCodecModel.encode_audio`). Without it the trailing partial frame is
  dropped and the frame count is one short.
* **Convolutions pad on both sides in `replicate` mode**, not zeros — the
  decoder is causal and zero-padded, the encoder is neither.

The residue is not weight precision: an f32 codec produces byte-identical codes
to the f16 one, and switching the FSQ rounding from round-half-away to
round-half-to-even (matching `torch.round`) does not move it either. It is
near-ties in the convolution arithmetic landing either side of a rounding
boundary. For a conditioning embedding that is immaterial.

## Reproducing the checks

Capture a reference from the real NeMo path (not a reconstruction of it), then
compare:

```bash
# from the magpie-tts-server checkout, with its venv
.venv/bin/python ~/devel/NeMo-Speech.cpp/scripts/tts/capture-magpie-reference.py --out /tmp/ref

MAGPIETTS_CONTEXT_DUMP=/tmp/cpp.bin MAGPIETTS_CODES_DUMP=/tmp/codes \
nemo-speech synthesize x --tokens-file /tmp/ref/text-tokens.txt \
    --context-codes /tmp/ref/context-codes.txt \
    --magpie-model magpie-zs.f16.gguf --codec-model ... --tokenizer-dir ... \
    --top-k 1 --temperature 0.01 --seed 1 -o /tmp/out.wav --force

scripts/tts/compare-context-prefix.py /tmp/ref/ce-out.npy /tmp/cpp.bin
```

`MAGPIETTS_CONTEXT_DUMP` writes the prefix as raw f32; `MAGPIETTS_CODES_DUMP`
writes generated codec frames one per line, suffixed with the run label.

## The two bugs worth knowing about

Both had the same character: **shapes and magnitudes stayed right, only values
were wrong.** Neither crashed, and both produced confident, speech-like,
completely wrong audio.

### 1. Non-causal convolutions were causally padded

NeMo's `ConvolutionLayer` pads **causally** when `is_causal` and
**symmetrically** otherwise (`(kernel-1)/2` either side). The C++ `causal_conv1d`
always left-padded, because until now every transformer in the runtime was
causal. The context encoder is the first bidirectional one, and it has
`kernel_size: 3`.

The failure mode is the reason this is worth writing down: **the shapes and the
magnitudes were right, and only the values were wrong.** Cosine similarity 0.773
- clearly broken, but nothing crashed, and audio would still have come out. Only
comparing against the reference caught it.

`conv1d(ctx, x, kernels, causal)` now selects the padding; `causal_conv1d`
remains as the causal-only wrapper, so the ASR/codec callers are untouched.

### 2. The conditioning sequence is padded to a fixed length, and the padding is not inert

This one cost the most time and is the more surprising of the two.

NeMo's dataset pads the context sequence out to a **fixed** length derived from
the config, not from the reference audio
(`text_to_speech_dataset.py`):

```python
_required_len = int(context_duration_max * sample_rate / samples_per_frame) + 2
# 10.0 s at 22050/1024 fps -> 217
```

The context encoder then runs over all 217 positions with the padding **masked
in as valid**. Because that encoder is bidirectional, those 108 zero rows change
every output position - and all 217 rows become the decoder's conditioning
prefix, not just the 109 real ones.

The natural implementation - embed the reference, encode it, use the result -
gives a 109-row prefix whose content is also wrong, and produces fluent
gibberish. `magpietts.context_encoder.max_duration_s` now carries the duration
so the runtime can reproduce the length from the codec's frame rate.

### 3. The text EOS was hard-coded to one checkpoint's vocabulary

Symptom: `s3` came out as "...in C plus plus **plus**". The model appends a text
EOS at `text_vocab_size - 1` — 3358 for our checkpoint, one of two embedding
rows past the tokenizer's 3357-token vocabulary. Without it the model is never
told the text ended and keeps going.

It surfaced first in my own token dump, but the runtime had the same bug more
permanently: `tokenizer_impl.cpp` declared `int eos_id = 2361;` — the *public*
checkpoint's last token — and never assigned it from anywhere. It is now derived
from the loaded model's text embedding table, which is what made
`--tokenizer-dir` work for our checkpoint and retired the `--tokens-file`
workaround.

### How the oracle hid it

The first version of `dump-magpie-reference.py` built the context embedding by
calling `embed_audio_tokens` and the context encoder directly - reproducing my
own assumption rather than NeMo's real path. It reported **cosine 0.99999763**
against a C++ implementation that was wrong in exactly the same way.

The fix was to stop reconstructing the reference and start *capturing* it:
hooks on `model.context_encoder.forward` and `prepare_context_tensors` inside a
real `engine.synthesize()` call, which immediately showed a (217, 768) prefix
where I expected (109, 768). **An oracle you wrote from the same understanding
as the implementation is not an oracle.**

## What changed

**Converter** (`conversion/tts.py`, `registry.py`, `convert_model.py`)
- Detects conditioning from the *weights*, not the config: baked and
  context-encoder checkpoints are mutually exclusive by construction, and some
  sidecar configs do not set `has_baked_context_embedding` at all.
- Accepts the GRPO `MagpieTTSModelOnlinePO` target, and drops the 106
  `squim_objective_model.*` reward-model tensors (asserted absent at write).
- Accepts a bare Lightning `.ckpt` plus `--config-yaml`, so no hand-staging.
- Emits `magpietts.conditioning`, `emit_codebooks`,
  `local_transformer.has_in_projection`, and the `context_encoder.*` dims.
- Cross-checks that the embedding tables, the LT output projections and
  `final_proj` all agree on the emitted-codebook count.

**Runtime**
- `emit_codebooks` (= `audio_codebooks * frame_stacking_factor`) now indexes
  everything per-(codebook, stack slot): embedding tables, LT projections, and
  the final-projection logits. `audio_codebooks` keeps meaning real codec
  codebooks, which is what the NanoCodec consumes.
- `MagpieContextEncoder` (`context.cpp`) turns reference codes into the prefix:
  BOS/EOS frames, zero-pad to the stacking factor, de-interleave to one input
  per table, sum-embed, encode.
- The decoder's conditioning prefix is now a `magpietts_context_prefix` held on
  `MagpieDecoder`. It is constant for a request, so it lives there instead of
  being threaded through eight long eval signatures.
- Optional `baked_context_embedding` and `local_transformer_in_projection`
  (absent iff `lt_hidden == n_embd`, which is asserted).
- One decoder step now yields `stacking` codec frames; `split_stacked_frame`
  de-interleaves before the codec worker and all frame counting.
- `--context-codes` / `tts.context-codes`.

**Backward compatibility.** Already-published GGUFs carry none of the new keys,
so they default to `conditioning = baked`, `emit_codebooks = codebooks *
stacking`, `has_in_projection = true`. The public checkpoint still synthesizes
at the same RTF.

## Not done

- **Correct audio from our checkpoint.** See [Where it stands](#where-it-stands).
- **Non-English tokenization for this checkpoint** (gap 13). The runtime's
  per-language token offsets are the public checkpoint's. English agrees by
  coincidence — both put `english_phoneme` at offset 0 — so English is exact,
  but our checkpoint inserts `text_ce_tokenizer` at 96 and shifts every later
  language. Fixing it means carrying the offsets in the GGUF.
- **Reference audio must already be 22050 Hz mono.** Deliberate: see above.
- **Full greedy parity beyond step 1.** The first decoder step matches
  bit-exactly; later steps drift on f16 argmax ties. Closing that would mean an
  f32 GGUF, and is only worth doing if a future bug needs the longer baseline.

- **Performance** (gaps 8, 9) - untouched, deliberately, since phase 2 rewrote
  the hot loop.
- `generate_magpietts_codes` in `model.cpp` is dead code (no callers) and was
  left at one frame per step; it now refuses stacked or zero-shot models rather
  than silently mis-decoding.
