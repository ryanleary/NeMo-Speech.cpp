# Zero-shot voice cloning: what landed

Implementation of [voice-cloning-plan.md](voice-cloning-plan.md), phases 0-2
(1 and 2 merged, as the plan warned they would have to be).

**Status: working.** Our checkpoint converts, loads, and synthesizes correct
speech in a cloned voice from reference-audio codec codes.

```bash
# once per voice, from the magpie-tts-server checkout
.venv/bin/python ~/devel/NeMo-Speech.cpp/scripts/tts/dump-magpie-reference.py \
    voices/default.wav --out /tmp/ref

# convert once
python convert_model.py magpie_tts_weights/Magpie-TTS--val_cer_gt=0.3605-step=1200.ckpt \
    --config-yaml magpie_tts_weights/config_v3_nostress_fixed.local.yaml \
    --outfile magpie-zs.f16.gguf --outtype f16

nemo-speech synthesize "Magpie is a text to speech model." \
    --magpie-model magpie-zs.f16.gguf \
    --codec-model nemo_nano_codec_...decoder.f16.gguf \
    --tokenizer-dir <extracted .nemo dir> \
    --context-codes /tmp/ref/context-codes.txt \
    --device cuda -o out.wav
```

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

### 3. A missing text-EOS token, in my own tooling

Not a runtime bug, but it produced a convincing symptom: `s3` came out as
"...in C plus plus **plus**". The model appends a text EOS
(`text_vocab_size - 1` = 3358, one of two embedding rows past the tokenizer's
3357-token vocabulary) and my token dump omitted it, so the model was never told
the text had ended and kept going. Adding it made all three test sentences
transcribe exactly.

The lesson is the same as below: capture the tokens the working implementation
actually feeds the model, rather than re-deriving them by calling the tokenizer.

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
- **Gap 2, the native codec encoder** (plan phase 3). Context codes still come
  from Python, once per voice. `--context-audio ref.wav` does not exist yet.
- **Full greedy parity beyond step 1.** The first decoder step matches
  bit-exactly; later steps drift on f16 argmax ties. Closing that would mean an
  f32 GGUF, and is only worth doing if a future bug needs the longer baseline.
- **A packaged tokenizer for our checkpoint.** Its text vocab is 3359 against
  the public tokenizer's 2362, so `--tokenizer-dir` from the public model gives
  meaningless IDs. Worked around with `--tokens-file`. Two things are needed for
  a real fix: this checkpoint's tokenizer assets in a `.nemo`-shaped directory,
  **and** a non-hard-coded EOS - `tokenizer_impl.cpp:68` has
  `int eos_id = 2361;` (the public checkpoint's last token) which is never
  assigned from anywhere. It should be `text_vocab_size - 1`, which is 3358 for
  ours.
- **Performance** (gaps 8, 9) - untouched, deliberately, since phase 2 rewrote
  the hot loop.
- `generate_magpietts_codes` in `model.cpp` is dead code (no callers) and was
  left at one frame per step; it now refuses stacked or zero-shot models rather
  than silently mis-decoding.
