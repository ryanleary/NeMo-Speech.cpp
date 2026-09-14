# Plan: MagpieTTS zero-shot voice cloning in NeMo-Speech.cpp

**Status: proposed, not started.** Written 2026-08-28 against `4f96762`.
Companion to [magpie-parity-gaps.md](magpie-parity-gaps.md), which is the
running status list; this document is the how.

## Goal and shape of the problem

Run `magpie-tts-server`'s production checkpoint under `nemo-speech synthesize`
with zero-shot cloning from a reference wav.

The encouraging discovery from reading both implementations: **the baked-speaker
path the runtime already has and the zero-shot path we need converge on exactly
one tensor.** In NeMo (`magpietts.py:_prepare_decoder_context`) the entire
difference is:

```python
if self.has_baked_context_embedding:
    context_embeddings, lens = self.get_baked_context_embeddings_batch(...)  # table lookup
else:
    context_embeddings = self.context_encoder(context_input_embedded, context_mask)['output']
```

Both yield `context_embeddings (B, T_ctx, E)` plus a length, returned as
`additional_decoder_input` and prepended to the decoder sequence. The baked
table is literally a cached context-encoder output — which is why its
`baked_context_length` is 110, and our config's fixed 10 s reference at
21.53 fps with stacking 2 gives ~108 frames.

The C++ does the same thing, just with the lookup hardcoded
(`decoder.cpp:554` and three sibling sites):

```cpp
ctx_flat  = ggml_get_rows(ctx, model.baked_context, speaker_in);
ctx_emb   = ggml_reshape_2d(ctx, ctx_flat, h.n_embd, h.baked_context_length);
total_len = h.baked_context_length + audio_len;
```

So the decoder-side change is: **source that prefix from a computed tensor
instead of a lookup, and make its length dynamic.** Everything else is feeding
it.

Three further findings that shrink the job:

- **The context encoder needs no new attention code.** It is the same
  `transformer_2501.Transformer` as the text encoder — `n_layers: 1,
  d_model: 768, sa_n_heads: 12, kernel_size: 3, has_xattn: false` — differing
  only in `is_causal: false`. The C++ `load_transformer` already takes a
  `causal` flag and `causal_soft_max` (`model.cpp:977`) already returns a plain
  softmax when it is false. One extra `load_transformer` call reuses the
  existing encoder forward verbatim.
- **Context audio embedding reuses existing machinery.** NeMo embeds context
  codes with the *same* `embed_audio_tokens` (same `audio_embeddings` tables,
  same frame-stacking reshape) the decoder uses for its own audio input.
- **The codec encoder is small.** `HiFiGANEncoder`, `down_sample_rates
  [2,2,4,8,8]` (product 1024 = `samples_per_frame`), `base_channels: 24`,
  `encoded_dim: 32` — against the decoder's `base_channels: 864`. It is a
  fraction of the work already done for the decoder, and it is non-causal, so
  no streaming state is needed.

## Phasing

Ordered so that each phase is independently testable and the risky, expensive
work comes last. **Phase 1 makes the checkpoint run**; phases 2–3 make it fast
and self-contained.

| Phase | Delivers | Depends on |
|---|---|---|
| 0 | Correctness oracle and harness | — |
| 1 | Checkpoint runs, cloning from *precomputed* context codes | 0 |
| 2 | `frame_stacking_factor: 2` | 1 |
| 3 | Native codec encoder: clone from a wav | 1 |
| 4 | Performance (tracked separately, gaps 8 and 9) | 2 |

The key sequencing decision is **deferring the codec encoder to phase 3**. It
is the largest single item and it is not needed to prove the AR model works: we
can dump context codes from the Python server, which already exposes
`model._codec_helper.audio_to_codes` (`engine.py:673`). That turns cloning into
a build-time step for a fixed voice library, unblocks everything else, and gives
us a byte-exact oracle for phase 3 to reproduce.

---

## Phase 0 — Oracle and harness

Nothing here touches the runtime; it is what makes the rest verifiable.

1. **Dump reference tensors from Python.** A script in
   `scripts/tts/dump-magpie-reference.py` that, for a fixed reference wav, seed
   and text, writes: context codes `(8, T_ctx)` from `audio_to_codes`, the
   context-encoder output `(T_ctx, 768)`, the first N decoder-step logits, and
   the final codec tokens.
2. **Comparison test.** Feed the same inputs to the C++ and compare, tightest
   first: context-encoder output (rtol 1e-3 in f32), then greedy codes
   (must match exactly for the first N steps), then audio.
3. **Benchmarks.** Already done — [bench/](bench/). Re-run `./run-all.sh` after
   each phase.

Greedy (`--top-k 1`, fixed seed) is the comparison mode; sampled output will not
match and is not expected to.

## Phase 1 — Run the checkpoint, cloning from precomputed codes

### 1a. Converter (`conversion/tts.py`)

- Accept `...magpietts_preference_optimization.MagpieTTSModelOnlinePO` alongside
  the plain target (gap 7), and **drop `squim_objective_model.*`** — 106 GRPO
  reward-model tensors that must not reach the GGUF.
- Replace the `has_baked_context_embedding` guard (gap 6) with a branch:
  emit either the baked table *or* `context_encoder.*`, and record which in a
  new `magpietts.conditioning` key (`baked` | `context_encoder`).
- Emit `magpietts.context_encoder.{layers,heads,kernel_size,causal}` and
  `magpietts.context_encoder.max_positions` from
  `context_encoder.position_embeddings.weight`.
- Make `local_transformer_in_projection` optional (gap 5); emit
  `magpietts.local_transformer.has_in_projection`.
- Accept a Lightning `.ckpt` plus a sidecar hparams YAML directly, so staging
  by hand is not required. Note the `.ckpt`'s embedded
  `hyper_parameters.cfg` declares the PO target while the sidecar declares the
  plain one — prefer the sidecar, and say so in the error when neither matches.

### 1b. Model load (`src/tts/magpietts/model.cpp`)

- `baked_context_embedding.weight` and `local_transformer_in_projection.*`
  become optional (`require_tensor` -> optional lookup), driven by the new
  metadata keys. A missing in-projection means identity — legal exactly when
  `lt_hidden == n_embd`, which should be asserted rather than assumed.
- Load the context encoder:
  ```cpp
  load_transformer(model, model.context_encoder, "context_encoder",
                   h.ctx_enc_layers, h.n_embd, h.ctx_enc_heads,
                   h.ctx_enc_kernel, /*causal=*/false, /*has_cross=*/false, 0, 0);
  ```
- Fail loudly and specifically when a GGUF declares `context_encoder`
  conditioning but the build or request supplies no context.

### 1c. Conditioning prefix (`src/tts/magpietts/decoder.cpp`)

The four sites that reshape a baked lookup become one accessor returning
`{tensor, length}`, backed by either the table or a caller-supplied prefix.
`h.baked_context_length` becomes a per-request `ctx_len`. Audit every use:
`total_len = ctx_len + audio_len`, the attention-prior zero-padding (NeMo pads
the prior by `dec_context_size`, `magpietts.py:1841`), and decoder position
offsets.

Risk: the prefix length is currently compile-time-constant per model and the
KV cache and any captured graphs are sized from it. Cleanest fix is to keep it
fixed *per request* — the context does not change mid-utterance — and rebuild
cached graphs when it changes. Since a voice is fixed for a whole script, that
is one rebuild per request at worst.

### 1d. Context path (new, `src/tts/magpietts/context.cpp`)

Codes `(8, T_ctx)` -> `embed_audio_tokens` (existing) -> context encoder
(existing forward) -> `(T_ctx, 768)`. Wrap with `context_audio_bos_id` /
`context_audio_eos_id`, which the converter already emits
(`codebook_size + 2/3`) — confirm against NeMo rather than assuming.

### 1e. Surface

`--context-codes PATH` on `synthesize`, plus `tts.context-codes` in the config
tree and the equivalent HTTP field. A small documented container (magic,
n_codebooks, T, int32 codes) so files are self-describing.

**Exit criterion:** greedy codes match the Python oracle exactly for 100 decoder
steps on a 1-sentence input, with `frame_stacking_factor` still 1 — which means
testing against a stacking-1 checkpoint, or accepting phase 1 as
compile-and-load-only and moving the correctness gate into phase 2.
*This is the main sequencing risk: our production checkpoint is stacking-2, so
phase 1 cannot be fully validated on it alone. Prefer to find or export a
stacking-1 zero-shot checkpoint for the phase-1 gate; if none exists, merge the
phase-1 and phase-2 gates and accept the larger step.*

## Phase 2 — `frame_stacking_factor: 2`

Remove the `model.cpp:798` rejection and thread the factor through:

- **Audio embeddings.** 16 tables = 8 codebooks x 2 stacked frames; the
  converter's `n_codebooks //= frame_stacking` must stay consistent with what
  the runtime indexes.
- **Local transformer.** 16 output projections (32 tensors with biases), and
  `position_embeddings` of 18 = 16 + 2. The LT currently loops
  `h.audio_codebooks`; it must loop `codebooks * stacking`.
- **Decoder step.** Each step emits 2 frames; EOS detection, the codec feed and
  the attention-prior advance all count frames, not steps.
- **Codec.** Unchanged — it still consumes frames.

This is the most invasive phase and touches the hot loop, so re-run
`bench/run-all.sh` and `profile-native.sh` after it: the step count halves,
which should show up as ~2x on RTF at unchanged steps/s.

## Phase 3 — Native codec encoder

Lets `--context-audio reference.wav` work without Python.

- **Converter (`conversion/codec.py`).** Currently walks only
  `audio_decoder.`; add `audio_encoder.` behind a flag so the encoder is opt-in
  and existing decoder-only GGUFs stay byte-identical. Same weight-norm
  materialization as the decoder path.
- **Runtime (`src/tts/nanocodec/`).** `HiFiGANEncoder`: 5 downsampling blocks
  `[2,2,4,8,8]`, `base_channels 24`, `encoded_dim 32`, leaky-ReLU, **replicate**
  padding (the decoder uses zeros — do not copy its padding).
- **FSQ analysis.** The synthesis direction already stores `num_levels`,
  `dim_base_index` and `scale`/`offset` (`codec.py:74`); analysis is
  bound/round then combine with `dim_base_index`. Group FSQ: 32 dims / 8 groups
  = 4 dims per group, levels `[8,7,6,6]`, 2016 codes per group.
- **Wav input.** `nemo_speech_common` already has `load_wav_file` with
  resampling and stereo downmix; TTS does not link it yet — one line in
  `src/tts/CMakeLists.txt`.
- Crop the reference to `context_duration` (10 s in our config) the same way
  the dataset does, or cloning will drift from the Python behaviour.

**Exit criterion:** codes from `--context-audio ref.wav` match
`audio_to_codes(ref.wav)` from Python exactly. Integer codes, so exact equality
is the right bar.

## Phase 4 — Performance

Tracked as gaps 8 and 9 and quantified in
[rtf-benchmark.md](rtf-benchmark.md) / [optimization-parity.md](optimization-parity.md).
Ordered: chunk batching (~4x), then the decoder-stack CUDA graph and the 116
syncs/step. Deliberately after correctness — phase 2 changes the hot loop, and
optimizing it first would mean doing that work twice.

## What could go wrong

| Risk | Why it matters | Mitigation |
|---|---|---|
| Stacking-2 blocks phase-1 validation | Our only zero-shot ckpt is stacking-2, so phases 1 and 2 may not be independently gateable | Find a stacking-1 zero-shot ckpt; otherwise merge the gates and expect a harder debug |
| Dynamic prefix length vs cached graphs | Prefix length is currently constant; graphs and KV caches size off it | Fix per request, rebuild on change — a voice is constant for a whole script |
| Attention prior offsets | Prior is padded by context size; an off-by-one degrades alignment subtly, not loudly | Compare `estimate_alignment_from_layers` scores against the oracle, not just audio |
| `use_text_conditioning_encoder: true` in our config | A second, text-context path exists (`text_ce_tokenizer`) | Out of scope for cloning; assert audio context and reject text context explicitly |
| GRPO reward tensors | 106 `squim_objective_model.*` tensors bloat or break conversion | Drop in the converter, assert absent in the GGUF |

## Rough size

| Phase | Scope | Confidence |
|---|---|---|
| 0 | ~200 lines of Python tooling | high |
| 1 | Converter + load + one accessor refactor + small forward | high — the integration point is a single tensor |
| 2 | Threads a factor through the hot loop | **low** — invasive, and our checkpoint gates it |
| 3 | A conv encoder + FSQ analysis; self-contained, exactly verifiable | medium |
| 4 | See gaps 8 and 9 | medium |

The honest summary: phase 1 is small and well-understood because the runtime
already has the shape it needs. Phase 2 is the one to be wary of — it is the
least contained and it gates validation of everything before it.
