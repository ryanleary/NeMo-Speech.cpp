# Gaps to running our Magpie checkpoint natively

**Living document.** Update the status column as gaps close; keep the
verification commands runnable so the claims stay checkable.
The implementation plan for closing gaps 1–7 is
[voice-cloning-plan.md](voice-cloning-plan.md).

Last verified: 2026-08-28, after the phase 1+2 implementation.

## The goal

Run `~/devel/magpie-tts-server`'s production checkpoint under
`nemo-speech synthesize` with the same behaviour the Python server gives us:
**zero-shot voice cloning from a short reference wav**.

Our stack:

| Piece | File |
|---|---|
| Token generator | `magpie_tts_weights/Magpie-TTS--val_cer_gt=0.3605-step=1200.ckpt` (3.7 GB) |
| Config | `magpie_tts_weights/config_v3_nostress_fixed.local.yaml` (Magpie TTS 4.0.1 GRPO step 1200) |
| Codec | `magpie_tts_weights/21fps_causal_codecmodel.nemo` |

## Status summary

| # | Gap | Severity | Status |
|---|---|---|---|
| 1 | Zero-shot context encoder unimplemented | **Blocker** | **Closed** |
| 2 | NanoCodec *encoder* never converted | **Blocker** | Open — deferred to phase 3 |
| 3 | No reference-audio input surface | **Blocker** | **Closed** (codes; wav needs 2) |
| 4 | `frame_stacking_factor: 2` rejected | Blocker **+ ~2x perf** | **Closed** |
| 5 | `local_transformer_in_projection` assumed present | Blocker | **Closed** |
| 6 | Converter guards on `has_baked_context_embedding` | Cosmetic once 1–5 land | **Closed** |
| 7 | GRPO checkpoints declare a different `target` | Cosmetic | **Closed** |
| 8 | Long-form chunks decoded sequentially, not batched | ~4x perf, not a blocker | Open |
| 9 | Decoder stack not graph-replayed; 116 GPU syncs per step | perf, not a blocker | Open |
| 10 | Non-causal conv was causally padded | **Was a silent correctness bug** | **Closed** |
| 11 | Context sequence not padded to the fixed config length | **Was a silent correctness bug** | **Closed** |
| 12 | This checkpoint's tokenizer is not packaged (vocab 3359 vs 2362), and `eos_id` is hard-coded to the public model's 2361 | Workaround: `--tokens-file` | Open |
| — | Codec decoder conversion | — | **Works today** |

**Our checkpoint now works**: it converts, loads, and synthesizes correct speech
in a cloned voice from reference codes. The context prefix matches NeMo at
cosine 0.99999863 and the first decoder step is bit-identical. What remains for
full parity is gap 2 — encoding a wav to those codes natively instead of
dumping them from Python once per voice — plus packaging this checkpoint's
tokenizer. See [voice-cloning-status.md](voice-cloning-status.md).

Gaps 6 and 7 are one-line guards. Gaps 1–3 are the real work: they are the
difference between "speaks in one of five stock voices" and "speaks in yours".

Gaps 4 and 8 are the two that also cost throughput — together they account for
essentially all of the 7.5x speed deficit measured in
[rtf-benchmark.md](rtf-benchmark.md). Gap 8 is independent of everything else
here and would speed up the *public* checkpoint today.

---

## 1. Zero-shot conditioning is not implemented (blocker)

The native runtime conditions the decoder on a **baked speaker embedding**: a
fixed 110-frame prefix looked up from a 5-row table, prepended to the decoder
sequence (`src/tts/magpietts/decoder.cpp`, four call sites around lines
554, 718, 892, 1113).

Our checkpoint has no such table. It conditions on a `context_encoder` —
8 tensors, 1 transformer layer plus `norm_out` and a 2048-slot position
embedding — that consumes embedded codec tokens from the *reference audio*.

```
ours                            public 357M / runtime
----                            ---------------------
context_encoder.norm_out.weight        baked_context_embedding.weight  [5, 110*768]
context_encoder.layers.0.*             _baked_embedding_T
context_encoder.position_embeddings    _baked_embedding_D
  .weight  [2048, 768]                 baked_context_embedding_len
```

`model.cpp:879` calls `require_tensor` on `baked_context_embedding.weight`, so
loading fails hard even if conversion were forced through.

**To close:** emit `context_encoder.*` in the converter, add a context-encoder
graph, and let the decoder take a variable-length conditioning prefix instead
of the fixed `h.baked_context_length`. Smaller than it looks: the baked table is
a *cached context-encoder output*, so both paths meet at one `(T_ctx, 768)`
tensor, and the context encoder is the same transformer block the C++ already
has (`n_layers: 1`, differing only in `is_causal: false`, which
`causal_soft_max` already handles). See
[voice-cloning-plan.md](voice-cloning-plan.md) phase 1.

## 2. The codec encoder is deliberately not converted (blocker)

Gap 1 needs reference audio expressed as codec tokens. `conversion/codec.py`
walks only tensors prefixed `audio_decoder.` — its docstring says the audio
encoder, discriminators, and loss modules "are intentionally omitted". There is
also no encoder graph in `src/tts/nanocodec/`.

So today there is no path, at any layer, from a wav to context codes.

**To close:** convert `audio_encoder.*` plus the FSQ *analysis* direction, and
implement the encoder forward pass. This is the largest single item.

**This is now the plan's phase 3**, deliberately deferred: phase 1 accepts
*pre-computed* context codes (dumped once per voice from the Python stack via
`audio_to_codes`), which sidesteps this gap entirely, unblocks everything else,
and provides the byte-exact oracle phase 3 must reproduce.

Smaller than first assumed: `HiFiGANEncoder` with `base_channels: 24` against
the decoder's 864, non-causal so no streaming state.

## 3. No reference-audio input surface (blocker)

`nemo-speech synthesize` exposes only `--voice NAME` and `--speaker N`, and
`magpietts.cpp:1053` bounds the speaker index against `h.baked_speakers`.
Nothing in `src/tts/`, the HTTP API, or the config tree accepts an audio prompt.
Grep for `context audio`, `reference audio`, `voice clon`, or `prompt audio`
across `docs/`, `src/tts/`, and `app/` returns nothing.

**To close:** a `--context-audio PATH` (or `--context-codes PATH`) flag through
`SynthesisRequest`, plus the equivalent HTTP field.

## 4. `frame_stacking_factor: 2` is rejected (blocker)

`model.cpp:798`:

```
unsupported frame_stacking_factor=%d; this example currently supports 1
```

Ours is 2; the public checkpoint is 1. This also changes tensor bookkeeping:
we carry 16 `audio_embeddings.*` and 16 `local_transformer_out_projections.*`
(32 tensors incl. biases), and the converter divides by the stacking factor to
report 8 codebooks — so a forced conversion would silently load only half our
output projections.

### Performance consequence

Beyond compatibility, stacking is worth ~2x wall-clock: our checkpoint needs
half as many autoregressive steps for the same audio duration. See
[rtf-benchmark.md](rtf-benchmark.md).

## 5. `local_transformer_in_projection` is assumed to exist (blocker)

`model.cpp:882` requires it unconditionally. NeMo only creates that projection
when `local_transformer_hidden_dim != embedding_dim`. Ours are both 768, so the
module does not exist; the public checkpoint's are 256 vs 768, so it does.

**To close:** treat a missing in-projection as identity.

## 6. Converter guard on `has_baked_context_embedding`

`conversion/tts.py:312` requires the flag to be true. Our config does not set
it. One-line guard — but do not relax it before 1–5 land, or conversion will
produce a GGUF that fails at load with a less obvious error.

## 7. GRPO checkpoints declare a different `target`

`conversion/tts.py:310` accepts only
`nemo.collections.tts.models.magpietts.MagpieTTSModel`.

The `.ckpt`'s own embedded `hyper_parameters.cfg` says
`...magpietts_preference_optimization.MagpieTTSModelOnlinePO` — the GRPO
training wrapper — while the sidecar YAML the server actually loads says the
plain `MagpieTTSModel`. Either accept the PO target or keep using the sidecar.

Note the `.ckpt` also carries 106 `squim_objective_model.*` tensors (the GRPO
reward model) that must be dropped before conversion.

## 8. Long-form chunks are decoded sequentially (performance, not a blocker)

Not a parity gap with our checkpoint — the native runtime already chunks
long text the same way ours does (20 sentences -> 20 chunks, `longform=on`) —
but it decodes those chunks one after another, splicing history context between
them (`src/tts/magpietts/magpietts.cpp:1052`). Our Python server decodes them
as a batch, and since decoding is launch-bound a batch costs barely more than a
single chunk.

Measured effect: native throughput is flat at ~83 decoder steps/s regardless of
input length, while the Python server reaches 311–361 steps/s on a 20-sentence
script. This is the single largest performance item and is orthogonal to gaps
1–7.

**To close:** give the decoder loop a batch dimension over the existing chunk
list. Note the tension with prosody: our server exposes this as
`longform: auto | never | always`, where `never` batches independent chunks
(~13% faster here) at the cost of a seam at each join.

## 9. Decoder stack is not graph-replayed, and the loop fences 116x per step (performance)

Measured with `nsys` (see [optimization-parity.md](optimization-parity.md)):
513 `cudaLaunchKernel` + 8.8 `cudaGraphLaunch` + **116 `cudaStreamSynchronize`**
per decoder step, with synchronization accounting for 66% of all CUDA API time.

The 8.8 graph launches are the local transformer's 8 codebooks replaying from
`LocalTransformerGraphBank`. The 12-layer decoder stack is not graphed, and
`ggml_backend_graph_compute` synchronizes on every call.

Both of these are things we already fixed on the Python side
(`decoder_graph.py`, `decode_sync.py`), so the shape of the fix is known.

**To close:** graph-replay the decoder step the way the local transformer
already is, and stop fencing per graph eval / per staged tensor set.

---

## What already works

`21fps_causal_codecmodel.nemo` converts cleanly with no changes:

```
$ ./.venv-convert/bin/python convert_model.py \
    ~/devel/magpie-tts-server/magpie_tts_weights/21fps_causal_codecmodel.nemo \
    --outfile /tmp/nanocodec.f16.gguf --outtype f16
wrote /tmp/nanocodec.f16.gguf
stored 387 decoder/FSQ tensors; skipped 97 source tensors
```

It is the same NanoCodec family as the public decoder: 22050 Hz,
`samples_per_frame` 1024 (=> 21.53 fps), `GroupFiniteScalarQuantizer` with
8 groups x `[8, 7, 6, 6]` levels = 2016 codes. Our token generator's audio vocab
is 2024 = 2016 + 8 special tokens, which is consistent.

So the token->audio half of our stack is already portable. Only the
text->token half is blocked.

---

## Side-by-side

| Property | Ours (4.0.1 GRPO) | Public 357M | Runtime supports |
|---|---|---|---|
| Conditioning | `context_encoder`, zero-shot | 5 baked speakers | baked only |
| `frame_stacking_factor` | 2 | 1 | 1 only |
| `embedding_dim` | 768 | 768 | any |
| `local_transformer_hidden_dim` | 768 (no in-proj) | 256 (in-proj) | in-proj required |
| `text_vocab_size` | 3359 | 2362 | from GGUF |
| Audio codebooks | 8 (x2 stacking = 16 tensors) | 8 | from GGUF |
| Audio vocab | 2024 | 2024 | from GGUF |
| Encoder / decoder layers | 6 / 12 | 6 / 12 | from GGUF |

## Reproducing the checks

Stage the checkpoint into the layout the converter expects, then run it:

```bash
# 1. strip the GRPO reward model, re-save a weights_only-loadable state dict
~/devel/magpie-tts-server/.venv/bin/python - <<'PY'
import torch, yaml
ck = torch.load('.../Magpie-TTS--val_cer_gt=0.3605-step=1200.ckpt',
                map_location='cpu', weights_only=False, mmap=True)
sd = {k: v for k, v in ck['state_dict'].items()
      if not k.startswith('squim_objective_model')}
torch.save(sd, '/tmp/magpie-stage/model_weights.ckpt')          # 264 tensors
cfg = yaml.safe_load(open('.../config_v3_nostress_fixed.local.yaml'))['cfg']
yaml.safe_dump(cfg, open('/tmp/magpie-stage/model_config.yaml', 'w'))
PY

# 2. attempt conversion
./.venv-convert/bin/python convert_model.py /tmp/magpie-stage \
    --outfile /tmp/magpie.f16.gguf --outtype f16
# -> error: this GGML example expects MagpieTTS decoder_ce with baked context embeddings
```

Confirm the missing tensors directly:

```bash
./.venv-convert/bin/python - <<'PY'
import torch
sd = torch.load('/tmp/magpie-stage/model_weights.ckpt',
                map_location='cpu', weights_only=True, mmap=True)
for k in ['_baked_embedding_T', '_baked_embedding_D',
          'baked_context_embedding_len', 'baked_context_embedding.weight',
          'local_transformer_in_projection.weight']:
    print('PRESENT' if k in sd else 'MISSING', k)
print('context_encoder tensors:', len([k for k in sd if k.startswith('context_encoder')]))
PY
```

Expected: all five MISSING, 8 context-encoder tensors present.
