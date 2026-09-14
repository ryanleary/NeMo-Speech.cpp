#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert NVIDIA NeMo MagpieTTS checkpoints to GGUF.

Handles both conditioning styles of the ``decoder_ce`` model type:

* **baked** - the public MagpieTTS multilingual 357M checkpoint
  (https://huggingface.co/nvidia/magpie_tts_multilingual_357m), which ships a
  table of precomputed context embeddings, one per stock speaker.
* **context_encoder** - zero-shot checkpoints, which keep the context encoder
  itself and compute the conditioning prefix from reference audio codes at
  inference time.

The two are the same thing at different times: a baked row *is* a cached
context-encoder output, which is why they share every downstream tensor. Which
one a checkpoint uses is recorded in ``magpietts.conditioning``.

The generated GGUF stores the autoregressive MagpieTTS model weights and
metadata needed by the runtime. The NanoCodec vocoder referenced by the NeMo
config is not bundled in the Magpie .nemo archive and is not converted here.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import gguf
import numpy as np
import torch

from .source import extract_archive, find_checkpoint_files, load_state_dict, read_checkpoint_config
from .tts_tokenizer_profiles import tokenizer_profile

SPECIAL_AUDIO_TOKENS = 8
SPEAKER_NAMES = ["John", "Sofia", "Aria", "Jason", "Leo"]

# The plain model and the GRPO preference-optimization wrapper produce the same
# inference graph; only the training harness around it differs.
SUPPORTED_TARGETS = (
    "nemo.collections.tts.models.magpietts.MagpieTTSModel",
    "nemo.collections.tts.models.magpietts_preference_optimization.MagpieTTSModelOnlinePO",
)

# Modules that exist only for training and must never reach the GGUF. The GRPO
# checkpoints carry a full torchaudio SQUIM reward model (106 tensors).
TRAINING_ONLY_PREFIXES = ("squim_objective_model.", "_speaker_verification_model.", "_codec_model.")

CONDITIONING_BAKED = "baked"
CONDITIONING_CONTEXT_ENCODER = "context_encoder"


def is_lightning_checkpoint(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() == ".ckpt"


def read_sidecar_config(path: Path) -> dict[str, Any]:
    """Read a model config that lives outside the checkpoint.

    Accepts either the model config itself or a Lightning `hparams.yaml`, which
    nests it under a top-level `cfg:` key.
    """
    import yaml

    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise RuntimeError(f"sidecar config is not a mapping: {path}")
    if "cfg" in value and isinstance(value["cfg"], dict):
        value = value["cfg"]
    return value


def read_lightning_state_dict(path: Path) -> dict[str, torch.Tensor]:
    """Load a bare Lightning checkpoint.

    `weights_only=True` cannot be used: these carry an omegaconf DictConfig in
    `hyper_parameters`. The file is the caller's own training output, and only
    tensors are read out of it.
    """
    blob = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(blob, dict) or "state_dict" not in blob:
        raise RuntimeError(f"not a Lightning checkpoint (no state_dict): {path}")
    return {
        k: v for k, v in blob["state_dict"].items() if isinstance(v, torch.Tensor)
    }


def extract_nemo(path: Path) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    if path.is_dir():
        return path, None

    tmp = tempfile.TemporaryDirectory(prefix="magpietts-nemo-")
    out = Path(tmp.name)
    extract_archive(path, out, basenames={"model_config.yaml", "model_weights.ckpt"})
    return out, tmp


def read_config(root: Path) -> dict[str, Any]:
    return read_checkpoint_config(root)


def read_state_dict(root: Path) -> dict[str, torch.Tensor]:
    checkpoint = find_checkpoint_files(root)["weights"]
    if checkpoint is None:
        raise RuntimeError(f"checkpoint contains no model_weights.ckpt: {root}")
    return load_state_dict(checkpoint, weights_only=True)


def cfg_get(cfg: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def add_i32(writer: gguf.GGUFWriter, key: str, value: Any) -> None:
    writer.add_int32(key, int(value))


def add_f32(writer: gguf.GGUFWriter, key: str, value: Any) -> None:
    writer.add_float32(key, float(value))


def add_bool(writer: gguf.GGUFWriter, key: str, value: Any) -> None:
    writer.add_bool(key, bool(value))


def add_i32_array(writer: gguf.GGUFWriter, key: str, value: Any) -> None:
    if value is None:
        return
    items = [int(x) for x in value]
    if items:
        writer.add_array(key, items)


BAKED_TENSORS = (
    "baked_context_embedding.weight",
    "_baked_embedding_T",
    "_baked_embedding_D",
    "baked_context_embedding_len",
)


def detect_conditioning(sd: dict[str, torch.Tensor]) -> str:
    """Decide from the weights, not the config.

    NeMo drops the context encoder when it bakes the embeddings and vice versa,
    so the tensors are authoritative; `has_baked_context_embedding` is not even
    set in some sidecar configs.
    """
    baked = [name for name in BAKED_TENSORS if name in sd]
    has_context_encoder = any(k.startswith("context_encoder.") for k in sd)

    if baked and has_context_encoder:
        raise RuntimeError(
            "checkpoint has both baked context embeddings and a context encoder; "
            "cannot tell which conditioning path it was trained for"
        )
    if baked:
        missing = [name for name in BAKED_TENSORS if name not in sd]
        if missing:
            raise RuntimeError(f"baked conditioning is missing {', '.join(missing)}")
        return CONDITIONING_BAKED
    if has_context_encoder:
        return CONDITIONING_CONTEXT_ENCODER
    raise RuntimeError(
        "checkpoint has neither baked context embeddings nor a context encoder; "
        "this is not a supported MagpieTTS decoder_ce checkpoint"
    )

def _indexed_weight_indices(sd: dict[str, torch.Tensor], prefix: str) -> list[int]:
    suffix = ".weight"
    indices: list[int] = []
    for name in sd:
        if not name.startswith(prefix) or not name.endswith(suffix):
            continue
        raw_index = name[len(prefix) : -len(suffix)]
        if not raw_index.isascii() or not raw_index.isdigit():
            raise ValueError(f"{prefix} contains a non-numeric weight index: {name}")
        indices.append(int(raw_index))
    return indices


def _require_contiguous_indices(label: str, indices: list[int], expected_count: int) -> None:
    expected = list(range(expected_count))
    actual = sorted(indices)
    if actual != expected:
        raise ValueError(
            f"{label} indexes must be contiguous from 0 through {expected_count - 1}: "
            f"found={actual}"
        )


def add_metadata(
    writer: gguf.GGUFWriter, cfg: dict[str, Any], sd: dict[str, torch.Tensor]
) -> dict[str, Any]:
    encoder = cfg["encoder"]
    decoder = cfg["decoder"]
    lt_hidden = int(cfg.get("local_transformer_hidden_dim", 256))
    frame_stacking = int(cfg.get("frame_stacking_factor", 1))
    audio_embedding_indices = _indexed_weight_indices(sd, "audio_embeddings.")
    n_stacked_codebooks = len(audio_embedding_indices)
    if frame_stacking < 1 or n_stacked_codebooks == 0 or n_stacked_codebooks % frame_stacking:
        raise ValueError(
            "audio embedding count must be a positive multiple of frame_stacking_factor: "
            f"embeddings={n_stacked_codebooks} frame_stacking_factor={frame_stacking}"
        )
    _require_contiguous_indices("audio embedding", audio_embedding_indices, n_stacked_codebooks)

    audio_vocab = int(sd["audio_embeddings.0.weight"].shape[0])
    codebook_size = audio_vocab - SPECIAL_AUDIO_TOKENS
    text_vocab = int(sd["text_embedding.weight"].shape[0])

    conditioning = detect_conditioning(sd)
    baked_t = baked_d = 0
    baked_lens: list[int] = []
    baked_speakers = 0
    if conditioning == CONDITIONING_BAKED:
        baked_t = int(sd["_baked_embedding_T"].item())
        baked_d = int(sd["_baked_embedding_D"].item())
        baked_lens = [int(x) for x in sd["baked_context_embedding_len"].tolist()]
        baked_speakers = int(sd["baked_context_embedding.weight"].shape[0])

    # One embedding table per (codebook, stack slot); NeMo indexes them as
    # `c + i * C`, so the table count is C * stacking and the real codebook
    # count - what the codec consumes - is the quotient.
    emit_codebooks = int(
        len([k for k in sd if k.startswith("audio_embeddings.") and k.endswith(".weight")])
    )
    if emit_codebooks % frame_stacking != 0:
        raise RuntimeError(
            f"{emit_codebooks} audio embedding tables is not a multiple of "
            f"frame_stacking_factor {frame_stacking}"
        )
    n_codebooks = emit_codebooks // frame_stacking
    profile = tokenizer_profile(cfg, text_vocab, frame_stacking)

    # The local transformer emits one projection per table, and the final
    # projection is (tables * vocab). Both must agree or the runtime will read
    # the wrong slice of the logits.
    n_lt_out = len(
        [k for k in sd if k.startswith("local_transformer_out_projections.") and k.endswith(".weight")]
    )
    if n_lt_out != emit_codebooks:
        raise RuntimeError(
            f"expected {emit_codebooks} local transformer output projections, found {n_lt_out}"
        )
    final_rows = int(sd["final_proj.weight"].shape[0])
    if final_rows != emit_codebooks * audio_vocab:
        raise RuntimeError(
            f"final_proj has {final_rows} rows; expected {emit_codebooks} * {audio_vocab}"
        )

    inf = cfg.get("inference_parameters") or {}

    has_lt_in_projection = "local_transformer_in_projection.weight" in sd
    if not has_lt_in_projection and lt_hidden != int(cfg.get("embedding_dim", decoder["d_model"])):
        # NeMo only omits the projection when it would be square; without it the
        # runtime has to treat the step as identity, which is only valid then.
        raise RuntimeError(
            f"no local_transformer_in_projection, but local_transformer_hidden_dim "
            f"{lt_hidden} != embedding_dim {cfg.get('embedding_dim')}"
        )

    context_encoder = cfg.get("context_encoder") or {}
    if conditioning == CONDITIONING_CONTEXT_ENCODER and not context_encoder:
        raise RuntimeError("checkpoint has context_encoder weights but no context_encoder config")

    summary: dict[str, Any] = {
        "architecture": "magpietts",
        "model_type": cfg.get("model_type"),
        "nemo_target": cfg.get("target"),
        "nemo_version": cfg.get("nemo_version"),
        "tokenizer_profile": profile,
        "codec_model": cfg.get("codecmodel_path"),
        "conditioning": conditioning,
        "text_vocab_size": text_vocab,
        "audio_codebooks": n_codebooks,
        "emit_codebooks": emit_codebooks,
        "stacked_audio_codebooks": n_stacked_codebooks,
        "audio_codebook_size": codebook_size,
        "audio_vocab_size": audio_vocab,
        "frame_stacking_factor": frame_stacking,
        "embedding_dim": int(cfg.get("embedding_dim", decoder["d_model"])),
        "encoder_layers": int(encoder["n_layers"]),
        "decoder_layers": int(decoder["n_layers"]),
        "context_length": int(decoder["max_length_causal_mask"]),
        "has_local_transformer_in_projection": has_lt_in_projection,
        "context_duration_max": float(cfg.get("context_duration_max", 0.0))
        if conditioning == CONDITIONING_CONTEXT_ENCODER
        else 0.0,
        "speaker_names": SPEAKER_NAMES if conditioning == CONDITIONING_BAKED else [],
        "baked_context_length": baked_t,
        "baked_context_dim": baked_d,
        "baked_context_lens": baked_lens,
        "baked_speakers": baked_speakers,
    }

    writer.add_name(
        "NVIDIA MagpieTTS multilingual 357M"
        if conditioning == CONDITIONING_BAKED
        else "NVIDIA MagpieTTS zero-shot"
    )
    writer.add_description(
        "MagpieTTS autoregressive codec-token generator converted from NeMo to GGUF"
    )
    writer.add_string("magpietts.nemo_target", str(cfg.get("target", "")))
    writer.add_string("magpietts.nemo_version", str(cfg.get("nemo_version", "")))
    writer.add_string("magpietts.tokenizer_profile", profile)
    writer.add_string("magpietts.model_type", str(cfg.get("model_type", "")))
    writer.add_string("magpietts.codec_model", str(cfg.get("codecmodel_path", "")))
    writer.add_string("magpietts.config_json", json.dumps(cfg, ensure_ascii=False, sort_keys=True))
    writer.add_string("magpietts.conditioning", conditioning)
    add_bool(writer, "magpietts.local_transformer.has_in_projection", has_lt_in_projection)
    if conditioning == CONDITIONING_BAKED:
        writer.add_array("magpietts.speaker_names", SPEAKER_NAMES)
        writer.add_array("magpietts.baked_context_lens", baked_lens)
    else:
        add_i32(writer, "magpietts.context_encoder.layers", int(context_encoder["n_layers"]))
        add_i32(writer, "magpietts.context_encoder.heads", int(context_encoder["sa_n_heads"]))
        add_i32(
            writer, "magpietts.context_encoder.kernel_size", int(context_encoder["kernel_size"])
        )
        add_bool(
            writer, "magpietts.context_encoder.causal", bool(context_encoder.get("is_causal", False))
        )
        add_i32(
            writer,
            "magpietts.context_encoder.max_positions",
            int(sd["context_encoder.position_embeddings.weight"].shape[0]),
        )
        # The dataset pads the conditioning sequence out to a fixed length
        # derived from context_duration_max (text_to_speech_dataset.py:
        # `int(context_duration_max * sample_rate / samples_per_frame) + 2`),
        # and the context encoder then runs over the padding as valid
        # positions. The runtime needs the duration to reproduce that, because
        # the frame rate is a property of the codec, not of this checkpoint.
        add_f32(
            writer,
            "magpietts.context_encoder.max_duration_s",
            float(cfg.get("context_duration_max", 0.0)),
        )

    add_i32(writer, "magpietts.text_vocab_size", text_vocab)
    add_i32(writer, "magpietts.audio_codebooks", n_codebooks)
    add_i32(writer, "magpietts.emit_codebooks", emit_codebooks)
    add_i32(writer, "magpietts.stacked_audio_codebooks", n_stacked_codebooks)
    add_i32(writer, "magpietts.audio_codebook_size", codebook_size)
    add_i32(writer, "magpietts.audio_vocab_size", audio_vocab)
    add_i32(writer, "magpietts.audio_bos_id", codebook_size + 0)
    add_i32(writer, "magpietts.audio_eos_id", codebook_size + 1)
    add_i32(writer, "magpietts.context_audio_bos_id", codebook_size + 2)
    add_i32(writer, "magpietts.context_audio_eos_id", codebook_size + 3)
    add_i32(writer, "magpietts.mask_token_id", codebook_size + 4)
    add_i32(writer, "magpietts.frame_stacking_factor", frame_stacking)
    add_i32(writer, "magpietts.embedding_dim", summary["embedding_dim"])
    add_i32(writer, "magpietts.ffn_dim", int(decoder["d_ffn"]))
    add_i32(writer, "magpietts.context_length", summary["context_length"])
    add_i32(writer, "magpietts.encoder.layers", int(encoder["n_layers"]))
    add_i32(writer, "magpietts.encoder.heads", int(encoder["sa_n_heads"]))
    add_i32(writer, "magpietts.encoder.kernel_size", int(encoder["kernel_size"]))
    add_i32(writer, "magpietts.decoder.layers", int(decoder["n_layers"]))
    add_i32(writer, "magpietts.decoder.heads", int(decoder["sa_n_heads"]))
    add_i32(writer, "magpietts.decoder.cross_heads", int(decoder["xa_n_heads"]))
    add_i32(writer, "magpietts.decoder.cross_head_dim", int(decoder["xa_d_head"]))
    add_i32(writer, "magpietts.decoder.kernel_size", int(decoder["kernel_size"]))
    add_i32(
        writer, "magpietts.local_transformer.layers", int(cfg.get("local_transformer_n_layers", 1))
    )
    add_i32(
        writer, "magpietts.local_transformer.heads", int(cfg.get("local_transformer_n_heads", 1))
    )
    add_i32(writer, "magpietts.local_transformer.hidden_dim", lt_hidden)
    add_i32(
        writer,
        "magpietts.local_transformer.context_length",
        int(sd["local_transformer.position_embeddings.weight"].shape[0]),
    )
    if conditioning == CONDITIONING_BAKED:
        add_i32(writer, "magpietts.baked_context_length", baked_t)
        add_i32(writer, "magpietts.baked_context_dim", baked_d)
        add_i32(writer, "magpietts.baked_speakers", baked_speakers)
    add_i32(writer, "magpietts.inference.max_decoder_steps", inf.get("max_decoder_steps", 500))
    add_i32(writer, "magpietts.inference.topk", inf.get("topk", 80))
    add_i32(writer, "magpietts.inference.min_generated_frames", inf.get("min_generated_frames", 4))
    add_f32(writer, "magpietts.inference.temperature", inf.get("temperature", 0.6))
    add_f32(writer, "magpietts.inference.cfg_scale", inf.get("cfg_scale", 2.5))
    add_bool(
        writer,
        "magpietts.inference.apply_attention_prior",
        inf.get("apply_attention_prior", False),
    )
    add_f32(
        writer,
        "magpietts.inference.attention_prior_epsilon",
        inf.get("attention_prior_epsilon", 0.1),
    )
    add_i32(
        writer,
        "magpietts.inference.attention_prior_lookahead_window",
        inf.get("attention_prior_lookahead_window", 5),
    )
    add_i32(
        writer,
        "magpietts.inference.start_prior_after_n_audio_steps",
        inf.get("start_prior_after_n_audio_steps", 0),
    )
    add_i32(
        writer,
        "magpietts.inference.attention_prior_advance_threshold",
        inf.get("attention_prior_advance_threshold", 8),
    )
    add_i32(
        writer,
        "magpietts.inference.attention_prior_decay_threshold",
        inf.get("attention_prior_decay_threshold", 10),
    )
    add_i32_array(
        writer,
        "magpietts.inference.estimate_alignment_from_layers",
        inf.get("estimate_alignment_from_layers"),
    )
    add_i32_array(
        writer,
        "magpietts.inference.apply_prior_to_layers",
        inf.get("apply_prior_to_layers"),
    )

    return summary


def should_store_f32(name: str, tensor: torch.Tensor) -> bool:
    if not tensor.is_floating_point():
        return False
    if tensor.ndim <= 1:
        return True
    return name.endswith(".bias")


def tensor_to_numpy(name: str, tensor: torch.Tensor, outtype: str) -> np.ndarray:
    tensor = tensor.detach().cpu().contiguous()
    if tensor.is_floating_point():
        if outtype == "f16" and not should_store_f32(name, tensor):
            tensor = tensor.to(torch.float16)
        else:
            tensor = tensor.to(torch.float32)
    elif tensor.dtype == torch.int64:
        tensor = tensor.to(torch.int32)
    return tensor.numpy()


def add_tensor(writer: gguf.GGUFWriter, name: str, tensor: torch.Tensor, outtype: str) -> None:
    writer.add_tensor(name, tensor_to_numpy(name, tensor, outtype))


def is_local_transformer_tensor(name: str) -> bool:
    return name.startswith(
        (
            "audio_embeddings.",
            "local_transformer.",
            "local_transformer_in_projection.",
            "local_transformer_out_projections.",
        )
    )


def add_tensors(
    writer: gguf.GGUFWriter,
    sd: dict[str, torch.Tensor],
    outtype: str,
    local_transformer_outtype: str | None,
) -> tuple[int, list[str]]:
    n_written = 0
    skipped: list[str] = []

    for name, tensor in sd.items():
        if name.endswith(".causal_mask"):
            skipped.append(name)
            continue
        if name.startswith(TRAINING_ONLY_PREFIXES):
            raise RuntimeError(f"training-only tensor reached the writer: {name}")

        tensor_outtype = (
            local_transformer_outtype
            if local_transformer_outtype and is_local_transformer_tensor(name)
            else outtype
        )

        if name.endswith(".pos_ff.proj.conv.weight") or name.endswith(".pos_ff.o_net.conv.weight"):
            if tensor.ndim != 3:
                raise ValueError(
                    f"expected Conv1d weight to be rank 3: {name} {tuple(tensor.shape)}"
                )
            kernel = int(tensor.shape[2])
            if kernel == 1:
                add_tensor(writer, name + ".k0", tensor[:, :, 0], tensor_outtype)
                n_written += 1
            else:
                for k in range(kernel):
                    add_tensor(writer, f"{name}.k{k}", tensor[:, :, k], tensor_outtype)
                    n_written += 1
            continue

        add_tensor(writer, name, tensor, tensor_outtype)
        n_written += 1

    return n_written, skipped


def convert(
    source: Path,
    output: Path,
    outtype: str = "f16",
    metadata_json: Path | None = None,
    local_transformer_outtype: str | None = None,
    config_yaml: Path | None = None,
) -> None:
    root: Path | None = None
    tmp: tempfile.TemporaryDirectory[str] | None = None
    if is_lightning_checkpoint(source):
        if config_yaml is None:
            raise RuntimeError(
                f"{source.name} is a bare Lightning checkpoint and carries no usable "
                "model config; pass --config-yaml with the config the model is served under"
            )
        cfg = read_sidecar_config(config_yaml)
        sd = read_lightning_state_dict(source)
    else:
        root, tmp = extract_nemo(source)
        cfg = read_sidecar_config(config_yaml) if config_yaml else read_config(root)
        sd = read_state_dict(root)
    try:

        if cfg.get("target") not in SUPPORTED_TARGETS:
            raise RuntimeError(
                f"unsupported target: {cfg.get('target')}\n"
                f"       supported: {', '.join(SUPPORTED_TARGETS)}\n"
                "       a Lightning .ckpt embeds the training target in its own hparams; "
                "pass the sidecar config the server loads instead"
            )
        if cfg.get("model_type") != "decoder_ce":
            raise RuntimeError(
                f"unsupported model_type: {cfg.get('model_type')} (expected decoder_ce)"
            )

        sd = {k: v for k, v in sd.items() if not k.startswith(TRAINING_ONLY_PREFIXES)}

        output.parent.mkdir(parents=True, exist_ok=True)
        writer = gguf.GGUFWriter(output, "magpietts")
        summary = add_metadata(writer, cfg, sd)
        n_written, skipped = add_tensors(writer, sd, outtype, local_transformer_outtype)

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

        summary["tensors_written"] = n_written
        summary["tensors_skipped"] = skipped
        summary["output"] = str(output)
        summary["local_transformer_outtype"] = local_transformer_outtype or outtype
        if metadata_json:
            metadata_json.parent.mkdir(parents=True, exist_ok=True)
            metadata_json.write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )

        print(f"wrote {output}")
        print(f"stored {n_written} tensors; skipped {len(skipped)} deterministic causal masks")
    finally:
        if tmp is not None:
            tmp.cleanup()
