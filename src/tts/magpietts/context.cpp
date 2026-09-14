// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "context.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "graph.h"
#include "nvtx_utils.h"

namespace nemo_speech::tts {

namespace {

// NeMo wraps context codes in one BOS frame and one EOS frame
// (magpietts_modules.add_special_tokens) before stacking.
constexpr int kContextSpecialFrames = 2;

int
padded_frames(int frames, int stacking) {
    const int wrapped = frames + kContextSpecialFrames;
    return ((wrapped + stacking - 1) / stacking) * stacking;
}

}  // namespace

int
MagpieContextEncoder::paddedLength(double codec_fps) const {
    const auto& h = model_.hparams;
    if (h.ctx_enc_max_duration_s <= 0.0f || codec_fps <= 0.0) {
        return 0;
    }
    // Mirrors text_to_speech_dataset.py:
    //   int(context_duration_max * sample_rate / samples_per_frame) + 2
    return (int)((double)h.ctx_enc_max_duration_s * codec_fps) + kContextSpecialFrames;
}

bool
MagpieContextEncoder::encode(
    const std::vector<std::vector<int32_t>>& codes, int threads, double codec_fps,
    std::vector<float>& out, int& out_len) const {
    const ggml_nvtx::range nvtx_range("magpietts_context_encoder");
    const auto& h = model_.hparams;

    if (h.conditioning != MAGPIETTS_CONDITIONING_CONTEXT_ENCODER) {
        fprintf(stderr, "this checkpoint has no context encoder\n");
        return false;
    }
    if ((int)codes.size() != h.audio_codebooks) {
        fprintf(
            stderr, "context codes have %zu codebooks; the model expects %d\n", codes.size(),
            h.audio_codebooks);
        return false;
    }
    const int frames = codes.empty() ? 0 : (int)codes[0].size();
    if (frames <= 0) {
        fprintf(stderr, "context codes are empty\n");
        return false;
    }
    for (const auto& codebook : codes) {
        if ((int)codebook.size() != frames) {
            fprintf(stderr, "context codebooks have inconsistent lengths\n");
            return false;
        }
    }

    const int stacking = h.frame_stacking_factor;
    const int total = padded_frames(frames, stacking);
    const int embedded = total / stacking;

    // NeMo pads the conditioning sequence to a fixed length and lets the
    // context encoder attend over the padding as ordinary positions. Because
    // the encoder is bidirectional those rows change every output, so the
    // padding has to be reproduced rather than trimmed.
    const int padded = paddedLength(codec_fps);
    const int positions_out = padded > embedded ? padded : embedded;
    if (padded > 0 && embedded > padded) {
        fprintf(
            stderr,
            "reference audio is longer than the model's %.2fs context window "
            "(%d positions > %d); crop it first\n",
            (double)h.ctx_enc_max_duration_s, embedded, padded);
        return false;
    }
    if (positions_out > h.ctx_enc_max_positions) {
        fprintf(
            stderr,
            "reference audio is too long: %d context positions exceeds the model's %d\n",
            positions_out, h.ctx_enc_max_positions);
        return false;
    }

    // [BOS, codes..., EOS] then zero-padded up to the stacking multiple, which
    // is what pad_audio_codes does on the NeMo side.
    std::vector<std::vector<int32_t>> wrapped(
        (size_t)h.audio_codebooks, std::vector<int32_t>((size_t)total, 0));
    for (int c = 0; c < h.audio_codebooks; ++c) {
        wrapped[c][0] = h.context_audio_bos_id;
        for (int t = 0; t < frames; ++t) {
            wrapped[c][t + 1] = codes[c][t];
        }
        wrapped[c][frames + 1] = h.context_audio_eos_id;
    }

    // De-interleave into one input per embedding table. NeMo indexes the tables
    // as `c + i * C` and feeds them frames `i, i + stacking, ...`, so the split
    // happens here rather than as a strided view in the graph.
    std::vector<std::pair<std::string, std::vector<int32_t>>> i32_inputs;
    std::vector<std::pair<std::string, std::vector<float>>> f32_inputs;
    i32_inputs.reserve((size_t)h.emit_codebooks + 1);
    std::vector<std::vector<int32_t>> table_tokens((size_t)h.emit_codebooks);
    for (int i = 0; i < stacking; ++i) {
        for (int c = 0; c < h.audio_codebooks; ++c) {
            std::vector<int32_t>& dst = table_tokens[(size_t)c + (size_t)i * h.audio_codebooks];
            dst.assign((size_t)positions_out, 0);
            for (int t = 0; t < embedded; ++t) {
                dst[(size_t)t] = wrapped[c][(size_t)t * stacking + i];
            }
        }
    }

    ggml_context* ctx = new_graph_context();
    ggml_cgraph* gf = ggml_new_graph_custom(ctx, MAGPIETTS_MAX_NODES, false);

    ggml_tensor* sum = nullptr;
    for (int table = 0; table < h.emit_codebooks; ++table) {
        const std::string name = "magpietts_context_tokens_" + std::to_string(table);
        ggml_tensor* tok = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, positions_out);
        ggml_set_name(tok, name.c_str());
        ggml_set_input(tok);
        i32_inputs.push_back({name, table_tokens[(size_t)table]});
        ggml_tensor* emb = ggml_get_rows(ctx, model_.audio_embeddings[table], tok);
        sum = sum ? ggml_add(ctx, sum, emb) : emb;
    }
    ggml_tensor* x = ggml_scale(ctx, sum, 1.0f / (float)h.emit_codebooks);

    if (positions_out > embedded) {
        // Rows past the reference are zeros in NeMo, not the embedding of code
        // 0, so mask them explicitly rather than relying on the token value.
        std::vector<float> keep((size_t)positions_out, 0.0f);
        for (int t = 0; t < embedded; ++t) {
            keep[(size_t)t] = 1.0f;
        }
        ggml_tensor* keep_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, positions_out);
        ggml_set_name(keep_in, "magpietts_context_keep");
        ggml_set_input(keep_in);
        f32_inputs.push_back({"magpietts_context_keep", keep});
        x = ggml_mul(ctx, x, keep_in);
    }

    ggml_tensor* pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, positions_out);
    ggml_set_name(pos, "magpietts_context_positions");
    ggml_set_input(pos);
    i32_inputs.push_back({"magpietts_context_positions", positions(positions_out)});

    x = transformer_forward(ctx, model_.context_encoder, x, pos, nullptr);
    x = ggml_cont(ctx, ggml_cast(ctx, x, GGML_TYPE_F32));
    ggml_set_name(x, "magpietts_context_out");
    ggml_set_output(x);
    ggml_build_forward_expand(gf, x);

    ggml_gallocr_t allocr = nullptr;
    if (!compute_graph(model_, ctx, gf, i32_inputs, f32_inputs, threads, &allocr)) {
        ggml_free(ctx);
        return false;
    }

    out.resize((size_t)h.n_embd * (size_t)positions_out);
    MagpiePinnedHostScratch output_staging;
    magpietts_backend_tensor_get_staged(
        model_, output_staging, x, out.data(), 0, out.size() * sizeof(float));
    out_len = positions_out;

    // Debug hook for the parity check against NeMo: raw little-endian f32,
    // row-major [out_len][n_embd]. See scripts/tts/compare-context-prefix.py.
    if (const char* dump = std::getenv("MAGPIETTS_CONTEXT_DUMP")) {
        if (dump[0]) {
            if (FILE* fh = fopen(dump, "wb")) {
                fwrite(out.data(), sizeof(float), out.size(), fh);
                fclose(fh);
                fprintf(
                    stderr, "wrote context prefix [%d x %d] to %s\n", positions_out, h.n_embd,
                    dump);
            } else {
                fprintf(stderr, "failed to open MAGPIETTS_CONTEXT_DUMP path: %s\n", dump);
            }
        }
    }

    ggml_gallocr_free(allocr);
    ggml_free(ctx);
    return true;
}

}  // namespace nemo_speech::tts
