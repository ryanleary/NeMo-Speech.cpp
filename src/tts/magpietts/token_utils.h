// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace nemo_speech::tts {

// One decoder step emits `codebooks * stacking` codes. NeMo lays the embedding
// tables out as `c + i * C` (codebook c, stack slot i), so the codes for codec
// frame i are the contiguous run starting at `i * C`.
//
// Appends the `stacking` codec frames, in playback order, to `out`.
inline void
split_stacked_frame(
    const std::vector<int32_t>& emitted, int codebooks, int stacking,
    std::vector<std::vector<int32_t>>& out) {
    for (int i = 0; i < stacking; ++i) {
        const size_t start = (size_t)i * (size_t)codebooks;
        out.emplace_back(emitted.begin() + start, emitted.begin() + start + codebooks);
    }
}

// Inverse of split_stacked_frame: interleave `stacking` codec frames back into
// one decoder-step vector. `frames` must hold exactly `stacking` entries.
inline std::vector<int32_t>
join_stacked_frame(const std::vector<std::vector<int32_t>>& frames, int codebooks) {
    std::vector<int32_t> emitted;
    emitted.reserve(frames.size() * (size_t)codebooks);
    for (const auto& frame : frames) {
        emitted.insert(emitted.end(), frame.begin(), frame.begin() + codebooks);
    }
    return emitted;
}

inline std::vector<int32_t>
flatten_token_chunks(const std::vector<std::vector<int32_t>>& token_chunks) {
    std::vector<int32_t> tokens;
    size_t total = 0;
    for (const auto& chunk : token_chunks) {
        total += chunk.size();
    }
    tokens.reserve(total);
    for (const auto& chunk : token_chunks) {
        tokens.insert(tokens.end(), chunk.begin(), chunk.end());
    }
    return tokens;
}

}  // namespace nemo_speech::tts
