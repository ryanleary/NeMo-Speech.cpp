// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "model.h"

namespace nemo_speech::tts {

// Turns reference-audio codec codes into the decoder's conditioning prefix.
//
// This is the zero-shot counterpart of the baked embedding table: the prefix it
// produces is exactly what a baked row holds, computed per request instead of
// looked up. Only checkpoints converted with
// `magpietts.conditioning = context_encoder` carry the weights for it.
class MagpieContextEncoder {
   public:
    explicit MagpieContextEncoder(const magpietts_model& model) : model_(model) {}

    // `codes` is [audio_codebooks][frames] of raw codec codes for the reference
    // audio. Wraps them in the context BOS/EOS frames, pads to the frame
    // stacking factor, embeds, zero-pads to the model's fixed conditioning
    // length, and encodes.
    //
    // `codec_fps` is the codec's frame rate, needed to reproduce NeMo's fixed
    // padding length; pass 0 to skip the padding.
    //
    // `out` is filled row-major [out_len][n_embd] - the layout the decoder
    // expects for a supplied prefix.
    bool encode(
        const std::vector<std::vector<int32_t>>& codes, int threads, double codec_fps,
        std::vector<float>& out, int& out_len) const;

    // The fixed number of conditioning positions this model expects, or 0 when
    // the checkpoint declares no context duration (then the prefix is however
    // long the reference makes it).
    int paddedLength(double codec_fps) const;

   private:
    const magpietts_model& model_;
};

}  // namespace nemo_speech::tts
