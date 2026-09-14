// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <cmath>
#include <cstdio>
#include <vector>

#include "tts/magpietts/model.h"

namespace tts = nemo_speech::tts;

namespace {

bool
near(float a, float b) {
    return std::fabs(a - b) < 1.0e-6f;
}

bool
expect(bool cond, const char* msg) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        return false;
    }
    return true;
}

bool
has_active_prior(const std::vector<float>& prior, float epsilon) {
    for (float v : prior) {
        if (v > epsilon) {
            return true;
        }
    }
    return false;
}

tts::magpietts_hparams
base_hparams() {
    tts::magpietts_hparams h;
    h.apply_attention_prior = true;
    h.attention_prior_epsilon = 0.1f;
    h.attention_prior_lookahead_window = 3;
    h.start_prior_after_n_audio_steps = 0;
    h.n_cross_head = 1;
    h.n_cross_dhead = 128;
    return h;
}

}  // namespace

constexpr int kSink = nemo_speech::tts::kMagpieAttendedSinkAdvance;

int
main() {
    bool ok = true;

    {
        tts::MagpieAttentionPriorState state;
        tts::magpietts_hparams h = base_hparams();
        std::vector<float> scores = {0.0f, 0.1f, 0.2f, 0.9f, 0.8f, 0.7f, 0.6f, 0.5f, 0.4f, 0.3f};
        state.update(h, 0, (int)scores.size(), scores);
        const std::vector<float>& prior = state.prior();
        ok &= expect((int)prior.size() == (int)scores.size(), "prior length");
        if ((int)prior.size() >= 7) {
            ok &= expect(state.lastAttended() == 3, "the search takes the peak in its window");
            ok &= expect(near(prior[0], 0.1f), "history before exposed window uses epsilon");
            ok &= expect(near(prior[1], 0.1f), "before the exposed window uses epsilon");
            ok &= expect(near(prior[2], 1.0f), "previous timestep is exposed");
            ok &= expect(near(prior[3], 1.0f), "current timestep is exposed");
            ok &= expect(near(prior[4], 1.0f), "future timestep +1 is exposed");
            ok &= expect(near(prior[5], 1.0f), "future timestep +2 is exposed");
            ok &= expect(near(prior[6], 1.0f), "future timestep +3 is exposed");
            ok &= expect(near(prior[7], 0.1f), "outside lookahead uses epsilon");
        } else {
            ok &= expect(false, "prior size too small for indexed assertions");
        }

        state.update(h, 1, (int)scores.size(), scores);
        ok &= expect(state.lastAttended() == 3, "attention stays on the peak while it leads");
    }

    {
        tts::MagpieAttentionPriorState state;
        tts::magpietts_hparams h = base_hparams();
        std::vector<float> scores = {0.0f, 1.0f, 0.9f, 0.8f, 0.7f, 0.6f, 0.5f, 0.4f, 0.3f, 0.2f};
        for (int step = 0; step < kSink; ++step) {
            state.update(h, step, (int)scores.size(), scores);
        }
        ok &= expect(state.lastAttended() == 1, "early attention holds until the sink threshold");
        state.update(h, kSink, (int)scores.size(), scores);
        ok &= expect(state.lastAttended() == 2, "early attention advances once it is a sink");
    }

    {
        tts::MagpieAttentionPriorState state;
        tts::magpietts_hparams h = base_hparams();
        h.attention_prior_lookahead_window = 5;
        std::vector<float> scores = {0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 1.0f, 0.0f, 0.0f, 0.0f};
        for (int step = 0; step < 1 + kSink; ++step) {
            state.update(h, step, (int)scores.size(), scores);
        }
        ok &= expect(state.lastAttended() == 6, "near-end attention holds until it is a sink");
        state.update(h, 1 + kSink, (int)scores.size(), scores);
        ok &= expect(
            state.lastAttended() == (int)scores.size() - 1,
            "an empty search window means the sentence has ended");
    }

    {
        tts::MagpieAttentionPriorState state;
        tts::magpietts_hparams h = base_hparams();
        h.attention_prior_lookahead_window = 8;
        std::vector<float> scores(8, 0.0f);
        scores[4] = 1.0f;
        for (int step = 0; step < 1 + kSink; ++step) {
            state.update(h, step, (int)scores.size(), scores);
        }
        ok &= expect(state.lastAttended() == 7, "single-chunk attention reaches final token");
        state.update(h, 1 + kSink, (int)scores.size(), scores);
        ok &= expect(
            has_active_prior(state.prior(), h.attention_prior_epsilon),
            "single-chunk prior keeps active focus after final-token decay");
        ok &= expect(near(state.prior().back(), 1.0f), "single-chunk final token remains exposed");
    }

    {
        tts::MagpieAttentionPriorState state;
        tts::magpietts_hparams h = base_hparams();
        std::vector<float> scores = {0.0f, 0.3f, 0.2f, 0.1f, 0.4f};
        state.update(h, 0, (int)scores.size(), scores);
        for (float v : state.prior()) {
            ok &= expect(near(v, 1.0f), "short text disables restrictive prior");
        }
    }

    {
        tts::MagpieAttentionPriorState state;
        tts::magpietts_hparams h = base_hparams();
        h.start_prior_after_n_audio_steps = 2;
        std::vector<float> scores = {0.0f, 0.3f, 0.2f, 0.1f, 0.4f, 0.5f};
        state.update(h, 1, (int)scores.size(), scores);
        ok &= expect(state.prior().empty(), "prior is not built before configured start step");
    }

    {
        tts::MagpieLongformAttentionPriorState state;
        tts::magpietts_hparams h = base_hparams();
        state.beginChunk(h, 0, 8, 8, true);
        std::vector<float> scores = {0.0f, 0.0f, 0.0f, 0.9f, 0.8f, 0.7f, 0.6f, 0.5f};
        state.update(h, 0, (int)scores.size(), scores);
        ok &= expect(
            state.lastAttendedAbsolute() == 3,
            "longform first chunk jumps to best attention within lookahead");

        state.beginChunk(h, 2, 9, 4, false);
        scores = {0.0f, 0.0f, 0.1f, 0.2f, 0.9f, 0.8f, 0.7f, 0.6f, 0.5f};
        state.update(h, 0, (int)scores.size(), scores);
        ok &= expect(
            state.lastAttendedAbsolute() == 5,
            "longform later chunk jumps to best attention within lookahead");
        ok &= expect(
            state.lastAttendedRelative() == 3,
            "longform later chunk reports relative attended position");
    }

    {
        tts::MagpieLongformAttentionPriorState state;
        tts::magpietts_hparams h = base_hparams();
        h.attention_prior_lookahead_window = 5;
        state.beginChunk(h, 0, 10, 10, true);
        std::vector<float> scores = {0.0f, 0.1f, 1.0f, 0.8f, 0.7f, 0.6f, 0.5f, 0.4f, 0.3f, 0.2f};
        for (int step = 0; step < kSink; ++step) {
            state.update(h, step, (int)scores.size(), scores);
        }
        ok &= expect(
            state.lastAttendedAbsolute() == 2,
            "longform attention holds on an early token until it is a sink");
        state.update(h, kSink, (int)scores.size(), scores);
        ok &= expect(
            state.lastAttendedAbsolute() == 3, "longform early attention advances past the sink");
    }

    {
        tts::MagpieLongformAttentionPriorState state;
        tts::magpietts_hparams h = base_hparams();
        h.attention_prior_lookahead_window = 5;
        state.beginChunk(h, 0, 10, 10, true);
        std::vector<float> scores = {0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 1.0f, 0.0f, 0.0f, 0.0f};
        for (int step = 0; step < 1 + kSink; ++step) {
            state.update(h, step, (int)scores.size(), scores);
        }
        ok &= expect(
            state.lastAttendedAbsolute() == 6,
            "longform near-end attention holds until it is a sink");
        state.update(h, 1 + kSink, (int)scores.size(), scores);
        ok &= expect(
            state.lastAttendedAbsolute() == (int)scores.size() - 1,
            "an empty search window means the sentence has ended");
    }

    {
        tts::MagpieLongformAttentionPriorState state;
        tts::magpietts_hparams h = base_hparams();
        h.attention_prior_lookahead_window = 5;
        const int text_len = 20;
        std::vector<float> scores((size_t)text_len, 0.0f);
        scores[(size_t)text_len - 4] = 1.0f;
        h.attention_prior_lookahead_window = text_len;
        state.beginChunk(h, 0, text_len, text_len, true);
        for (int step = 0; step < 1 + kSink; ++step) {
            state.update(h, step, text_len, scores);
        }
        ok &= expect(
            state.lastAttendedAbsolute() == text_len - 1,
            "longform first chunk reaches final token without overshooting");
        state.update(h, 1 + kSink, text_len, scores);
        ok &= expect(
            has_active_prior(state.prior(), h.attention_prior_epsilon),
            "longform prior keeps active focus after final-token decay");
        ok &= expect(near(state.prior().back(), 1.0f), "longform final token remains exposed");
    }

    return ok ? 0 : 1;
}
