// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "magpietts.h"

#include <algorithm>
#include <array>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <exception>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <thread>
#include <utility>

#include "audio_pp.h"
#include "decoder.h"
#include "encoder.h"
#include "ggml_log_filter.h"
#include "lt.h"
#include "nvtx_utils.h"
#include "token_utils.h"
#include "tts/nanocodec/model.h"

namespace nemo_speech::tts {

namespace nc = nemo_speech::tts::nanocodec;

static void
log_attention_prior_trace_values(
    const char* label, int step, const magpietts_hparams& h, const std::vector<float>& prior,
    int last_attended) {
    if (prior.empty()) {
        fprintf(
            stderr, "%s attention-prior step=%d prior=<none> last_attended=%d\n", label, step,
            last_attended);
        return;
    }

    fprintf(
        stderr, "%s attention-prior step=%d last_attended=%d epsilon=%.4f active_indices=", label,
        step, last_attended, h.attention_prior_epsilon);
    bool first = true;
    for (int i = 0; i < (int)prior.size(); ++i) {
        if (prior[(size_t)i] > h.attention_prior_epsilon) {
            fprintf(stderr, "%s%d:%.2f", first ? "" : ",", i, prior[(size_t)i]);
            first = false;
        }
    }
    if (first) {
        fprintf(stderr, "<none>");
    }
    fprintf(stderr, "\n");
}

static void
log_longform_attention_prior_trace(
    const char* label, int chunk_index, int step, const magpietts_hparams& h,
    const MagpieLongformAttentionPriorState& state) {
    char scoped_label[128];
    snprintf(scoped_label, sizeof(scoped_label), "%s chunk=%d", label, chunk_index);
    log_attention_prior_trace_values(
        scoped_label, step, h, state.prior(), state.lastAttendedRelative());
}

static size_t
token_count(const std::vector<std::vector<int32_t>>& chunks) {
    size_t total = 0;
    for (const auto& chunk : chunks) {
        total += chunk.size();
    }
    return total;
}


static bool
splice_longform_history_context(
    std::vector<float>& text_cond, int text_len, int current_chunk_len, int n_embd,
    const std::vector<float>& previous_context, int previous_context_len) {
    if (text_len < 0 || current_chunk_len < 0 || current_chunk_len > text_len || n_embd <= 0) {
        fprintf(stderr, "invalid longform text context dimensions\n");
        return false;
    }
    const int history_len = text_len - current_chunk_len;
    if (history_len <= 0) {
        return true;
    }
    if (previous_context_len < history_len ||
        previous_context.size() < (size_t)previous_context_len * (size_t)n_embd ||
        text_cond.size() < (size_t)text_len * (size_t)n_embd) {
        fprintf(
            stderr, "longform history context cache is too short: need %d token(s), have %d\n",
            history_len, previous_context_len);
        return false;
    }

    const float* src =
        previous_context.data() + (size_t)(previous_context_len - history_len) * (size_t)n_embd;
    std::copy(src, src + (size_t)history_len * (size_t)n_embd, text_cond.data());
    return true;
}


// Per-chunk decode state that survives across steps.
struct MagpieChunkDecodeState {
    int chunk_frames_generated = 0;
    int near_end_frames = 0;
    bool suppress_nonfinal_codec_output = false;
    int suppressed_nonfinal_frames = 0;
};

// What one decoder step did to one chunk.
struct MagpieStepOutcome {
    std::vector<std::vector<int32_t>> frames;
    bool stop = false;
};

// Everything a step implies for one chunk once its codes are sampled: advance
// the attention prior, decide whether the chunk has run to the end of its text,
// apply the non-final-EOS suppression rule, and split the stacked step into
// codec frames. Reads only this chunk's state, so a wave runs it once per item
// and the sequential loop runs it once.
static bool
advance_chunk_state(
    const magpietts_hparams& h, const magpie_stream_params& params, const char* label,
    size_t chunk_index, size_t chunk_count, int step, int text_len, bool final_chunk,
    bool forbid_eos, int frames_remaining, const std::vector<int32_t>& next_codes,
    const std::vector<int32_t>& argmax_codes, const std::vector<float>* alignment_scores,
    MagpieLongformAttentionPriorState& prior, MagpieChunkDecodeState& chunk,
    std::vector<std::vector<int32_t>>& audio_codes, MagpieStepOutcome& out) {
    out.frames.clear();
    out.stop = false;

    bool end_chunk_after_frame = false;
    bool start_suppressing_after_frame = false;
    bool reached_chunk_end = final_chunk;
    bool can_catch_up_nonfinal = false;
    if (alignment_scores && !alignment_scores->empty()) {
        prior.update(h, step, text_len, *alignment_scores);
        if (params.verbose) {
            log_longform_attention_prior_trace(label, (int)chunk_index, step, h, prior);
        }
        const int near_end_threshold = 3;
        const int last_rel = prior.lastAttendedRelative();
        reached_chunk_end = last_rel >= std::max(0, text_len - 1);
        if (!final_chunk) {
            can_catch_up_nonfinal = true;
            if (last_rel >= std::max(0, text_len - near_end_threshold)) {
                ++chunk.near_end_frames;
            } else {
                chunk.near_end_frames = 0;
            }
            if (chunk.suppress_nonfinal_codec_output) {
                end_chunk_after_frame = reached_chunk_end;
            } else if (
                chunk.near_end_frames >= 1 &&
                chunk.chunk_frames_generated >= h.min_generated_frames) {
                if (reached_chunk_end) {
                    end_chunk_after_frame = true;
                } else {
                    start_suppressing_after_frame = true;
                }
            }
        } else if (reached_chunk_end && chunk.chunk_frames_generated >= h.min_generated_frames) {
            end_chunk_after_frame = true;
        }
    }

    std::vector<std::vector<int32_t>> codec_frames;
    if (!magpietts_unstack_codes(next_codes, h, codec_frames)) {
        fprintf(stderr, "sampled an invalid stacked MagpieTTS frame\n");
        return false;
    }
    const int eos_lane = forbid_eos ? -1 : magpietts_first_eos_lane(next_codes, argmax_codes, h);
    const bool has_eos = eos_lane >= 0 && eos_lane < frames_remaining;
    if (has_eos) {
        ggml_nvtx::mark("magpietts_stream_eos");
        if (params.verbose) {
            fprintf(
                stderr, "%s EOS detected at frame %d for text chunk %zu/%zu\n", label, step,
                chunk_index + 1, chunk_count);
        }
        if (final_chunk || reached_chunk_end || !can_catch_up_nonfinal) {
            out.stop = true;
        }
        if (!chunk.suppress_nonfinal_codec_output) {
            chunk.suppress_nonfinal_codec_output = true;
            if (params.verbose) {
                fprintf(
                    stderr,
                    "%s suppressing codec output for text chunk %zu/%zu after non-final EOS "
                    "until attention reaches chunk end (relative=%d text_len=%d)\n",
                    label, chunk_index + 1, chunk_count, prior.lastAttendedRelative(), text_len);
            }
        }
    }

    for (int c = 0; c < h.audio_codebooks; ++c) {
        for (int lane = 0; lane < h.frame_stacking_factor; ++lane) {
            audio_codes[(size_t)c].push_back(next_codes[(size_t)(c + lane * h.audio_codebooks)]);
        }
    }

    const int frames_to_emit = magpietts_frames_to_emit(
        frames_remaining, h.frame_stacking_factor, has_eos ? eos_lane : -1);
    if (!chunk.suppress_nonfinal_codec_output) {
        for (int lane = 0; lane < frames_to_emit; ++lane) {
            out.frames.push_back(codec_frames[(size_t)lane]);
        }
        chunk.chunk_frames_generated += frames_to_emit;
    } else {
        chunk.suppressed_nonfinal_frames += frames_to_emit;
    }
    if (start_suppressing_after_frame) {
        chunk.suppress_nonfinal_codec_output = true;
        if (params.verbose) {
            fprintf(
                stderr,
                "%s continuing text chunk %zu/%zu without codec output until attention reaches "
                "chunk end (relative=%d text_len=%d)\n",
                label, chunk_index + 1, chunk_count, prior.lastAttendedRelative(), text_len);
        }
    }
    if (end_chunk_after_frame) {
        if (params.verbose) {
            fprintf(
                stderr,
                "%s ending text chunk %zu/%zu after attention reached chunk end (relative=%d "
                "text_len=%d suppressed_codec_frames=%d)\n",
                label, chunk_index + 1, chunk_count, prior.lastAttendedRelative(), text_len,
                chunk.suppressed_nonfinal_frames);
        }
        out.stop = true;
    }
    return true;
}

// The text-only half of starting a chunk: how much already-spoken text to carry
// into the window, and what that window is. Touches no decoder state, so a wave
// can run it for a whole group before decoding any of it.
//
// required_history is the one decode-time input, and it is only consulted on the
// adaptive path (longform_history_tokens < 0). A wave must not use that path --
// there chunk N's text really does depend on chunk N-1's decode -- so it passes
// required_history = 0 and keeps longform_history_tokens >= 0.
std::vector<int>
plan_wave_admission(const std::vector<char>& lane_idle, size_t pending, int threshold) {
    std::vector<int> lanes;
    if (pending == 0) {
        return lanes;
    }
    int live = 0;
    for (size_t lane = 0; lane < lane_idle.size(); ++lane) {
        if (lane_idle[lane]) {
            lanes.push_back((int)lane);
        } else {
            ++live;
        }
    }
    if (lanes.empty() || ((int)lanes.size() < std::max(1, threshold) && live > 0)) {
        lanes.clear();
        return lanes;
    }
    if (lanes.size() > pending) {
        lanes.resize(pending);
    }
    return lanes;
}

MagpieChunkPlan
plan_text_chunk(
    const magpietts_hparams& h, const magpie_stream_params& params,
    const std::vector<int32_t>& prior_text_tokens, const std::vector<int32_t>& current_tokens,
    int absolute_token_offset, int required_history, int available_history) {
    MagpieChunkPlan plan;
    const int max_history = std::max(0, h.n_ctx - (int)current_tokens.size());
    // The history tokens are spliced from the *previous chunk's* encoder output, so the
    // window can never reach back further than that one chunk -- which is unrelated to
    // how many tokens have been seen in total. Clamping only by prior_text_tokens asks
    // for history the cache cannot supply as soon as a chunk is shorter than the
    // requested history, which ordinary prose produces routinely: a short sentence makes
    // a short chunk, and the next chunk then asks for more than it left behind.
    const int history_cap =
        std::min<int>((int)prior_text_tokens.size(), std::max(0, available_history));
    const int default_history =
        std::min<int>({history_cap, (int)current_tokens.size(), 20, max_history});
    plan.history_len =
        params.longform_history_tokens >= 0
            ? std::min<int>({params.longform_history_tokens, history_cap, max_history})
            : std::min<int>(
                  history_cap, std::min(max_history, std::max(default_history, required_history)));
    plan.left_offset = absolute_token_offset - plan.history_len;
    plan.text_window.reserve((size_t)plan.history_len + current_tokens.size());
    if (plan.history_len > 0) {
        plan.text_window.insert(
            plan.text_window.end(), prior_text_tokens.end() - plan.history_len,
            prior_text_tokens.end());
    }
    plan.text_window.insert(plan.text_window.end(), current_tokens.begin(), current_tokens.end());
    plan.text_len = (int)plan.text_window.size();
    return plan;
}

static std::vector<std::vector<int32_t>>
select_token_chunks(
    const magpie_stream_params& params, const std::vector<std::vector<int32_t>>& token_chunks) {
    std::vector<std::vector<int32_t>> selected;
    for (const auto& chunk : token_chunks) {
        if (!chunk.empty()) {
            selected.push_back(chunk);
        }
    }
    if (selected.empty() && !params.tokens.empty()) {
        selected.push_back(params.tokens);
    }
    if (selected.empty()) {
        return selected;
    }
    if (params.longform_mode == MAGPIE_LONGFORM_OFF) {
        return {flatten_token_chunks(selected)};
    }
    if (params.longform_mode == MAGPIE_LONGFORM_AUTO && selected.size() <= 1) {
        return selected;
    }
    return selected;
}

#if defined(MAGPIETTS_CUDA_SAMPLING)
struct MagpieCudaSamplerDeleter {
    void operator()(magpietts_cuda_sampler* sampler) const { magpietts_cuda_sampler_free(sampler); }
};
#endif

class MagpieStreamingWorkspace {
   private:
    magpietts_model& magpie_;

   public:
    MagpieStreamingWorkspace(magpietts_model& magpie, const nc::NanoCodecModel& codec)
        : magpie_(magpie), encoder(magpie), decoder(magpie), local_sampler(magpie, 1),
          codec_decoder(codec) {}

    // A wave samples `batch` items per round, so the sampler's code and top-k
    // buffers have to hold stacked_codebooks * batch slots, not just one item's.
    bool beginRequest(int threads, bool use_cuda_sampling, int audio_codebooks, int batch = 1) {
        audio_codebooks *= batch > 0 ? batch : 1;
        local_sampler.setThreads(threads);
        if (local_transformer_cpu_sampler) {
            local_transformer_cpu_sampler->setThreads(threads);
        }
        if (local_transformer_fp32_cpu_sampler) {
            local_transformer_fp32_cpu_sampler->setThreads(threads);
        }
        if (local_transformer_fp32_cuda_sampler) {
            local_transformer_fp32_cuda_sampler->setThreads(threads);
        }
        cond_kv.clear();
        uncond_kv.clear();
        cond_cross_kv.clear();
        codec_stream_state.clear();

#if defined(MAGPIETTS_CUDA_SAMPLING)
        if (use_cuda_sampling && (!cuda_sampler || cuda_sampler_codebooks != audio_codebooks)) {
            cuda_sampler.reset(magpietts_cuda_sampler_create(audio_codebooks));
            cuda_sampler_codebooks = cuda_sampler ? audio_codebooks : 0;
            if (!cuda_sampler) {
                fprintf(stderr, "failed to create CUDA sampler\n");
                return false;
            }
        }
#else
        if (use_cuda_sampling) {
            fprintf(stderr, "CUDA sampling was not compiled into this MagpieTTS build\n");
            return false;
        }
#endif
        return true;
    }

    LocalCodebookSampler* localSampler(bool use_cuda_lt, bool fp32, int threads) {
        if (fp32) {
            MagpieModel& fp32_model =
                use_cuda_lt ? local_transformer_fp32_cuda_model : local_transformer_fp32_cpu_model;
            std::unique_ptr<LocalCodebookSampler>& fp32_sampler =
                use_cuda_lt ? local_transformer_fp32_cuda_sampler
                            : local_transformer_fp32_cpu_sampler;
            if (!fp32_sampler) {
                if (!magpietts_model_init_local_transformer_fp32(
                        magpie_, fp32_model, use_cuda_lt)) {
                    return nullptr;
                }
                fp32_sampler = std::make_unique<LocalCodebookSampler>(fp32_model, threads);
            }
            fp32_sampler->setThreads(threads);
            return fp32_sampler.get();
        }
        if (use_cuda_lt || !magpietts_backend_is_cuda(magpie_.backend)) {
            local_sampler.setThreads(threads);
            return &local_sampler;
        }
        if (!local_transformer_cpu_sampler) {
            if (!magpietts_model_init_local_transformer_cpu(magpie_, local_transformer_cpu_model)) {
                return nullptr;
            }
            local_transformer_cpu_sampler =
                std::make_unique<LocalCodebookSampler>(local_transformer_cpu_model, threads);
        }
        local_transformer_cpu_sampler->setThreads(threads);
        return local_transformer_cpu_sampler.get();
    }

    bool prewarmLocalTransformer(
        LocalCodebookSampler& sampler, bool use_cuda_lt, bool fp32, bool use_cfg, bool verbose) {
        const bool use_cpu_mirror = !use_cuda_lt && magpietts_backend_is_cuda(magpie_.backend);
        bool& prewarmed =
            fp32 ? (use_cuda_lt ? (use_cfg ? local_transformer_fp32_cuda_pair_prewarmed
                                           : local_transformer_fp32_cuda_single_prewarmed)
                                : (use_cfg ? local_transformer_fp32_cpu_pair_prewarmed
                                           : local_transformer_fp32_cpu_single_prewarmed))
                 : (use_cpu_mirror ? (use_cfg ? local_transformer_cpu_pair_prewarmed
                                              : local_transformer_cpu_single_prewarmed)
                                   : (use_cfg ? local_transformer_main_pair_prewarmed
                                              : local_transformer_main_single_prewarmed));
        if (prewarmed) {
            return true;
        }
        if (verbose) {
            fprintf(
                stderr, "prewarming MagpieTTS local-transformer %s precision=%s %s graphs\n",
                use_cuda_lt ? "CUDA" : "CPU", fp32 ? "fp32" : "native",
                use_cfg ? "CFG pair" : "single");
        }
        if (!sampler.prewarm(use_cfg, 2)) {
            fprintf(stderr, "failed to prewarm MagpieTTS local-transformer graphs\n");
            return false;
        }
        prewarmed = true;
        if (verbose) {
            fprintf(
                stderr, "prewarmed MagpieTTS local-transformer %s precision=%s %s graphs\n",
                use_cuda_lt ? "CUDA" : "CPU", fp32 ? "fp32" : "native",
                use_cfg ? "CFG pair" : "single");
        }
        return true;
    }

#if defined(MAGPIETTS_CUDA_SAMPLING)
    magpietts_cuda_sampler* cudaSampler() const { return cuda_sampler.get(); }
#endif

    MagpieEncoder encoder;
    MagpieDecoder decoder;
    LocalCodebookSampler local_sampler;
    nc::NanoCodecDecoder codec_decoder;
    std::vector<float> text_cond;
    magpietts_backend_tensor text_cond_device;
    magpietts_backend_tensor cond_hidden_device;
    magpietts_backend_tensor uncond_hidden_device;
    DecoderKvCache cond_kv;
    DecoderKvCache uncond_kv;
    DecoderCrossKvCache cond_cross_kv;
    // Declared after the state: the graph holds nodes pointing at the state's
    // cache tensors, so it must be destroyed first.
    nc::NanoCodecStreamState codec_stream_state;
    nc::NanoCodecStreamGraph codec_stream_graph;

   private:
    MagpieModel local_transformer_cpu_model;
    std::unique_ptr<LocalCodebookSampler> local_transformer_cpu_sampler;
    MagpieModel local_transformer_fp32_cpu_model;
    MagpieModel local_transformer_fp32_cuda_model;
    std::unique_ptr<LocalCodebookSampler> local_transformer_fp32_cpu_sampler;
    std::unique_ptr<LocalCodebookSampler> local_transformer_fp32_cuda_sampler;
#if defined(MAGPIETTS_CUDA_SAMPLING)
    std::unique_ptr<magpietts_cuda_sampler, MagpieCudaSamplerDeleter> cuda_sampler;
    int cuda_sampler_codebooks = 0;
#endif
    bool local_transformer_main_single_prewarmed = false;
    bool local_transformer_main_pair_prewarmed = false;
    bool local_transformer_cpu_single_prewarmed = false;
    bool local_transformer_cpu_pair_prewarmed = false;
    bool local_transformer_fp32_cpu_single_prewarmed = false;
    bool local_transformer_fp32_cpu_pair_prewarmed = false;
    bool local_transformer_fp32_cuda_single_prewarmed = false;
    bool local_transformer_fp32_cuda_pair_prewarmed = false;
};

class MagpieStreamingRuntime::Impl {
   public:
    MagpieModel magpie;
    nc::NanoCodecModel codec;
    std::unique_ptr<MagpieStreamingWorkspace> workspace;
};

namespace {
GgmlLogFilter magpie_ggml_logs;
}

MagpieStreamingRuntime::MagpieStreamingRuntime() : impl_(std::make_unique<Impl>()) {}

MagpieStreamingRuntime::~MagpieStreamingRuntime() = default;

bool
MagpieStreamingRuntime::load(
    const std::string& magpie_model, const std::string& codec_model, magpietts_uma_mode uma_mode,
    bool magpie_cpu, bool codec_cpu, bool verbose) {
    magpie_ggml_logs.set_verbose(verbose);
    ggml_log_set(GgmlLogFilter::callback, &magpie_ggml_logs);
    impl_->workspace.reset();
    if (!impl_->magpie.load(magpie_model, uma_mode, magpie_cpu, verbose)) {
        return false;
    }
    if (!impl_->codec.load(codec_model, codec_cpu, verbose)) {
        impl_->magpie.reset();
        return false;
    }
    impl_->workspace = std::make_unique<MagpieStreamingWorkspace>(impl_->magpie, impl_->codec);
    return true;
}

int
MagpieStreamingRuntime::sampleRate() const {
    return impl_ ? impl_->codec.sampleRate() : 0;
}

int
MagpieStreamingRuntime::speakerCount() const {
    return impl_ ? impl_->magpie.hparams.baked_speakers : 0;
}

std::vector<std::string>
MagpieStreamingRuntime::speakerNames() const {
    std::vector<std::string> out;
    if (!impl_ || !impl_->magpie.gguf) {
        return out;
    }
    const int64_t id = gguf_find_key(impl_->magpie.gguf, "magpietts.speaker_names");
    if (id < 0 || gguf_get_kv_type(impl_->magpie.gguf, id) != GGUF_TYPE_ARRAY ||
        gguf_get_arr_type(impl_->magpie.gguf, id) != GGUF_TYPE_STRING) {
        return out;
    }
    const size_t n = gguf_get_arr_n(impl_->magpie.gguf, id);
    out.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        const char* value = gguf_get_arr_str(impl_->magpie.gguf, id, i);
        if (value && value[0]) {
            out.emplace_back(value);
        }
    }
    return out;
}

const std::string&
MagpieStreamingRuntime::tokenizerProfile() const {
    static const std::string empty;
    return impl_ ? impl_->magpie.tokenizer_profile : empty;
}

int
MagpieStreamingRuntime::textVocabSize() const {
    return impl_ ? impl_->magpie.hparams.text_vocab_size : 0;
}

void
stream_latency_metrics::begin(int64_t now_us) {
    *this = {};
    start_us = now_us;
    inter_event_min_ms = std::numeric_limits<double>::max();
}

double
stream_latency_metrics::record_event(int64_t now_us, bool& first_event) {
    first_event = first_event_us == 0;
    double inter_ms = 0.0;
    if (first_event) {
        first_event_us = now_us;
        first_event_ms = start_us > 0 ? (double)(now_us - start_us) / 1000.0 : 0.0;
    } else {
        inter_ms = (double)(now_us - last_event_us) / 1000.0;
        inter_event_sum_ms += inter_ms;
        inter_event_min_ms = std::min(inter_event_min_ms, inter_ms);
        inter_event_max_ms = std::max(inter_event_max_ms, inter_ms);
        inter_event_ms.push_back(inter_ms);
    }
    last_event_us = now_us;
    ++events;
    return inter_ms;
}

void
stream_latency_metrics::finish(int64_t now_us) {
    elapsed_s = start_us > 0 ? (double)(now_us - start_us) / 1000000.0 : 0.0;
}

double
stream_latency_metrics::inter_event_avg_ms() const {
    return events > 1 ? inter_event_sum_ms / (double)(events - 1) : 0.0;
}

double
stream_latency_metrics::inter_event_min_value_ms() const {
    return events > 1 ? inter_event_min_ms : 0.0;
}

double
stream_latency_metrics::inter_event_percentile_ms(double percentile) const {
    if (inter_event_ms.empty()) {
        return 0.0;
    }
    std::vector<double> sorted = inter_event_ms;
    std::sort(sorted.begin(), sorted.end());
    const double clamped = std::max(0.0, std::min(100.0, percentile));
    const size_t rank = (size_t)std::ceil((clamped / 100.0) * (double)sorted.size());
    const size_t idx = rank == 0 ? 0 : std::min(rank - 1, sorted.size() - 1);
    return sorted[idx];
}

double
stream_latency_metrics::inter_event_p95_ms() const {
    return inter_event_percentile_ms(95.0);
}

double
stream_latency_metrics::inter_event_p99_ms() const {
    return inter_event_percentile_ms(99.0);
}

void
stream_run_metrics::begin() {
    *this = {};
    start_us = ggml_time_us();
    decoder.begin(start_us);
    codec.begin(start_us);
    e2e.begin(start_us);
}

double
stream_run_metrics::record_decoder_frame(int64_t now_us, bool& first_frame) {
    return decoder.record_event(now_us, first_frame);
}

double
stream_run_metrics::record_codec_chunk(int64_t now_us, bool& first_chunk) {
    const double inter_ms = codec.record_event(now_us, first_chunk);
    chunks = codec.events;
    return inter_ms;
}

double
stream_run_metrics::record_audio_write(int64_t now_us, bool& first_write) {
    const double inter_ms = e2e.record_event(now_us, first_write);
    e2e_chunks = e2e.events;
    ttfa_ms = e2e.first_event_ms;
    return inter_ms;
}

void
stream_run_metrics::add_codec_work(double elapsed_s, double audio_s) {
    codec_elapsed_s += elapsed_s;
    codec_audio_s += audio_s;
    codec_rtfx = codec_elapsed_s > 0.0 ? codec_audio_s / codec_elapsed_s : 0.0;
}

void
stream_run_metrics::finish(uint64_t samples_written, int sample_rate, double codec_fps) {
    const int64_t now_us = ggml_time_us();
    this->samples_written = samples_written;
    if (decoder.elapsed_s == 0.0) {
        decoder.finish(now_us);
    }
    if (codec.elapsed_s == 0.0) {
        codec.finish(now_us);
    }
    e2e.finish(now_us);
    e2e_elapsed_s = start_us > 0 ? (double)(now_us - start_us) / 1000000.0 : 0.0;
    audio_s = sample_rate > 0 ? (double)samples_written / (double)sample_rate : 0.0;
    e2e_rtf = audio_s > 0.0 ? e2e_elapsed_s / audio_s : 0.0;
    e2e_rtfx = e2e_elapsed_s > 0.0 ? audio_s / e2e_elapsed_s : 0.0;
    decoder_audio_s = codec_fps > 0.0 ? (double)generated_frames / codec_fps : 0.0;
    decoder_rtfx = decoder.elapsed_s > 0.0 ? decoder_audio_s / decoder.elapsed_s : 0.0;
    codec_rtfx = codec_elapsed_s > 0.0 ? codec_audio_s / codec_elapsed_s : 0.0;
    chunks = codec.events;
    e2e_chunks = e2e.events;
    ttfa_ms = e2e.first_event_ms;
}

double
stream_run_metrics::inter_chunk_avg_ms() const {
    return e2e.inter_event_avg_ms();
}

double
stream_run_metrics::inter_chunk_min_value_ms() const {
    return e2e.inter_event_min_value_ms();
}

struct stream_audio_outputs {
    int sample_rate = 0;
    uint64_t samples_written = 0;
    stream_run_metrics* metrics = nullptr;
    std::function<bool(const std::vector<uint8_t>&)> pcm_callback;

    bool write_audio(const std::vector<float>& audio) {
        const ggml_nvtx::range nvtx_range("magpietts_stream_audio_write");
        std::vector<uint8_t> bytes;
        bytes.reserve(audio.size() * 2);
        for (float x : audio) {
            x = std::max(-1.0f, std::min(1.0f, x));
            const int32_t v = (int32_t)std::lrintf(x * 32767.0f);
            const int16_t s = (int16_t)v;
            bytes.push_back((uint8_t)((uint16_t)s & 0xff));
            bytes.push_back((uint8_t)(((uint16_t)s >> 8) & 0xff));
        }

        if (pcm_callback && !pcm_callback(bytes)) {
            fprintf(stderr, "failed to write PCM callback output\n");
            return false;
        }

        samples_written += audio.size();
        if (metrics && !audio.empty()) {
            bool first_write = false;
            metrics->record_audio_write(ggml_time_us(), first_write);
        }
        return true;
    }
};

struct stream_code_writer {
    std::ofstream out;
    bool enabled = false;

    bool open(const std::string& path) {
        if (path.empty()) {
            return true;
        }
        out.open(path);
        if (!out) {
            fprintf(stderr, "failed to open %s for writing codec frames\n", path.c_str());
            return false;
        }
        enabled = true;
        return true;
    }

    bool write_frame(const std::vector<int32_t>& frame) {
        const ggml_nvtx::range nvtx_range("magpietts_stream_write_codec_frame");
        if (!enabled) {
            return true;
        }
        for (int c = 0; c < (int)frame.size(); ++c) {
            if (c) {
                out << ' ';
            }
            out << frame[c];
        }
        out << '\n';
        out.flush();
        return (bool)out;
    }
};

static bool
load_forced_code_frames(
    const char* path, int audio_codebooks, std::vector<std::vector<int32_t>>& frames) {
    frames.clear();
    if (!path || !path[0]) {
        return true;
    }

    std::ifstream in(path);
    if (!in) {
        fprintf(stderr, "failed to open MagpieTTS forced-code file: %s\n", path);
        return false;
    }

    std::string line;
    int line_no = 0;
    while (std::getline(in, line)) {
        ++line_no;
        if (line.empty()) {
            continue;
        }
        std::vector<int32_t> frame;
        try {
            frame = parse_token_list(line);
        }
        catch (const std::exception& e) {
            fprintf(
                stderr, "failed to parse forced-code frame %d in %s: %s\n", line_no, path,
                e.what());
            return false;
        }
        if ((int)frame.size() != audio_codebooks) {
            fprintf(
                stderr, "forced-code frame %d in %s has %zu codebooks, expected %d\n", line_no,
                path, frame.size(), audio_codebooks);
            return false;
        }
        frames.push_back(std::move(frame));
    }
    return true;
}

static bool
make_codec_chunk(
    const nc::NanoCodecHParams& h, const std::vector<std::vector<int32_t>>& generated,
    nc::NanoCodecFrames& frames) {
    const ggml_nvtx::range nvtx_range("magpietts_stream_make_codec_chunk");
    if (h.num_codebooks != 8) {
        fprintf(
            stderr, "NanoCodec model has %d codebooks; this streaming runner expects 8\n",
            h.num_codebooks);
        return false;
    }

    frames.resize(generated.size());
    for (size_t i = 0; i < generated.size(); ++i) {
        if ((int)generated[i].size() != h.num_codebooks) {
            fprintf(
                stderr, "frame %zu has %zu codebooks; expected %d\n", i, generated[i].size(),
                h.num_codebooks);
            return false;
        }
        for (int c = 0; c < h.num_codebooks; ++c) {
            const int32_t token = generated[i][c];
            if (token < 0 || token >= h.codebook_size) {
                fprintf(
                    stderr, "frame %zu codebook %d token %d outside codec codebook size %d\n", i, c,
                    token, h.codebook_size);
                return false;
            }
            frames[i][c] = token;
        }
    }
    return true;
}

static bool
decode_and_stream_chunk(
    const nc::NanoCodecModel& codec, const nc::NanoCodecDecoder& decoder,
    nc::NanoCodecStreamState* stream_state, nc::NanoCodecStreamGraph* stream_graph,
    const std::vector<std::vector<int32_t>>& chunk, int threads, stream_audio_outputs& outputs,
    AudioPostProcessor& audio_pp, stream_run_metrics* metrics, const char* run_label,
    int chunk_index, int history_frames, bool final_chunk, bool verbose) {
    const ggml_nvtx::range nvtx_range("magpietts_stream_decode_and_write_chunk");
    if (chunk.empty()) {
        return true;
    }

    nc::NanoCodecFrames codec_frames;
    if (!make_codec_chunk(codec.hparams(), chunk, codec_frames)) {
        return false;
    }

    std::vector<float> audio;
    const int64_t t_start = ggml_time_us();
    if (metrics && metrics->codec.events == 0) {
        metrics->codec.begin(t_start);
    }
    bool decoded = false;
    if (stream_state) {
        if (!stream_graph) {
            fprintf(stderr, "stateful codec stream requested without a persistent graph\n");
            return false;
        }
        if (!stream_graph->initialized()) {
            fprintf(stderr, "stateful codec stream graph was not initialized\n");
            return false;
        }
        // Short chunks are zero-padded and trimmed inside the decoder, so a partial final
        // chunk reuses the graph rather than forcing a rebuild.
        if ((int)codec_frames.size() > stream_graph->chunkFrames()) {
            fprintf(
                stderr, "codec chunk has %zu frames, over the graph's %d\n", codec_frames.size(),
                stream_graph->chunkFrames());
            return false;
        }
        decoded = decoder.decodeStream(*stream_state, *stream_graph, codec_frames, threads, audio);
    } else {
        decoded = decoder.decode(codec_frames, threads, audio);
    }
    if (!decoded) {
        return false;
    }
    const int64_t decoded_us = ggml_time_us();
    bool first_chunk = false;
    double inter_ms = 0.0;
    if (metrics) {
        inter_ms = metrics->record_codec_chunk(decoded_us, first_chunk);
    }
    const double elapsed_ms = (double)(decoded_us - t_start) / 1000.0;
    const double audio_s =
        codec.sampleRate() > 0 ? (double)audio.size() / (double)codec.sampleRate() : 0.0;
    const double elapsed_s = elapsed_ms / 1000.0;
    if (metrics) {
        metrics->add_codec_work(elapsed_s, audio_s);
    }
    if (!audio_pp.writeDecodedAudio(
            audio, history_frames, final_chunk,
            [&](const std::vector<float>& processed) { return outputs.write_audio(processed); })) {
        return false;
    }

    const double rtf = audio_s > 0.0 ? elapsed_s / audio_s : 0.0;
    const double rtfx = elapsed_s > 0.0 ? audio_s / elapsed_s : 0.0;
    if (verbose && (chunk_index < 4 || chunk_index % 10 == 0)) {
        const char* label = run_label ? run_label : "stream";
        if (metrics && first_chunk) {
            fprintf(
                stderr,
                "%s codec chunk %d: %zu frames history=%d -> %zu decoded samples in %.2f ms "
                "rtf=%.4f rtfx=%.2f ttfa=%.2f ms%s\n",
                label, chunk_index, chunk.size(), history_frames, audio.size(), elapsed_ms, rtf,
                rtfx, metrics->ttfa_ms, final_chunk ? " (final)" : "");
        } else if (metrics) {
            fprintf(
                stderr,
                "%s codec chunk %d: %zu frames history=%d -> %zu decoded samples in %.2f ms "
                "rtf=%.4f rtfx=%.2f inter=%.2f ms%s\n",
                label, chunk_index, chunk.size(), history_frames, audio.size(), elapsed_ms, rtf,
                rtfx, inter_ms, final_chunk ? " (final)" : "");
        } else {
            fprintf(
                stderr,
                "%s codec chunk %d: %zu frames history=%d -> %zu decoded samples in %.2f ms "
                "rtf=%.4f rtfx=%.2f%s\n",
                label, chunk_index, chunk.size(), history_frames, audio.size(), elapsed_ms, rtf,
                rtfx, final_chunk ? " (final)" : "");
        }
    }
    return true;
}

struct codec_read_result {
    std::vector<std::vector<int32_t>> frames;
    int history_frames = 0;
    int chunk_index = 0;
    bool final_read = false;
};

struct codec_stream_worker {
    const nc::NanoCodecModel& codec;
    nc::NanoCodecDecoder& decoder;
    int threads = 1;
    stream_audio_outputs& outputs;
    AudioPostProcessor audio_pp;
    nc::NanoCodecStreamState& stream_state;
    nc::NanoCodecStreamGraph& stream_graph;
    stream_run_metrics* metrics = nullptr;
    const char* run_label = "stream";
    int chunk_size = 3;
    int history_size = 1;
    int future_size = 1;
    bool true_stateful = true;
    bool verbose = false;
    size_t max_buffered_frames = 16;

    std::vector<std::vector<int32_t>> audio_codes;
    std::mutex mutex;
    std::condition_variable has_work;
    std::condition_variable has_room;
    std::thread worker;
    bool input_closed = false;
    bool abort_requested = false;
    bool failed = false;
    bool is_last_token_in = false;
    bool send_final_audio = false;
    bool final_audio_sent = false;
    std::string error;
    int read_idx = 0;
    int write_idx = 0;
    int last_token_id = -1;
    int chunks_done = 0;

    codec_stream_worker(
        const nc::NanoCodecModel& codec_, nc::NanoCodecDecoder& decoder_,
        nc::NanoCodecStreamState& stream_state_, nc::NanoCodecStreamGraph& stream_graph_,
        int threads_, stream_audio_outputs& outputs_, int samples_per_frame, int chunk_size_,
        int history_size_, int future_frames, int window_samples, size_t queue_depth,
        bool true_stateful_, stream_run_metrics* metrics_, const char* run_label_, bool verbose_)
        : codec(codec_), decoder(decoder_), threads(threads_), outputs(outputs_),
          audio_pp(samples_per_frame, future_frames, window_samples), stream_state(stream_state_),
          stream_graph(stream_graph_), metrics(metrics_),
          run_label(run_label_ ? run_label_ : "stream"), chunk_size(std::max(1, chunk_size_)),
          history_size(std::max(0, history_size_)), future_size(std::max(0, future_frames)),
          true_stateful(true_stateful_), verbose(verbose_) {
        max_buffered_frames = std::max<size_t>(1, queue_depth) * (size_t)chunk_size +
                              (size_t)history_size + (size_t)future_size + 1;
    }

    void start() { worker = std::thread(&codec_stream_worker::run, this); }

    bool write_frame(const std::vector<int32_t>& frame) {
        const ggml_nvtx::range nvtx_range("magpietts_stream_queue_write_frame");
        std::unique_lock<std::mutex> lock(mutex);
        has_room.wait(lock, [&] {
            const int buffered = write_idx - read_idx;
            return buffered < (int)max_buffered_frames || failed || abort_requested || input_closed;
        });
        if (failed || abort_requested || input_closed) {
            return false;
        }
        audio_codes.push_back(frame);
        ++write_idx;
        has_work.notify_one();
        return true;
    }

    bool is_failed() {
        std::lock_guard<std::mutex> lock(mutex);
        return failed;
    }

    std::vector<int32_t> eos_frame() const {
        static const int32_t nemo_eos[8] = {621, 1455, 1184, 1038, 463, 377, 1536, 1742};
        std::vector<int32_t> frame(codec.numCodebooks(), 0);
        for (int c = 0; c < codec.numCodebooks(); ++c) {
            int32_t token = c < 8 ? nemo_eos[c] : 0;
            if (token < 0 || token >= codec.codebookSize()) {
                token = 0;
            }
            frame[c] = token;
        }
        return frame;
    }

    std::vector<int32_t> silence_frame() const {
        static const int32_t nemo_silence[8] = {621, 1455, 1184, 1038, 463, 377, 1536, 1742};
        std::vector<int32_t> frame(codec.numCodebooks(), 0);
        for (int c = 0; c < codec.numCodebooks(); ++c) {
            int32_t token = c < 8 ? nemo_silence[c] : 0;
            if (token < 0 || token >= codec.codebookSize()) {
                token = 0;
            }
            frame[c] = token;
        }
        return frame;
    }

    void finish_tokens() {
        std::lock_guard<std::mutex> lock(mutex);
        if (!is_last_token_in) {
            last_token_id = write_idx;
            is_last_token_in = true;
            for (int i = 0; i < future_size; ++i) {
                audio_codes.push_back(eos_frame());
                ++write_idx;
            }
        }
        input_closed = true;
        has_work.notify_one();
    }

    void close_input() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            input_closed = true;
        }
        has_work.notify_one();
    }

    bool join() {
        close_input();
        if (worker.joinable()) {
            worker.join();
        }

        std::lock_guard<std::mutex> lock(mutex);
        if (failed && !error.empty()) {
            fprintf(stderr, "codec worker failed: %s\n", error.c_str());
        }
        return !failed;
    }

    void cancel() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            abort_requested = true;
            input_closed = true;
        }
        has_work.notify_all();
        has_room.notify_all();
        if (worker.joinable()) {
            worker.join();
        }
    }

    void set_failed(const std::string& message) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            failed = true;
            if (error.empty()) {
                error = message;
            }
            input_closed = true;
        }
        has_work.notify_all();
        has_room.notify_all();
    }

    bool has_tokens_locked() const {
        const int diff = write_idx - read_idx;
        if (true_stateful) {
            if (diff <= 0) {
                return false;
            }
            if (is_last_token_in || input_closed) {
                return true;
            }
            return diff >= chunk_size;
        }
        if (diff <= future_size) {
            return false;
        }
        if (read_idx == 0 && diff > future_size) {
            return true;
        }
        if (read_idx == 1 && diff > future_size) {
            return true;
        }
        if (is_last_token_in && diff > future_size) {
            return true;
        }
        return diff >= chunk_size;
    }

    codec_read_result read_tokens_locked() {
        codec_read_result out;
        const int end = write_idx;
        if (true_stateful) {
            if (end <= read_idx) {
                return out;
            }
            const int chunk_end = std::min(end, read_idx + chunk_size);
            out.history_frames = 0;
            out.chunk_index = chunks_done;
            out.final_read = is_last_token_in && last_token_id <= chunk_end;
            out.frames.assign(audio_codes.begin() + read_idx, audio_codes.begin() + chunk_end);
            read_idx = chunk_end;
            has_room.notify_one();
            return out;
        }
        if (end - read_idx <= future_size) {
            return out;
        }
        const int start = std::max(0, read_idx - history_size);
        out.history_frames = read_idx - start;
        out.chunk_index = chunks_done;
        out.final_read = is_last_token_in && last_token_id <= end;
        out.frames.assign(audio_codes.begin() + start, audio_codes.begin() + end);
        read_idx = end - future_size;
        has_room.notify_one();
        return out;
    }

    bool next_work(codec_read_result& item, bool& flush_final) {
        const ggml_nvtx::range nvtx_range("magpietts_stream_worker_next_work");
        std::unique_lock<std::mutex> lock(mutex);
        has_work.wait(lock, [&] {
            return abort_requested || send_final_audio || input_closed || has_tokens_locked();
        });
        if (abort_requested) {
            return false;
        }
        if (has_tokens_locked()) {
            item = read_tokens_locked();
            return !item.frames.empty();
        }
        if (send_final_audio) {
            send_final_audio = false;
            final_audio_sent = true;
            flush_final = true;
            return true;
        }
        if (input_closed) {
            return false;
        }
        return false;
    }

    // Build the fixed-size graph and push one throwaway chunk through it before any real
    // tokens arrive. That moves the graph allocation and the backend's first-run graph
    // capture off the first audio chunk's latency, while the acoustic model is still
    // generating. The caches are zeroed afterwards, so the stream still starts from silence.
    bool prewarm() {
        const ggml_nvtx::range nvtx_range("magpietts_stream_codec_prewarm");
        // A workspace reused across requests keeps its graph, and with it the backend's
        // captured version of it; zeroing the caches is all a fresh stream needs.
        if (stream_graph.initialized() && stream_graph.chunkFrames() == chunk_size) {
            stream_state.clear();
            return true;
        }
        if (!decoder.initStreamGraph(stream_state, chunk_size, stream_graph)) {
            set_failed("failed to initialize the codec stream graph");
            return false;
        }
        const nc::NanoCodecFrames warm((size_t)chunk_size, nc::NanoCodecFrame{});
        std::vector<float> discard;
        if (!decoder.decodeStream(stream_state, stream_graph, warm, threads, discard)) {
            set_failed("failed to warm up the codec stream graph");
            return false;
        }
        stream_state.clear();
        return true;
    }

    void run() {
        const ggml_nvtx::range nvtx_range("magpietts_stream_codec_worker_run");
        if (verbose) {
            fprintf(
                stderr,
                "codec worker started: chunk_size=%d history=%d future=%d max_buffered=%zu\n",
                chunk_size, history_size, future_size, max_buffered_frames);
        }
        if (true_stateful && !prewarm()) {
            return;
        }
        try {
            for (;;) {
                codec_read_result item;
                bool flush_final = false;
                if (!next_work(item, flush_final)) {
                    break;
                }
                if (flush_final) {
                    if (!audio_pp.flush([&](const std::vector<float>& processed) {
                            return outputs.write_audio(processed);
                        })) {
                        set_failed("failed to flush final audio");
                        return;
                    }
                    continue;
                }
                // Reads are capped at chunk_size and a short final chunk is zero-padded by
                // the decoder, so one graph serves the whole stream.
                nc::NanoCodecStreamGraph* graph = true_stateful ? &stream_graph : nullptr;
                if (!decode_and_stream_chunk(
                        codec, decoder, true_stateful ? &stream_state : nullptr, graph, item.frames,
                        threads, outputs, audio_pp, metrics, run_label, item.chunk_index,
                        item.history_frames, item.final_read, verbose)) {
                    set_failed("decode or audio output failed");
                    return;
                }
                ++chunks_done;
                if (item.final_read) {
                    std::lock_guard<std::mutex> lock(mutex);
                    send_final_audio = true;
                    has_work.notify_one();
                }
            }

            bool should_flush = false;
            {
                std::lock_guard<std::mutex> lock(mutex);
                should_flush = !abort_requested && !failed && !final_audio_sent;
            }
            if (should_flush && !audio_pp.flush([&](const std::vector<float>& processed) {
                    return outputs.write_audio(processed);
                })) {
                set_failed("failed to flush overlap audio");
                return;
            }
        }
        catch (const std::exception& e) {
            set_failed(e.what());
            return;
        }
        catch (...) {
            set_failed("unknown exception");
            return;
        }
        if (verbose) {
            fprintf(stderr, "codec worker stopped after %d chunks\n", chunks_done);
        }
    }
};

// One chunk of one request: everything the decoder reads for it and everything
// it writes back. A lane holds a pointer to one of these and nothing else, which
// is what lets a wave carry chunks from unrelated requests.
struct WaveItem {
    size_t chunk_index = 0;
    const std::vector<int32_t>* current_tokens = nullptr;
    int text_len = 0;
    int left_offset = 0;
    std::vector<float> text_cond;
    magpietts_backend_tensor text_cond_device;
    DecoderCrossKvCache cross_kv;
    std::vector<std::vector<int32_t>> audio_codes;
    MagpieLongformAttentionPriorState prior;
    std::vector<std::vector<int32_t>> frames;
    MagpieChunkDecodeState chunk;
    // This chunk's own decode step. Two lanes of the same wave are at different
    // steps, so the step is the chunk's, not the loop's.
    int step = 0;
    bool done = false;
};

// One request's worth of work: its chunks, the conditioning that rolls between
// them, where its audio goes, and how far it has drained. Everything here
// belongs to one caller -- nothing in it is shared with another request, which
// is the property that lets one wave serve several.
//
// What it deliberately does NOT own is the decoder, the lane table or the
// sampler. Those are the engine's, and a session reaches them only by being
// admitted.
struct WaveSession {
    // Borrowed from the process.
    magpietts_model& magpie;
    const MagpieEncoder& encoder;
    // Borrowed from the request.
    const magpietts_hparams& h;
    const magpie_stream_params& params;
    const std::vector<std::vector<int32_t>>& token_chunks;
    const char* label;
    stream_run_metrics& metrics;
    stream_code_writer& code_writer;
    codec_stream_worker& codec_worker;
    MagpiePinnedHostScratch& text_context_staging;
    int& frames_generated;
    int& decoder_frames_generated;
    std::mt19937& boundary_silence_rng;
    std::uniform_int_distribution<int>& boundary_silence_dist;

    // Owned.
    std::vector<size_t> chunk_ids;
    std::vector<std::unique_ptr<WaveItem>> plan;
    // Chunk N's conditioning splices from chunk N-1's ENCODER output, never
    // from its decode -- `required_history` is 0 below and the adaptive rule is
    // gated off for waves. So these roll forward as chunks are prepared, and the
    // decodes they produce are independent of each other.
    std::vector<int32_t> seen_tokens;
    std::vector<float> carry_cond;
    int carry_len = 0;
    int absolute = 0;
    // The widest text window any of this request's chunks needs.
    int text_capacity = 0;
    // Cursors: the next chunk to admit, and the next to hand the codec.
    size_t next_chunk = 0;
    size_t next_drain = 0;
    // Count frames and drop them instead of streaming them. Lets several
    // sessions share one engine before per-session codec state exists, which is
    // what the decode-throughput measurement needs.
    bool discard_audio = false;
    int64_t discarded_frames = 0;

    // Collect the non-empty chunks and the widest window they can need. Cheap:
    // no encoding, just enough to size the run.
    bool plan_chunks() {
        for (size_t ci = 0; ci < token_chunks.size(); ++ci) {
            const std::vector<int32_t>& current = token_chunks[ci];
            if (current.empty()) {
                continue;
            }
            if ((int)current.size() > h.n_ctx) {
                fprintf(
                    stderr, "%s text chunk %zu has %zu tokens, exceeding model context %d\n", label,
                    ci, current.size(), h.n_ctx);
                return false;
            }
            chunk_ids.push_back(ci);
        }
        // plan_text_chunk bounds a window at its pinned history plus its own
        // tokens, so the widest is known before a single chunk is encoded.
        for (size_t ci : chunk_ids) {
            text_capacity = std::max(
                text_capacity, params.longform_history_tokens + (int)token_chunks[ci].size());
        }
        text_capacity = std::min(text_capacity, h.n_ctx);
        return true;
    }

    // Encode one chunk and splice its history. Must be called with strictly
    // increasing slots -- the conditioning rolls forward.
    bool prepare_chunk(size_t slot) {
        if (slot < plan.size()) {
            return true;
        }
        const size_t ci = chunk_ids[slot];
        const std::vector<int32_t>& current = token_chunks[ci];
        auto item = std::make_unique<WaveItem>();
        item->chunk_index = ci;
        item->current_tokens = &current;
        const MagpieChunkPlan chunk_plan =
            plan_text_chunk(h, params, seen_tokens, current, absolute, 0, carry_len);
        const int history_len = chunk_plan.history_len;
        const std::vector<int32_t>& window = chunk_plan.text_window;
        item->left_offset = chunk_plan.left_offset;
        item->text_len = chunk_plan.text_len;

        const int64_t enc_start = ggml_time_us();
        if (!encoder.evalDevice(window, params.threads, item->text_cond_device)) {
            return false;
        }
        item->text_cond.resize((size_t)h.n_embd * (size_t)item->text_len);
        magpietts_backend_tensor_get_staged(
            magpie, text_context_staging, item->text_cond_device.tensor, item->text_cond.data(), 0,
            item->text_cond.size() * sizeof(float));
        if (ci > 0) {
            if (!splice_longform_history_context(
                    item->text_cond, item->text_len, (int)current.size(), h.n_embd, carry_cond,
                    carry_len)) {
                return false;
            }
            if (history_len > 0) {
                magpietts_backend_tensor_set_staged(
                    magpie, text_context_staging, item->text_cond_device.tensor,
                    item->text_cond.data(), 0, item->text_cond.size() * sizeof(float));
            }
        }
        carry_cond = item->text_cond;
        carry_len = item->text_len;
        metrics.encoder_ms += (double)(ggml_time_us() - enc_start) / 1000.0;

        // Without this the chunk resumes at relative 0 and crawls through its
        // own history one token a step, re-speaking it.
        if (ci > 0) {
            item->prior.seedLastAttendedAbsolute(
                absolute - 1,
                std::max(h.attention_prior_advance_threshold, h.attention_prior_decay_threshold));
        }
        item->prior.beginChunk(h, item->left_offset, item->text_len, (int)current.size(), ci == 0);
        item->audio_codes.assign(h.audio_codebooks, {});
        for (int c = 0; c < h.audio_codebooks; ++c) {
            item->audio_codes[c].assign((size_t)h.frame_stacking_factor, h.audio_bos_id);
        }
        seen_tokens.insert(seen_tokens.end(), current.begin(), current.end());
        absolute += (int)current.size();
        plan.push_back(std::move(item));
        return true;
    }

    // Everything a step does once a chunk's hidden state exists: sample the
    // codebooks, advance the prior, decide whether the chunk is over, and buffer
    // its frames. Codes arrive round-major from the batched sampler -- round c
    // of a wave of B occupies slots [c*B, c*B+B) -- so item b's codebook c is at
    // c*B+b.
    //
    // Only ever called for a lane whose chunk is still going: a finished one
    // holds its lane until another chunk is admitted, but it no longer grows,
    // because its position stops advancing with it.
    bool step_finish(
        WaveItem& item, std::vector<float>& scores, bool collected,
        const std::vector<int32_t>& all_codes, const std::vector<int32_t>& all_argmax,
        int item_index, int width) {
        const int step = item.step;
        const int frames_remaining = h.max_decoder_steps - step * h.frame_stacking_factor;
        if (frames_remaining <= 0) {
            item.done = true;
            return true;
        }
        const bool forbid_eos = step * h.frame_stacking_factor < h.min_generated_frames;
        const int stacked = h.stacked_audio_codebooks();
        if ((int)all_codes.size() != stacked * width ||
            (int)all_argmax.size() != stacked * width) {
            fprintf(stderr, "CUDA sampler returned an unexpected number of codebooks\n");
            return false;
        }
        std::vector<int32_t> next_codes((size_t)stacked);
        std::vector<int32_t> argmax_codes((size_t)stacked);
        for (int c = 0; c < stacked; ++c) {
            next_codes[(size_t)c] = all_codes[(size_t)(c * width + item_index)];
            argmax_codes[(size_t)c] = all_argmax[(size_t)(c * width + item_index)];
        }

        MagpieStepOutcome outcome;
        if (!advance_chunk_state(
                h, params, label, item.chunk_index, token_chunks.size(), step, item.text_len,
                item.chunk_index + 1 == token_chunks.size(), forbid_eos, frames_remaining,
                next_codes, argmax_codes, collected ? &scores : nullptr, item.prior, item.chunk,
                item.audio_codes, outcome)) {
            return false;
        }
        bool first_frame = false;
        metrics.record_decoder_frame(ggml_time_us(), first_frame);
        decoder_frames_generated += h.frame_stacking_factor;
        for (const std::vector<int32_t>& frame : outcome.frames) {
            item.frames.push_back(frame);
        }
        item.done = outcome.stop;
        return true;
    }

    // Hand the codec whatever this chunk has produced so far. Only ever called
    // on the chunk at the head of this request's drain order, so the codec's
    // single in-order stream stays in order while the chunk is still being
    // decoded -- that is what keeps first audio early.
    bool drain_item(WaveItem& item) {
        if (discard_audio) {
            discarded_frames += (int64_t)item.frames.size();
            item.frames.clear();
            return true;
        }
        for (const std::vector<int32_t>& frame : item.frames) {
            if (!code_writer.write_frame(frame) || !codec_worker.write_frame(frame)) {
                fprintf(stderr, "failed to write streamed codec frame\n");
                codec_worker.join();
                return false;
            }
            ++frames_generated;
        }
        item.frames.clear();
        return true;
    }

    bool flush_item(WaveItem& item, bool last_chunk) {
        if (params.verbose) {
            fprintf(
                stderr, "%s wave flush chunk %zu: %zu frames pending (suppressed %d)\n", label,
                item.chunk_index, item.frames.size(), item.chunk.suppressed_nonfinal_frames);
        }
        if (!drain_item(item)) {
            return false;
        }
        if (discard_audio) {
            return true;
        }
        if (!last_chunk) {
            const int silence_frames = boundary_silence_dist(boundary_silence_rng);
            const std::vector<int32_t> silence = codec_worker.silence_frame();
            for (int i = 0; i < silence_frames; ++i) {
                if (!code_writer.write_frame(silence) || !codec_worker.write_frame(silence)) {
                    fprintf(stderr, "failed to write streamed silence codec frame\n");
                    codec_worker.join();
                    return false;
                }
                ++frames_generated;
            }
        }
        return true;
    }

    // Frames leave in chunk order -- the codec is one stream over a serial
    // convolution state, so a later chunk cannot overtake -- but an earlier one
    // need not wait for its neighbours. Each chunk buffers its own frames until
    // its turn, so this is the reorder buffer a wave that retires out of order
    // needs.
    bool drain_in_order() {
        if (next_drain < plan.size() && !drain_item(*plan[next_drain])) {
            return false;
        }
        while (next_drain < plan.size() && plan[next_drain]->done) {
            if (!flush_item(*plan[next_drain], next_drain + 1 == chunk_ids.size())) {
                return false;
            }
            ++next_drain;
            if (next_drain < plan.size() && !drain_item(*plan[next_drain])) {
                return false;
            }
        }
        return true;
    }

    bool finished() const { return next_drain >= chunk_ids.size(); }

    // What the engine has to ask a session rather than decide for itself. Once a
    // wave carries chunks from several requests these differ lane by lane, so
    // the engine reads them through the lane's owner.
    int speaker() const { return params.speaker; }
    int position_budget() const {
        return (h.max_decoder_steps + h.frame_stacking_factor - 1) / h.frame_stacking_factor;
    }
    magpietts_cuda_sample_item sampling(const WaveItem& item, int frame_index) const {
        magpietts_cuda_sample_item slot;
        slot.seed = (uint64_t)(uint32_t)params.seed;
        slot.cfg_scale = h.cfg_scale;
        slot.temperature = h.temperature;
        slot.top_k = h.top_k;
        slot.frame_index = frame_index;
        // The chunk's opening frames are its own, so the floor that stops it
        // ending before it has said anything is its own too.
        slot.forbid_audio_eos = item.step * h.frame_stacking_factor < h.min_generated_frames;
        return slot;
    }
};

// A lane's occupant. The engine knows a chunk only through this pair, which is
// what lets lanes of one wave belong to unrelated requests.
struct WaveLane {
    WaveSession* session = nullptr;
    WaveItem* item = nullptr;

    bool live() const { return item && !item->done; }
    bool idle() const { return !item || item->done; }
};

// One chunk a session wants admitted.
struct WaveAdmission {
    WaveSession* session = nullptr;
    size_t slot = 0;  // index into that session's plan
    int lane = -1;
};

// The decode engine: one wave of fixed-width lanes, and the machinery to keep
// them full. It owns the decoder runtime, the guidance pair the batched local
// transformer reads, the lane table and the RNG stream position -- everything
// that is one instance regardless of how many requests are in flight.
//
// It knows nothing about requests beyond what a lane's session answers when
// asked: the voice to open a chunk with, the sampling settings to draw it with,
// and what to do with a finished frame. That is the whole of the multi-tenant
// contract.
struct WaveEngine {
    magpietts_model& magpie;
    MagpieStreamingWorkspace& workspace;
    const MagpieDecoder& decoder;
    LocalCodebookSampler* local_sampler;
    const char* label;
    bool verbose = false;
    int threads = 1;

    // Fixed for the life of the engine: the decoder's captured graph, the cross
    // arena it is built around, and the local transformer's composed chain all
    // bake these in.
    int lanes = 0;
    int text_capacity = 0;
    int position_budget = 0;

    std::vector<WaveLane> lane;
    // One [n_embd, lanes] pair carries the whole wave's guidance states.
    // Allocated once: the composed chain bakes in both the width and these
    // addresses, so reallocating would recompose it.
    magpietts_backend_tensor cond;
    magpietts_backend_tensor uncond;
    // The RNG stream position, counted across everything this engine decodes.
    int frame_index = 0;

    int64_t idle_lane_steps = 0;
    int64_t steps = 0;
    int bursts = 0;

    bool open(magpietts_model& model, int lane_count, int capacity, int budget) {
        lanes = lane_count;
        text_capacity = capacity;
        position_budget = budget;
        lane.assign((size_t)lanes, WaveLane{});
        return cond.alloc2d(model, GGML_TYPE_F32, model.hparams.n_embd, lanes, "wave_hidden_cond") &&
               uncond.alloc2d(
                   model, GGML_TYPE_F32, model.hparams.n_embd, lanes, "wave_hidden_uncond");
    }

    void close() {
        decoder.resetWave();
        cond.reset();
        uncond.reset();
        lane.clear();
    }

    int idle_lanes(std::vector<int>* into = nullptr) const {
        int n = 0;
        for (int l = 0; l < lanes; ++l) {
            if (lane[(size_t)l].idle()) {
                ++n;
                if (into) {
                    into->push_back(l);
                }
            }
        }
        return n;
    }

    int live_lanes() const { return lanes - idle_lanes(); }

    // One entry per lane, true where the lane has nothing live in it. This is
    // what the admission policy reads.
    std::vector<char> idle_mask() const {
        std::vector<char> mask((size_t)lanes, 0);
        for (int l = 0; l < lanes; ++l) {
            mask[(size_t)l] = lane[(size_t)l].idle() ? 1 : 0;
        }
        return mask;
    }

    // The ring holds one opening plus the position budget and no more, so a
    // chunk that has spent its budget is finished whether or not it has said so
    // -- the decoder would refuse the next step. The budget is the session's,
    // because a request may lower it with --steps.
    void retire_if_exhausted(WaveLane& slot) {
        if (!slot.live()) {
            return;
        }
        const int budget = slot.session->position_budget();
        if (slot.item->step < budget) {
            return;
        }
        if (verbose) {
            fprintf(
                stderr, "%s wave chunk %zu hit the %d-step budget\n", label,
                slot.item->chunk_index, budget);
        }
        slot.item->done = true;
    }

    // Open the given chunks into the given lanes, building the runtime if there
    // is none. Step 0's guidance pair comes back in the same tensors every later
    // step writes, so the chunks are sampled straight off the prefill.
    bool admit(const std::vector<WaveAdmission>& admissions) {
        if (admissions.empty()) {
            return true;
        }
        const magpietts_hparams& mh = magpie.hparams;
        std::vector<MagpieWavePrefillItem> opening(admissions.size());
        std::vector<std::vector<float>> scores(admissions.size());
        std::vector<char> collect(admissions.size(), 0);
        std::vector<int> into(admissions.size());
        for (size_t j = 0; j < admissions.size(); ++j) {
            WaveSession& owner = *admissions[j].session;
            WaveItem& item = *owner.plan[admissions[j].slot];
            MagpieWavePrefillItem& slot = opening[j];
            into[j] = admissions[j].lane;
            slot.text_cond = &item.text_cond;
            slot.text_cond_device = &item.text_cond_device;
            slot.text_len = item.text_len;
            slot.speaker = owner.speaker();
            slot.audio_codes = &item.audio_codes;
            slot.cross_kv = &item.cross_kv;
            slot.prior = item.prior.priorForStep(mh, item.text_len);
            if (item.prior.shouldCollect(mh, 0, item.text_len)) {
                collect[j] = 1;
                slot.alignment_scores = &scores[j];
            }
        }
        const int64_t admit_start = ggml_time_us();
        if (!decoder.prefillWave(
                opening, into, lanes, threads, position_budget, text_capacity, &cond, &uncond)) {
            fprintf(stderr, "%s wave prefill failed\n", label);
            return false;
        }
        for (size_t j = 0; j < admissions.size(); ++j) {
            WaveSession& owner = *admissions[j].session;
            WaveItem& item = *owner.plan[admissions[j].slot];
            // The encoder output and the chunk's own cross-K/V have done their
            // only job: the text the wave attends over now lives in the
            // runtime's arena.
            item.text_cond_device.reset();
            item.text_cond.clear();
            item.text_cond.shrink_to_fit();
            item.cross_kv.reset();
            lane[(size_t)admissions[j].lane] = WaveLane{&owner, &item};
        }
        ++bursts;
        if (verbose) {
            fprintf(
                stderr, "%s wave admit: %zu chunks into %d lanes, %.2f ms\n", label,
                admissions.size(), lanes, (double)(ggml_time_us() - admit_start) / 1000.0);
        }

        // Step 0, sampled off the prefill's own hidden pair.
        std::vector<int32_t> codes;
        std::vector<int32_t> argmax;
        if (!sample(codes, argmax)) {
            return false;
        }
        for (size_t j = 0; j < admissions.size(); ++j) {
            WaveSession& owner = *admissions[j].session;
            WaveItem& item = *owner.plan[admissions[j].slot];
            if (!owner.step_finish(
                    item, scores[j], collect[j] != 0, codes, argmax, admissions[j].lane, lanes)) {
                return false;
            }
            ++item.step;
        }
        return true;
    }

    // One batched sampler call for the whole wave: the local transformer runs
    // its rounds once, with one slot per lane. Each lane's settings come from
    // its own session.
    bool sample(std::vector<int32_t>& codes, std::vector<int32_t>& argmax) {
        const magpietts_hparams& mh = magpie.hparams;
        std::vector<magpietts_cuda_sample_item> per_item((size_t)lanes);
        for (int l = 0; l < lanes; ++l) {
            const WaveLane& slot = lane[(size_t)l];
            if (slot.item && slot.session) {
                per_item[(size_t)l] = slot.session->sampling(*slot.item, frame_index);
            } else {
                per_item[(size_t)l].frame_index = frame_index;
            }
        }
#if defined(MAGPIETTS_CUDA_SAMPLING)
        if (!local_sampler->sampleCuda(
                cond, uncond, true, mh.cfg_scale, mh.temperature, mh.top_k, false,
                workspace.cudaSampler(), 0, frame_index, codes, argmax, lanes,
                per_item.data())) {
            return false;
        }
#else
        (void)mh;
        fprintf(
            stderr, "wave decoding requires CUDA sampling, which was not compiled into this "
                    "build\n");
        return false;
#endif
        ++frame_index;
        return true;
    }

    // One decode step over every lane, live or not: the graph is a fixed width,
    // so a lane whose chunk has finished is decoded anyway and its result thrown
    // away. What `live` buys is that its position stops advancing, so it cannot
    // run past the ring.
    bool step() {
        const magpietts_hparams& mh = magpie.hparams;
        std::vector<std::vector<float>> scores((size_t)lanes);
        std::vector<char> collect((size_t)lanes, 0);
        std::vector<MagpieWaveDecodeItem> slots((size_t)lanes);
        for (int l = 0; l < lanes; ++l) {
            WaveLane& held = lane[(size_t)l];
            MagpieWaveDecodeItem& slot = slots[(size_t)l];
            if (!held.item) {
                // A lane no chunk has reached yet. It is decoded like any other
                // and its result discarded; what it must not do is advance a
                // position or be read as tokens.
                slot.live = false;
                slot.audio_codes = &idle_codes;
                continue;
            }
            slot.audio_codes = &held.item->audio_codes;
            slot.live = held.live();
            if (!slot.live) {
                continue;
            }
            slot.prior = held.item->prior.priorForStep(mh, held.item->text_len);
            if (held.item->prior.shouldCollect(mh, held.item->step, held.item->text_len)) {
                collect[(size_t)l] = 1;
                slot.alignment_scores = &scores[(size_t)l];
            }
        }
        const ggml_nvtx::range nvtx_step("magpietts_stream_wave_step");
        // One slot more than the step budget: the prefill takes one and each
        // step takes another, so at exactly the budget the last step finds the
        // ring full.
        if (!decoder.evalWave(slots, position_budget, text_capacity, &cond, &uncond)) {
            fprintf(stderr, "%s wave decode step failed\n", label);
            return false;
        }
        std::vector<int32_t> codes;
        std::vector<int32_t> argmax;
        if (!sample(codes, argmax)) {
            return false;
        }
        ++steps;
        for (int l = 0; l < lanes; ++l) {
            WaveLane& held = lane[(size_t)l];
            if (!held.live()) {
                ++idle_lane_steps;
                continue;
            }
            if (!held.session->step_finish(
                    *held.item, scores[(size_t)l], collect[(size_t)l] != 0, codes, argmax, l,
                    lanes)) {
                return false;
            }
            ++held.item->step;
        }
        return true;
    }

    // Tokens for a lane no chunk has reached. Never read for meaning -- it only
    // has to be in range, because the graph decodes every lane either way.
    std::vector<std::vector<int32_t>> idle_codes;
};

static bool
stream_magpie_to_audio(
    magpietts_model& magpie, const nc::NanoCodecModel& codec, MagpieStreamingWorkspace& workspace,
    magpie_stream_params& params, const std::vector<std::vector<int32_t>>& input_token_chunks,
    stream_audio_outputs& outputs, stream_run_metrics& metrics, const char* run_label,
    bool write_codes) {
    const ggml_nvtx::range nvtx_range("magpietts_stream_magpie_to_audio");
    // Request overrides are applied to this copy; the loaded model is shared across requests.
    magpietts_hparams h = magpie.hparams;
    const std::vector<std::vector<int32_t>> token_chunks =
        select_token_chunks(params, input_token_chunks);
    if (token_chunks.empty()) {
        fprintf(stderr, "text token chunk list is empty\n");
        return false;
    }
    const size_t total_text_tokens = token_count(token_chunks);
    const bool longform_active = token_chunks.size() > 1;
    if (params.speaker < 0 || params.speaker >= h.baked_speakers) {
        fprintf(stderr, "speaker must be in [0, %d]\n", h.baked_speakers - 1);
        return false;
    }
    if (params.batch_size > 1 && params.longform_history_tokens < 0) {
        // Refusing beats silently changing long-form audio: at batch > 1 the
        // adaptive history cannot be honoured, because it reads the previous
        // chunk's decode and the wave has not run it yet.
        fprintf(
            stderr,
            "batch-size %d needs a fixed longform-history-tokens: the adaptive history "
            "derives chunk N's text window from chunk N-1's alignment, which a wave decodes "
            "at the same time. Pass --tts.longform-history-tokens N (0 disables history).\n",
            params.batch_size);
        return false;
    }
    if (h.audio_codebooks != codec.numCodebooks()) {
        fprintf(
            stderr, "MagpieTTS emits %d codebooks but NanoCodec expects %d\n", h.audio_codebooks,
            codec.numCodebooks());
        return false;
    }
    if (params.steps > 0) {
        h.max_decoder_steps = params.steps;
    }
    if (params.top_k > 0) {
        h.top_k = params.top_k;
    }
    if (!std::isnan(params.temperature)) {
        h.temperature = params.temperature;
    }
    if (!std::isnan(params.cfg_scale)) {
        h.cfg_scale = params.cfg_scale;
    }
    if (params.seed < 0) {
        params.seed = (int)time(nullptr);
    }

    const int stateful_history_frames = codec.decoderLeftContextFrames();
    const int64_t stateful_history_samples = codec.decoderLeftContextSamples();
    if (params.use_stateful_codec) {
        params.codec_history_frames = 0;
        params.codec_future_frames = 0;
    } else if (params.codec_history_frames < 0) {
        params.codec_history_frames = 1;
    }

    std::mt19937 rng((uint32_t)params.seed);
    std::mt19937 boundary_silence_rng((uint32_t)params.seed ^ 0x9E3779B9u);
    std::uniform_int_distribution<int> boundary_silence_dist(6, 10);
    bool use_cuda_lt = false;
    if (params.use_local_transformer &&
        !magpietts_resolve_lt_backend(magpie, params.lt_backend, use_cuda_lt)) {
        return false;
    }
    bool use_cuda_sampling = false;
    if (!magpietts_resolve_sampling_backend(magpie, params.sampling_backend, use_cuda_sampling)) {
        return false;
    }
    if (params.sampling_backend == MAGPIETTS_BACKEND_AUTO && params.use_local_transformer &&
        !use_cuda_lt) {
        use_cuda_sampling = false;
    }
    if (params.use_local_transformer && !use_cuda_lt && use_cuda_sampling) {
        fprintf(
            stderr,
            "--lt-backend cpu with --sampling-backend cuda is not supported when the local "
            "transformer is enabled\n");
        return false;
    }
    const char* local_transformer_effective =
        params.use_local_transformer ? (use_cuda_lt ? "cuda" : "cpu") : "off";
    const double codec_fps = codec.samplesPerFrame() > 0
                                 ? (double)codec.sampleRate() / (double)codec.samplesPerFrame()
                                 : 0.0;
    const int window_samples = (int)((int64_t)codec.sampleRate() * params.window_ms / 1000);
    const char* label = run_label ? run_label : "stream";
    const char* logit_dump_path = std::getenv("MAGPIETTS_LOGIT_DUMP");
    const char* forced_codes_path = std::getenv("MAGPIETTS_FORCE_CODES");
    std::vector<std::vector<int32_t>> forced_code_frames;
    if (std::strcmp(label, "stream") == 0 &&
        !load_forced_code_frames(forced_codes_path, h.audio_codebooks, forced_code_frames)) {
        return false;
    }
    if (params.verbose) {
        fprintf(
            stderr,
            "%s encoding %zu text tokens in %zu chunk(s) longform=%s with speaker=%d seed=%d; "
            "audio worker chunk=%d history=%d "
            "future=%d "
            "stateful=%s kv-cache=%s lt-backend=%s effective=%s lt-precision=%s "
            "sampling-backend=%s effective=%s managed=%s "
            "rf=%d frames/%lld samples hanning window=%d ms (%d samples, %.2f fps)\n",
            label, total_text_tokens, token_chunks.size(), longform_active ? "on" : "off",
            params.speaker, params.seed, params.chunk_frames, params.codec_history_frames,
            params.codec_future_frames, params.use_stateful_codec ? "yes" : "no",
            params.use_kv_cache ? "yes" : "no",
            magpietts_backend_preference_name(params.lt_backend), local_transformer_effective,
            params.lt_fp32 ? "fp32" : "native",
            magpietts_backend_preference_name(params.sampling_backend),
            use_cuda_sampling ? "cuda" : "cpu", magpie.cuda_unified_memory ? "on" : "off",
            stateful_history_frames, (long long)stateful_history_samples, params.window_ms,
            window_samples, codec_fps);
    }

    metrics.begin();
    outputs.metrics = &metrics;

    if (!workspace.beginRequest(
            params.threads, use_cuda_sampling, h.stacked_audio_codebooks(),
            std::max(1, params.batch_size))) {
        return false;
    }
    LocalCodebookSampler* local_sampler = nullptr;
    if (params.use_local_transformer) {
        local_sampler = workspace.localSampler(use_cuda_lt, params.lt_fp32, params.threads);
        if (!local_sampler) {
            return false;
        }
        if (!workspace.prewarmLocalTransformer(
                *local_sampler, use_cuda_lt, params.lt_fp32, params.use_cfg, params.verbose)) {
            return false;
        }
    }

    std::vector<float>& text_cond = workspace.text_cond;
    magpietts_backend_tensor& text_cond_device = workspace.text_cond_device;
    magpietts_backend_tensor& cond_hidden_device = workspace.cond_hidden_device;
    magpietts_backend_tensor& uncond_hidden_device = workspace.uncond_hidden_device;
    DecoderKvCache& cond_kv = workspace.cond_kv;
    DecoderKvCache& uncond_kv = workspace.uncond_kv;
    DecoderCrossKvCache& cond_cross_kv = workspace.cond_cross_kv;
    MagpieDecoder& decoder = workspace.decoder;
    MagpieEncoder& encoder = workspace.encoder;

    stream_code_writer code_writer;
    if (!code_writer.open(write_codes ? params.codes_out : std::string())) {
        return false;
    }

    codec_stream_worker codec_worker(
        codec, workspace.codec_decoder, workspace.codec_stream_state, workspace.codec_stream_graph,
        params.codec_threads, outputs, codec.samplesPerFrame(), params.chunk_frames,
        params.codec_history_frames, params.codec_future_frames, window_samples,
        (size_t)params.codec_queue_depth, params.use_stateful_codec, &metrics, label,
        params.verbose);
    codec_worker.start();
    if (params.verbose) {
        fprintf(
            stderr, "%s MagpieTTS producer and %s NanoCodec worker are running in parallel\n",
            label, params.use_stateful_codec ? "fast stateful" : "rolling-window");
    }

    auto cancel_worker = [&]() -> bool {
        codec_worker.cancel();
        return false;
    };

    int frames_generated = 0;
    int decoder_frames_generated = 0;
    MagpieLongformAttentionPriorState attention_prior;
    std::vector<int32_t> prior_text_tokens;
    std::vector<float> history_text_context;
    int history_text_context_len = 0;
    MagpiePinnedHostScratch text_context_staging;

    const int64_t t_start = ggml_time_us();
    metrics.decoder.begin(t_start);
    {
        const ggml_nvtx::range nvtx_loop("magpietts_stream_generation_loop");
        int absolute_token_offset = 0;
        // A wave needs the fast CUDA path and a pinned history; anything else
        // falls through to the sequential loop below, unchanged.
        const int max_decoder_positions =
            (h.max_decoder_steps + h.frame_stacking_factor - 1) / h.frame_stacking_factor;
        const int wave_width = std::max(1, params.batch_size);
        // The batched local transformer needs one K/V history per lane, which
        // only the patched attention cache provides; without this the run would
        // form group 0 (width 1) and then fail on group 1.
        const bool wave_attention_ok = magpietts_fused_cached_attention_available(magpie.backend);
        // The batched sampler carries one EOS floor per lane in its config, which
        // is a fixed-size array, so that bounds the wave.
        const bool wave_width_ok = wave_width <= MAGPIETTS_CUDA_MAX_SAMPLE_SLOTS;
        const bool use_wave = (wave_width > 1) && token_chunks.size() > 1 && use_cuda_sampling &&
                              params.use_local_transformer && params.use_cfg &&
                              params.use_kv_cache && params.longform_history_tokens >= 0 &&
                              wave_attention_ok && wave_width_ok && h.dec_kernel == 1;
        // Asking for a wave and silently getting sequential decode is the worst
        // outcome, so say which requirement was not met. A single chunk is not
        // a failure: there is no wave to form.
        if (wave_width > 1 && token_chunks.size() > 1 && !use_wave) {
            const char* why = !use_cuda_sampling              ? "CUDA sampling is not active"
                              : !params.use_local_transformer ? "the local transformer is disabled"
                              : !params.use_cfg               ? "classifier-free guidance is off"
                              : !params.use_kv_cache          ? "the decoder K/V cache is off"
                              : !wave_attention_ok ? "this build lacks the patched cached attention"
                              : !wave_width_ok     ? "the batched sampler tops out at 256 lanes"
                              : h.dec_kernel != 1
                                  ? "this model's decoder feed-forward is a convolution"
                                  : "the long-form history is adaptive";
            fprintf(
                stderr, "%s --tts.batch-size %d ignored: %s. Decoding sequentially.\n", label,
                wave_width, why);
        }
        // ---- Wave scheduler -------------------------------------------------
        // Long-form chunks decoded in lockstep, several at a time through one
        // graph. This needs a pinned history (gated at entry): the adaptive rule
        // derives chunk N's text window from chunk N-1's alignment, and a wave
        // has not decoded chunk N-1 when it plans chunk N.
        // Decode-throughput probe: run N independent sessions through ONE engine,
        // counting frames and dropping audio. It exists to answer the question
        // serving turns on -- does a wave shared between unrelated requests
        // sustain its aggregate rate -- before per-session codec state makes
        // real concurrent output possible. Each session gets the whole input, so
        // N sessions is N identical requests arriving at once.
        if (use_wave && getenv("MAGPIE_MULTI_SESSION")) {
            const int n_sessions = std::max(1, atoi(getenv("MAGPIE_MULTI_SESSION")));
            const int probe_lanes = std::max(1, wave_width);
            std::vector<std::unique_ptr<WaveSession>> sessions;
            std::vector<std::unique_ptr<stream_run_metrics>> session_metrics;
            std::vector<std::unique_ptr<MagpiePinnedHostScratch>> session_staging;
            std::vector<int> session_frames(n_sessions, 0);
            int capacity = 0;
            for (int i = 0; i < n_sessions; ++i) {
                session_metrics.push_back(std::make_unique<stream_run_metrics>());
                session_staging.push_back(std::make_unique<MagpiePinnedHostScratch>());
                sessions.push_back(std::make_unique<WaveSession>(WaveSession{
                    magpie, workspace.encoder, h, params, token_chunks, label,
                    *session_metrics.back(), code_writer, codec_worker, *session_staging.back(),
                    session_frames[(size_t)i], session_frames[(size_t)i], boundary_silence_rng,
                    boundary_silence_dist}));
                sessions.back()->discard_audio = true;
                if (!sessions.back()->plan_chunks()) {
                    return cancel_worker();
                }
                capacity = std::max(capacity, sessions.back()->text_capacity);
            }

            WaveEngine engine{magpie, workspace, decoder, local_sampler, label, params.verbose};
            engine.threads = params.threads;
            engine.idle_codes.assign(h.audio_codebooks, {});
            for (int c = 0; c < h.audio_codebooks; ++c) {
                engine.idle_codes[c].assign((size_t)h.frame_stacking_factor, h.audio_bos_id);
            }
            if (!engine.open(magpie, probe_lanes, capacity, max_decoder_positions + 1)) {
                return cancel_worker();
            }
            const int admit_threshold = std::max(2, probe_lanes / 8);

            // Round robin, so no session can hold every lane.
            size_t turn = 0;
            auto fill = [&](const std::vector<int>& free_lanes) -> bool {
                std::vector<WaveAdmission> admissions;
                for (int l : free_lanes) {
                    bool placed = false;
                    for (size_t tried = 0; tried < sessions.size() && !placed; ++tried) {
                        WaveSession& who = *sessions[(turn + tried) % sessions.size()];
                        if (who.next_chunk >= who.chunk_ids.size()) {
                            continue;
                        }
                        if (!who.prepare_chunk(who.next_chunk)) {
                            return false;
                        }
                        admissions.push_back(WaveAdmission{&who, who.next_chunk, l});
                        ++who.next_chunk;
                        turn = (turn + tried + 1) % sessions.size();
                        placed = true;
                    }
                    if (!placed) {
                        break;
                    }
                }
                if (admissions.empty()) {
                    return true;
                }
                if (!engine.admit(admissions)) {
                    return false;
                }
                for (auto& who : sessions) {
                    if (!who->drain_in_order()) {
                        return false;
                    }
                }
                return true;
            };

            const int64_t probe_start = ggml_time_us();
            std::vector<int> all_lanes(probe_lanes);
            for (int l = 0; l < probe_lanes; ++l) {
                all_lanes[(size_t)l] = l;
            }
            if (!fill(all_lanes)) {
                return cancel_worker();
            }
            for (;;) {
                for (int l = 0; l < probe_lanes; ++l) {
                    engine.retire_if_exhausted(engine.lane[(size_t)l]);
                }
                size_t pending = 0;
                for (auto& who : sessions) {
                    pending += who->chunk_ids.size() - who->next_chunk;
                }
                const std::vector<int> admit =
                    plan_wave_admission(engine.idle_mask(), pending, admit_threshold);
                if (!admit.empty()) {
                    if (!fill(admit)) {
                        return cancel_worker();
                    }
                    continue;
                }
                if (engine.live_lanes() == 0 && pending == 0) {
                    break;
                }
                if (!engine.step()) {
                    return cancel_worker();
                }
                for (auto& who : sessions) {
                    if (!who->drain_in_order()) {
                        return cancel_worker();
                    }
                }
            }
            const double elapsed_s = (double)(ggml_time_us() - probe_start) / 1.0e6;
            int64_t total_frames = 0;
            for (auto& who : sessions) {
                total_frames += who->discarded_frames;
            }
            const double audio_s = (double)total_frames / codec_fps;
            fprintf(
                stderr,
                "%s multi-session probe: %d sessions, %d lanes, %lld frames, audio %.1f s in "
                "%.2f s = %.1fx realtime (occupancy %.1f%%, %d bursts)\n",
                label, n_sessions, probe_lanes, (long long)total_frames, audio_s, elapsed_s,
                audio_s / elapsed_s,
                engine.steps > 0 ? 100.0 * (1.0 - (double)engine.idle_lane_steps /
                                                      ((double)engine.steps * probe_lanes))
                                 : 0.0,
                engine.bursts);
            engine.close();
            codec_worker.cancel();
            return true;
        }

        if (use_wave) {
            WaveSession session{magpie,
                                workspace.encoder,
                                h,
                                params,
                                token_chunks,
                                label,
                                metrics,
                                code_writer,
                                codec_worker,
                                text_context_staging,
                                frames_generated,
                                decoder_frames_generated,
                                boundary_silence_rng,
                                boundary_silence_dist};
            if (!session.plan_chunks()) {
                return cancel_worker();
            }
            const std::vector<size_t>& chunk_ids = session.chunk_ids;
            const int wave_text_capacity = session.text_capacity;
            if (params.verbose) {
                fprintf(
                    stderr, "%s wave scheduler: %zu chunks, width %d, pinned history %d\n", label,
                    chunk_ids.size(), wave_width, params.longform_history_tokens);
            }

            // Lanes are capped below the chunk count, not at it. A wave with a
            // lane per chunk admits everything at once and never refills, which
            // is exactly the single-cohort behaviour continuous batching exists
            // to remove -- it pays max(steps) over the whole run. Measured at 37
            // chunks: 32 lanes gives 122x realtime and 37 lanes gives 86x.
            //
            // Above the floor, half the chunks is the cap: more lanes always win
            // on throughput -- the per-step cost is mostly fixed, so widening
            // amortises it -- until so few chunks are left over that the last
            // arrivals have nothing to hide behind. Below the floor the cap does
            // not apply, because at small chunk counts the parallelism is worth
            // more than the refill: 37 chunks prefer 32 lanes to 18.
            constexpr int kWaveLaneFloor = 32;
            const size_t wave_pending = chunk_ids.size();
            const int wave_lanes =
                wave_pending > 0
                    ? (int)std::min(
                          (size_t)wave_width,
                          std::min(
                              wave_pending,
                              std::max((size_t)kWaveLaneFloor, wave_pending / 2)))
                    : 0;

            WaveEngine engine{magpie,     workspace, decoder, local_sampler,
                              label,      params.verbose};
            engine.threads = params.threads;
            engine.idle_codes.assign(h.audio_codebooks, {});
            for (int c = 0; c < h.audio_codebooks; ++c) {
                engine.idle_codes[c].assign((size_t)h.frame_stacking_factor, h.audio_bos_id);
            }
            if (wave_lanes > 0 &&
                !engine.open(
                    magpie, wave_lanes, wave_text_capacity, max_decoder_positions + 1)) {
                return cancel_worker();
            }

            // Admitting one chunk at a time would be ~118 prefills on a
            // 150-chunk script, and a prefill costs about half a millisecond per
            // lane it computes regardless of how many it opens -- that would eat
            // the whole saving. So idle lanes accumulate and are filled in one
            // burst. Swept over 2377 chunks: an eighth of the lanes wins at both
            // widths measured, and the curve is shallow. Never fewer than two,
            // so a narrow wave does not end up admitting singly.
            constexpr int kWaveAdmitFraction = 8;
            const int admit_threshold = std::max(2, wave_lanes / kWaveAdmitFraction);

            // Fill the given lanes with this session's next chunks.
            auto admit_into = [&](const std::vector<int>& free_lanes) -> bool {
                std::vector<WaveAdmission> admissions;
                for (int l : free_lanes) {
                    if (session.next_chunk >= chunk_ids.size()) {
                        break;
                    }
                    if (!session.prepare_chunk(session.next_chunk)) {
                        return false;
                    }
                    admissions.push_back(WaveAdmission{&session, session.next_chunk, l});
                    ++session.next_chunk;
                }
                if (admissions.empty()) {
                    return true;
                }
                return engine.admit(admissions) && session.drain_in_order();
            };

            if (wave_lanes > 0) {
                std::vector<int> all_lanes(wave_lanes);
                for (int l = 0; l < wave_lanes; ++l) {
                    all_lanes[(size_t)l] = l;
                }
                if (!admit_into(all_lanes)) {
                    return cancel_worker();
                }
                for (;;) {
                    for (int l = 0; l < wave_lanes; ++l) {
                        engine.retire_if_exhausted(engine.lane[(size_t)l]);
                    }
                    const int live = engine.live_lanes();
                    const size_t pending = chunk_ids.size() - session.next_chunk;
                    const std::vector<int> admit =
                        plan_wave_admission(engine.idle_mask(), pending, admit_threshold);
                    if (!admit.empty()) {
                        if (!admit_into(admit)) {
                            return cancel_worker();
                        }
                        continue;
                    }
                    if (live == 0 && pending == 0) {
                        break;
                    }
                    if (codec_worker.is_failed()) {
                        codec_worker.join();
                        return false;
                    }
                    if (!engine.step() || !session.drain_in_order()) {
                        return cancel_worker();
                    }
                }
                if (params.verbose) {
                    fprintf(
                        stderr,
                        "%s wave admission: %d bursts, threshold %d/%d lanes, %lld steps, %lld "
                        "idle lane-steps, occupancy %.1f%%\n",
                        label, engine.bursts, admit_threshold, wave_lanes,
                        (long long)engine.steps, (long long)engine.idle_lane_steps,
                        engine.steps > 0 ? 100.0 * (1.0 - (double)engine.idle_lane_steps /
                                                              ((double)engine.steps * wave_lanes))
                                         : 0.0);
                }
                engine.close();
            }

            // Anything still holding frames -- the tail of the drain order.
            while (session.next_drain < session.plan.size()) {
                if (!session.flush_item(
                        *session.plan[session.next_drain],
                        session.next_drain + 1 == chunk_ids.size())) {
                    return false;
                }
                ++session.next_drain;
            }
        }

        for (size_t chunk_index = 0; !use_wave && chunk_index < token_chunks.size();
             ++chunk_index) {
            const std::vector<int32_t>& current_tokens = token_chunks[chunk_index];
            if (current_tokens.empty()) {
                continue;
            }
            if ((int)current_tokens.size() > h.n_ctx) {
                fprintf(
                    stderr, "%s text chunk %zu has %zu tokens, exceeding model context %d\n", label,
                    chunk_index, current_tokens.size(), h.n_ctx);
                return cancel_worker();
            }

            // The adaptive path asks where the previous chunk's decode ended up
            // attending, which makes chunk N's window depend on chunk N-1's
            // output. Fine sequentially, fatal to batching, which is why a wave
            // requires a pinned history and passes required_history = 0.
            int required_history = 0;
            if (params.longform_history_tokens < 0 && chunk_index > 0 &&
                attention_prior.initialized()) {
                const int last_abs = attention_prior.lastAttendedAbsolute();
                if (last_abs >= 0 && last_abs < absolute_token_offset) {
                    required_history = absolute_token_offset - last_abs;
                }
            }
            const MagpieChunkPlan chunk_plan = plan_text_chunk(
                h, params, prior_text_tokens, current_tokens, absolute_token_offset,
                required_history, history_text_context_len);
            const int history_len = chunk_plan.history_len;
            const int left_offset = chunk_plan.left_offset;
            const std::vector<int32_t>& text_window = chunk_plan.text_window;
            const int text_len = chunk_plan.text_len;
            const bool first_text_chunk = chunk_index == 0;
            const bool final_text_chunk = chunk_index + 1 == token_chunks.size();

            if (params.verbose && longform_active) {
                fprintf(
                    stderr,
                    "%s longform text chunk %zu/%zu: current=%zu history=%d window=%d "
                    "left_offset=%d absolute_start=%d\n",
                    label, chunk_index + 1, token_chunks.size(), current_tokens.size(), history_len,
                    text_len, left_offset, absolute_token_offset);
            }

            text_cond.clear();
            cond_kv.clear();
            uncond_kv.clear();
            cond_cross_kv.clear();
            std::vector<std::vector<int32_t>> audio_codes(h.audio_codebooks);
            for (int c = 0; c < h.audio_codebooks; ++c) {
                audio_codes[c].assign((size_t)h.frame_stacking_factor, h.audio_bos_id);
            }
            attention_prior.beginChunk(
                h, left_offset, text_len, (int)current_tokens.size(), first_text_chunk);

            const int64_t encoder_start_us = ggml_time_us();
            if (use_cuda_sampling) {
                if (!encoder.evalDevice(text_window, params.threads, text_cond_device)) {
                    return cancel_worker();
                }
                if (longform_active) {
                    text_cond.resize((size_t)h.n_embd * (size_t)text_len);
                    magpietts_backend_tensor_get_staged(
                        magpie, text_context_staging, text_cond_device.tensor, text_cond.data(), 0,
                        text_cond.size() * sizeof(float));
                }
                if (params.use_local_transformer) {
                    if (!cond_hidden_device.alloc2d(
                            magpie, GGML_TYPE_F32, h.n_embd, 1, "decoder_hidden_cond_device")) {
                        return cancel_worker();
                    }
                    if (params.use_cfg &&
                        !uncond_hidden_device.alloc2d(
                            magpie, GGML_TYPE_F32, h.n_embd, 1, "decoder_hidden_uncond_device")) {
                        return cancel_worker();
                    }
                }
            } else if (!encoder.eval(text_window, params.threads, text_cond)) {
                return cancel_worker();
            }
            if (longform_active) {
                if (!first_text_chunk) {
                    if (!splice_longform_history_context(
                            text_cond, text_len, (int)current_tokens.size(), h.n_embd,
                            history_text_context, history_text_context_len)) {
                        return cancel_worker();
                    }
                    if (use_cuda_sampling && history_len > 0) {
                        magpietts_backend_tensor_set_staged(
                            magpie, text_context_staging, text_cond_device.tensor, text_cond.data(),
                            0, text_cond.size() * sizeof(float));
                    }
                }
                history_text_context = text_cond;
                history_text_context_len = text_len;
            }
            metrics.encoder_ms += (double)(ggml_time_us() - encoder_start_us) / 1000.0;

            MagpieChunkDecodeState chunk_state;
            const int max_decoder_positions =
                (h.max_decoder_steps + h.frame_stacking_factor - 1) / h.frame_stacking_factor;
            for (int step = 0; step < max_decoder_positions; ++step) {
                const ggml_nvtx::range nvtx_step("magpietts_stream_generation_step");
                const int frames_remaining = h.max_decoder_steps - step * h.frame_stacking_factor;
                if (frames_remaining <= 0) {
                    break;
                }
                if (codec_worker.is_failed()) {
                    codec_worker.join();
                    return false;
                }
                if (params.verbose && step % 10 == 0) {
                    fprintf(
                        stderr, "%s generating codec frame %d/%d for text chunk %zu/%zu\n", label,
                        step, max_decoder_positions, chunk_index + 1, token_chunks.size());
                }

                decoder_result cond;
                decoder_result uncond;
                cond.logits_required = !params.use_local_transformer;
                uncond.logits_required = !params.use_local_transformer;
                const bool forbid_eos = step * h.frame_stacking_factor < h.min_generated_frames;
                std::vector<int32_t> next_codes;
                std::vector<int32_t> argmax_codes;
                std::vector<float> alignment_scores;
                magpietts_decoder_attention decoder_attention;
                decoder_attention.prior = attention_prior.priorForStep(h, text_len);
                if (attention_prior.shouldCollect(h, step, text_len)) {
                    decoder_attention.alignment_scores = &alignment_scores;
                }
                const magpietts_decoder_attention* decoder_attention_arg =
                    (decoder_attention.prior || decoder_attention.alignment_scores)
                        ? &decoder_attention
                        : nullptr;
                const int sample_frame_index = decoder_frames_generated;

                if (use_cuda_sampling) {
                    if (params.use_local_transformer) {
                        const bool decode_ok =
                            params.use_cfg
                                ? (params.use_kv_cache
                                       ? decoder.evalCachedPair(
                                             text_cond, text_len, audio_codes, params.speaker,
                                             params.threads, cond_kv, uncond_kv, cond, uncond,
                                             max_decoder_positions, nullptr, &text_cond_device,
                                             &cond_hidden_device, &uncond_hidden_device,
                                             &cond_cross_kv, decoder_attention_arg)
                                       : decoder.evalPair(
                                             text_cond, text_len, audio_codes, params.speaker,
                                             params.threads, cond, uncond, nullptr,
                                             &text_cond_device, &cond_hidden_device,
                                             &uncond_hidden_device, decoder_attention_arg))
                                : (params.use_kv_cache
                                       ? decoder.evalCached(
                                             text_cond, text_len, audio_codes, params.speaker, true,
                                             params.threads, cond_kv, cond, nullptr,
                                             &text_cond_device, &cond_hidden_device, &cond_cross_kv,
                                             decoder_attention_arg)
                                       : decoder.eval(
                                             text_cond, text_len, audio_codes, params.speaker, true,
                                             params.threads, cond, nullptr, &text_cond_device,
                                             &cond_hidden_device, decoder_attention_arg));
                        if (!decode_ok) {
                            return cancel_worker();
                        }
#if defined(MAGPIETTS_CUDA_SAMPLING)
                        if (!local_sampler->sampleCuda(
                                cond_hidden_device, uncond_hidden_device, params.use_cfg,
                                h.cfg_scale, h.temperature, h.top_k, forbid_eos,
                                workspace.cudaSampler(), (uint64_t)(uint32_t)params.seed,
                                sample_frame_index, next_codes, argmax_codes)) {
                            return cancel_worker();
                        }
#else
                        fprintf(
                            stderr, "CUDA sampling was not compiled into this MagpieTTS build\n");
                        return cancel_worker();
#endif
                    } else {
                        magpietts_cuda_sample_request cuda_sample;
#if defined(MAGPIETTS_CUDA_SAMPLING)
                        cuda_sample.sampler = workspace.cudaSampler();
#endif
                        cuda_sample.use_cfg = params.use_cfg;
                        cuda_sample.cfg_scale = h.cfg_scale;
                        cuda_sample.temperature = h.temperature;
                        cuda_sample.top_k = h.top_k;
                        cuda_sample.forbid_audio_eos = forbid_eos;
                        cuda_sample.seed = (uint64_t)(uint32_t)params.seed;
                        cuda_sample.frame_index = sample_frame_index;

                        const bool decode_ok =
                            params.use_cfg
                                ? (params.use_kv_cache
                                       ? decoder.evalCachedPair(
                                             text_cond, text_len, audio_codes, params.speaker,
                                             params.threads, cond_kv, uncond_kv, cond, uncond,
                                             max_decoder_positions, &cuda_sample, &text_cond_device,
                                             nullptr, nullptr, &cond_cross_kv,
                                             decoder_attention_arg)
                                       : decoder.evalPair(
                                             text_cond, text_len, audio_codes, params.speaker,
                                             params.threads, cond, uncond, &cuda_sample,
                                             &text_cond_device, nullptr, nullptr,
                                             decoder_attention_arg))
                                : (params.use_kv_cache
                                       ? decoder.evalCached(
                                             text_cond, text_len, audio_codes, params.speaker, true,
                                             params.threads, cond_kv, cond, &cuda_sample,
                                             &text_cond_device, nullptr, &cond_cross_kv,
                                             decoder_attention_arg)
                                       : decoder.eval(
                                             text_cond, text_len, audio_codes, params.speaker, true,
                                             params.threads, cond, &cuda_sample, &text_cond_device,
                                             nullptr, decoder_attention_arg));
                        if (!decode_ok) {
                            return cancel_worker();
                        }
                        next_codes = std::move(cuda_sample.codes);
                        argmax_codes = std::move(cuda_sample.argmax_codes);
                    }
                    if ((int)next_codes.size() != h.stacked_audio_codebooks() ||
                        (int)argmax_codes.size() != h.stacked_audio_codebooks()) {
                        fprintf(
                            stderr, "CUDA sampler returned an unexpected number of codebooks\n");
                        return cancel_worker();
                    }
                } else {
                    if (params.use_cfg) {
                        const bool pair_ok =
                            params.use_kv_cache
                                ? decoder.evalCachedPair(
                                      text_cond, text_len, audio_codes, params.speaker,
                                      params.threads, cond_kv, uncond_kv, cond, uncond,
                                      max_decoder_positions, nullptr, nullptr, nullptr, nullptr,
                                      &cond_cross_kv, decoder_attention_arg)
                                : decoder.evalPair(
                                      text_cond, text_len, audio_codes, params.speaker,
                                      params.threads, cond, uncond, nullptr, nullptr, nullptr,
                                      nullptr, decoder_attention_arg);
                        if (!pair_ok) {
                            return cancel_worker();
                        }
                    } else {
                        const bool cond_ok =
                            params.use_kv_cache
                                ? decoder.evalCached(
                                      text_cond, text_len, audio_codes, params.speaker, true,
                                      params.threads, cond_kv, cond, nullptr, nullptr, nullptr,
                                      &cond_cross_kv, decoder_attention_arg)
                                : decoder.eval(
                                      text_cond, text_len, audio_codes, params.speaker, true,
                                      params.threads, cond, nullptr, nullptr, nullptr,
                                      decoder_attention_arg);
                        if (!cond_ok) {
                            return cancel_worker();
                        }
                    }

                    if (params.use_local_transformer) {
                        LocalCodebookLogitDump logit_dump;
                        const bool dump_logits = logit_dump_path && logit_dump_path[0] &&
                                                 std::strcmp(label, "stream") == 0 &&
                                                 chunk_index == 0;
                        if (dump_logits) {
                            logit_dump.path = logit_dump_path;
                            logit_dump.label = label;
                            logit_dump.chunk_index = (int)chunk_index;
                            logit_dump.step = step;
                            logit_dump.frame_index = sample_frame_index;
                        }
                        std::vector<int32_t> stacked_forced_codes;
                        const std::vector<int32_t>* forced_codes = nullptr;
                        const size_t first_forced_frame =
                            static_cast<size_t>(step) *
                            static_cast<size_t>(h.frame_stacking_factor);
                        if (chunk_index == 0 && first_forced_frame < forced_code_frames.size()) {
                            if (!magpietts_stack_forced_code_frames(
                                    forced_code_frames, first_forced_frame, h,
                                    stacked_forced_codes)) {
                                return cancel_worker();
                            }
                            forced_codes = &stacked_forced_codes;
                        }
                        if (!local_sampler->sample(
                                cond.hidden_last, uncond.hidden_last, params.use_cfg, h.cfg_scale,
                                h.temperature, h.top_k, forbid_eos, rng, next_codes, argmax_codes,
                                dump_logits ? &logit_dump : nullptr, forced_codes)) {
                            return cancel_worker();
                        }
                    } else {
                        next_codes = MagpieCodebookSampler::sampleParallel(
                            cond.logits_last, uncond.logits_last, h, params.use_cfg, h.cfg_scale,
                            h.temperature, h.top_k, forbid_eos, rng, &argmax_codes);
                    }
                }

                MagpieStepOutcome outcome;
                if (!advance_chunk_state(
                        h, params, label, chunk_index, token_chunks.size(), step, text_len,
                        final_text_chunk, forbid_eos, frames_remaining, next_codes, argmax_codes,
                        decoder_attention.alignment_scores ? &alignment_scores : nullptr,
                        attention_prior, chunk_state, audio_codes, outcome)) {
                    return cancel_worker();
                }

                bool first_frame = false;
                metrics.record_decoder_frame(ggml_time_us(), first_frame);
                decoder_frames_generated += h.frame_stacking_factor;
                for (const std::vector<int32_t>& frame : outcome.frames) {
                    if (!code_writer.write_frame(frame) || !codec_worker.write_frame(frame)) {
                        fprintf(stderr, "failed to write streamed codec frame\n");
                        codec_worker.join();
                        return false;
                    }
                    ++frames_generated;
                }
                if (outcome.stop) {
                    break;
                }
            }

            if (longform_active && !final_text_chunk) {
                const int boundary_silence_frames = boundary_silence_dist(boundary_silence_rng);
                const std::vector<int32_t> silence = codec_worker.silence_frame();
                for (int i = 0; i < boundary_silence_frames; ++i) {
                    if (!code_writer.write_frame(silence)) {
                        fprintf(stderr, "failed to write streamed silence codec frame\n");
                        return cancel_worker();
                    }
                    ++frames_generated;
                    if (!codec_worker.write_frame(silence)) {
                        codec_worker.join();
                        return false;
                    }
                }
                if (params.verbose) {
                    fprintf(
                        stderr, "%s inserted %d silence codec frames after text chunk %zu/%zu\n",
                        label, boundary_silence_frames, chunk_index + 1, token_chunks.size());
                }
            }

            prior_text_tokens.insert(
                prior_text_tokens.end(), current_tokens.begin(), current_tokens.end());
            absolute_token_offset += (int)current_tokens.size();
        }
    }
    metrics.decoder.finish(
        metrics.decoder.last_event_us > 0 ? metrics.decoder.last_event_us : ggml_time_us());

    if (frames_generated == 0) {
        codec_worker.cancel();
        fprintf(stderr, "no codec frames generated\n");
        return false;
    }

    if (params.flush_partial_chunk) {
        codec_worker.finish_tokens();
    } else {
        codec_worker.close_input();
    }
    if (!codec_worker.join()) {
        return false;
    }

    metrics.generated_frames = frames_generated;
    metrics.finish(outputs.samples_written, codec.sampleRate(), codec_fps);
    const double generation_elapsed_s = (ggml_time_us() - t_start) / 1000000.0;
    const bool warmup = std::strcmp(label, "warmup") == 0;
    if (params.verbose || (params.benchmark && !warmup)) {
        const char* summary_prefix = warmup ? "warmup " : "";
        fprintf(
            stderr,
            "%sgenerated %d codec frames and streamed %zu samples in %.2f s; "
            "encoder_ms=%.2f "
            "decoder_ttft_ms=%.2f decoder_itl_avg_ms=%.2f decoder_itl_min_ms=%.2f "
            "decoder_itl_max_ms=%.2f decoder_itl_p95_ms=%.2f decoder_itl_p99_ms=%.2f "
            "decoder_rtfx=%.2f "
            "codec_ttfa_ms=%.2f codec_icl_avg_ms=%.2f codec_icl_min_ms=%.2f "
            "codec_icl_max_ms=%.2f codec_icl_p95_ms=%.2f codec_icl_p99_ms=%.2f "
            "codec_rtfx=%.2f "
            "e2e_audio_s=%.3f e2e_elapsed_s=%.2f e2e_rtf=%.4f e2e_rtfx=%.2f "
            "e2e_ttfa_ms=%.2f e2e_icl_avg_ms=%.2f e2e_icl_min_ms=%.2f "
            "e2e_icl_max_ms=%.2f e2e_icl_p95_ms=%.2f e2e_icl_p99_ms=%.2f "
            "codec_chunks=%d e2e_chunks=%d\n",
            summary_prefix, frames_generated, outputs.samples_written, generation_elapsed_s,
            metrics.encoder_ms, metrics.decoder.first_event_ms,
            metrics.decoder.inter_event_avg_ms(), metrics.decoder.inter_event_min_value_ms(),
            metrics.decoder.inter_event_max_ms, metrics.decoder.inter_event_p95_ms(),
            metrics.decoder.inter_event_p99_ms(), metrics.decoder_rtfx,
            metrics.codec.first_event_ms, metrics.codec.inter_event_avg_ms(),
            metrics.codec.inter_event_min_value_ms(), metrics.codec.inter_event_max_ms,
            metrics.codec.inter_event_p95_ms(), metrics.codec.inter_event_p99_ms(),
            metrics.codec_rtfx, metrics.audio_s, metrics.e2e_elapsed_s, metrics.e2e_rtf,
            metrics.e2e_rtfx, metrics.e2e.first_event_ms, metrics.e2e.inter_event_avg_ms(),
            metrics.e2e.inter_event_min_value_ms(), metrics.e2e.inter_event_max_ms,
            metrics.e2e.inter_event_p95_ms(), metrics.e2e.inter_event_p99_ms(), metrics.chunks,
            metrics.e2e_chunks);
    }
    return true;
}

bool
MagpieStreamingRuntime::synthesize(
    magpie_stream_params& params, const std::vector<int32_t>& tokens,
    const magpie_pcm_callback& pcm_callback, stream_run_metrics& metrics) {
    return synthesize(params, tokens, pcm_callback, metrics, "riva_tts", false);
}

bool
MagpieStreamingRuntime::synthesize(
    magpie_stream_params& params, const std::vector<int32_t>& tokens,
    const magpie_pcm_callback& pcm_callback, stream_run_metrics& metrics, const char* run_label,
    bool write_codes) {
    return synthesize(
        params, std::vector<std::vector<int32_t>>{tokens}, pcm_callback, metrics, run_label,
        write_codes);
}

bool
MagpieStreamingRuntime::synthesize(
    magpie_stream_params& params, const std::vector<std::vector<int32_t>>& token_chunks,
    const magpie_pcm_callback& pcm_callback, stream_run_metrics& metrics, const char* run_label,
    bool write_codes) {
    if (!impl_) {
        return false;
    }
    stream_audio_outputs outputs;
    outputs.sample_rate = impl_->codec.sampleRate();
    outputs.pcm_callback = pcm_callback;
    if (!impl_->workspace) {
        impl_->workspace = std::make_unique<MagpieStreamingWorkspace>(impl_->magpie, impl_->codec);
    }
    return stream_magpie_to_audio(
        impl_->magpie, impl_->codec, *impl_->workspace, params, token_chunks, outputs, metrics,
        run_label, write_codes);
}

bool
magpie_stream_runtime_init(
    magpie_stream_runtime*& runtime, const std::string& magpie_model,
    const std::string& codec_model, magpietts_uma_mode uma_mode, bool magpie_cpu, bool codec_cpu) {
    runtime = new magpie_stream_runtime();
    if (!runtime->load(magpie_model, codec_model, uma_mode, magpie_cpu, codec_cpu)) {
        magpie_stream_runtime_free(runtime);
        runtime = nullptr;
        return false;
    }
    return true;
}

void
magpie_stream_runtime_free(magpie_stream_runtime* runtime) {
    if (!runtime) {
        return;
    }
    delete runtime;
}

int
magpie_stream_runtime_sample_rate(const magpie_stream_runtime* runtime) {
    return runtime ? runtime->sampleRate() : 0;
}

int
magpie_stream_runtime_speaker_count(const magpie_stream_runtime* runtime) {
    return runtime ? runtime->speakerCount() : 0;
}

std::vector<std::string>
magpie_stream_runtime_speaker_names(const magpie_stream_runtime* runtime) {
    return runtime ? runtime->speakerNames() : std::vector<std::string>();
}

bool
magpie_stream_runtime_synthesize(
    magpie_stream_runtime* runtime, magpie_stream_params& params,
    const std::vector<int32_t>& tokens, const magpie_pcm_callback& pcm_callback,
    stream_run_metrics& metrics) {
    if (!runtime) {
        return false;
    }
    return runtime->synthesize(params, tokens, pcm_callback, metrics);
}

const char*
magpie_longform_mode_name(magpie_longform_mode mode) {
    switch (mode) {
        case MAGPIE_LONGFORM_AUTO:
            return "auto";
        case MAGPIE_LONGFORM_OFF:
            return "off";
        case MAGPIE_LONGFORM_ON:
            return "on";
    }
    return "unknown";
}

bool
parse_magpie_longform_mode(const std::string& value, magpie_longform_mode& mode) {
    if (value == "auto") {
        mode = MAGPIE_LONGFORM_AUTO;
        return true;
    }
    if (value == "off") {
        mode = MAGPIE_LONGFORM_OFF;
        return true;
    }
    if (value == "on") {
        mode = MAGPIE_LONGFORM_ON;
        return true;
    }
    return false;
}

}  // namespace nemo_speech::tts
