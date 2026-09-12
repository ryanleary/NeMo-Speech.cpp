// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>

struct magpietts_cuda_sampler;

// Widest batch the per-item EOS floor can describe; also the widest wave the
// batched sampler will take.
#define MAGPIETTS_CUDA_MAX_SAMPLE_SLOTS 256

magpietts_cuda_sampler* magpietts_cuda_sampler_create(int codebooks);
void magpietts_cuda_sampler_free(magpietts_cuda_sampler* sampler);
bool magpietts_cuda_device_is_uma(void);

// Bind to a caller-owned stream.
bool magpietts_cuda_sampler_bind_stream(
    magpietts_cuda_sampler* sampler, void* stream, char* error, size_t error_size);

// Configure per-frame values stored in a stable device buffer.
bool magpietts_cuda_sampler_configure(
    magpietts_cuda_sampler* sampler, bool use_cfg, float cfg_scale, float temperature, int top_k,
    bool forbid_audio_eos, uint64_t seed, int frame_index, char* error, size_t error_size);
// Everything one item of a batched round decides for itself. A round is one slot
// per item, so slot i is item i; once a wave carries chunks from more than one
// request, these are the fields that differ between them.
struct magpietts_cuda_sample_item {
    uint64_t seed = 0;
    float cfg_scale = 1.0f;
    float temperature = 0.0f;
    int top_k = 1;
    // The item's own position in its own RNG stream. A counter shared across the
    // wave would make a request's output depend on when its neighbours were
    // admitted.
    int frame_index = 0;
    // False once the chunk is past its opening frames. Under continuous batching
    // a freshly admitted chunk is inside them while its neighbours are hundreds
    // of steps in.
    bool forbid_audio_eos = false;
};

// Give each slot its own settings. Call after configure, which sets every slot
// to the scalar values.
bool magpietts_cuda_sampler_configure_items(
    magpietts_cuda_sampler* sampler, const magpietts_cuda_sample_item* items, int count,
    char* error, size_t error_size);
bool magpietts_cuda_sampler_upload_config(
    magpietts_cuda_sampler* sampler, char* error, size_t error_size);

// Compose and launch the local-transformer sequence as a CUDA graph.
bool magpietts_cuda_sampler_sequence_is_warm(const magpietts_cuda_sampler* sampler);
bool magpietts_cuda_sampler_sequence_is_ready(const magpietts_cuda_sampler* sampler);
bool magpietts_cuda_sampler_sequence_is_disabled(const magpietts_cuda_sampler* sampler);
bool magpietts_cuda_sampler_sequence_build_active(const magpietts_cuda_sampler* sampler);
void magpietts_cuda_sampler_sequence_mark_warm(magpietts_cuda_sampler* sampler);
bool magpietts_cuda_sampler_sequence_begin_build(
    magpietts_cuda_sampler* sampler, char* error, size_t error_size);
bool magpietts_cuda_sampler_sequence_finish_build_and_launch(
    magpietts_cuda_sampler* sampler, char* error, size_t error_size);
void magpietts_cuda_sampler_sequence_abort_build(magpietts_cuda_sampler* sampler);
void magpietts_cuda_sampler_sequence_disable(magpietts_cuda_sampler* sampler);
// Throw away a composed chain so the next call rebuilds it. Required whenever
// the batch width changes: the graph bakes in the shapes it was built from.
void magpietts_cuda_sampler_sequence_invalidate(magpietts_cuda_sampler* sampler);
bool magpietts_cuda_sampler_sequence_launch(
    magpietts_cuda_sampler* sampler, char* error, size_t error_size);
bool magpietts_cuda_sampler_sequence_add_ggml_graph(
    magpietts_cuda_sampler* sampler, void* graph_template, char* error, size_t error_size);
bool magpietts_cuda_sampler_sequence_add_device_copy(
    magpietts_cuda_sampler* sampler, const void* src_device, void* dst_device, size_t bytes,
    char* error, size_t error_size);

bool magpietts_cuda_sample_codebooks(
    magpietts_cuda_sampler* sampler, const float* logits_cond, const float* logits_uncond,
    int codebooks, int vocab_size, int audio_codebook_size, int audio_eos_id, bool use_cfg,
    float cfg_scale, float temperature, int top_k, bool forbid_audio_eos, uint64_t seed,
    int frame_index, int codebook_offset, int32_t* codes_out, int32_t* argmax_out, char* error,
    size_t error_size);

bool magpietts_cuda_sample_codebooks_device(
    magpietts_cuda_sampler* sampler, const float* logits_cond, const float* logits_uncond,
    int codebooks, int vocab_size, int audio_codebook_size, int audio_eos_id, bool use_cfg,
    float cfg_scale, float temperature, int top_k, bool forbid_audio_eos, uint64_t seed,
    int frame_index, int codebook_offset, int output_offset, char* error, size_t error_size);

// Launch using the most recently uploaded configuration.
bool magpietts_cuda_sample_codebooks_device_configured(
    magpietts_cuda_sampler* sampler, const float* logits_cond, const float* logits_uncond,
    int codebooks, int vocab_size, int audio_codebook_size, int audio_eos_id, int codebook_offset,
    int output_offset, char* error, size_t error_size);

// Hand `count` freshly sampled codes back to the device tensor the next round
// consumes. A batch's codes for one round are contiguous, so this is one copy
// rather than one per item.
bool magpietts_cuda_copy_sampled_code_to_device(
    magpietts_cuda_sampler* sampler, int first_codebook, int count, void* dst_device, char* error,
    size_t error_size);

bool magpietts_cuda_copy_sampled_codebooks(
    magpietts_cuda_sampler* sampler, int codebooks, int32_t* codes_out, int32_t* argmax_out,
    char* error, size_t error_size);
