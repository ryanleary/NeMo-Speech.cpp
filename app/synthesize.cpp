// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "commands.h"
#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#endif

#include "audio_file.h"
#include "cli_util.h"
#include "config.h"
#include "engine_registry.h"
#include "json.h"
#include "model_utils.h"
#include "parameter_parser.h"

namespace {

void
write_audio(const std::filesystem::path& path, const std::string& audio, bool force) {
    if (path == "-") {
#if defined(_WIN32)
        _setmode(_fileno(stdout), _O_BINARY);
#endif
        if (std::fwrite(audio.data(), 1, audio.size(), stdout) != audio.size())
            throw std::runtime_error("could not write synthesized audio to stdout");
        return;
    }
    if (!force && std::filesystem::exists(path))
        throw std::runtime_error(path.string() + " already exists (use --force to replace it)");
    if (path.has_parent_path())
        std::filesystem::create_directories(path.parent_path());
    const auto temporary = path.string() + ".tmp";
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    if (!output)
        throw std::runtime_error("cannot write " + path.string());
    output.write(audio.data(), static_cast<std::streamsize>(audio.size()));
    output.close();
    if (!output)
        throw std::runtime_error("cannot finish writing " + path.string());
    std::error_code error;
    if (force)
        std::filesystem::remove(path, error);
    std::filesystem::rename(temporary, path, error);
    if (error) {
        std::filesystem::remove(temporary);
        throw std::runtime_error("cannot finish writing " + path.string() + ": " + error.message());
    }
}

}  // namespace

void
print_synthesize_help(const char* program) {
    std::printf(
        "Usage: %s synthesize TEXT [options]\n\n"
        "Options:\n"
        "  --magpie-model MODEL      MagpieTTS GGUF path or indexed HF repo\n"
        "                            (default: nvidia/magpie_tts_multilingual_357m)\n"
        "  --codec-model MODEL       NanoCodec GGUF path or indexed HF repo\n"
        "                            (default: nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps)\n"
        "  --tokenizer-dir MODEL     Tokenizer directory or indexed HF repo\n"
        "                            (default: MagpieTTS repository)\n"
        "  --tn-model-dir DIR        Optional text-normalization grammars\n"
        "  -i, --input PATH          Read text from a UTF-8 file\n"
        "  -o, --output PATH         Output path (default: speech.wav; '-' = stdout)\n"
        "  --format wav|pcm          WAV container or raw signed PCM16\n"
        "  --language CODE           Text language (default: en-US)\n"
        "  --voice NAME              Voice name or speaker index\n"
        "  --speaker N               Speaker index\n"
        "  --sample-rate HZ          Output rate (8 kHz through model rate)\n"
        "  --device, --backend DEVICE\n"
        "                            auto, cpu, cuda[:N], metal, or vulkan[:N]\n"
        "  --seed N --steps N --top-k N --temperature N --cfg-scale N\n"
        "  --config FILE             Load the complete TTS YAML config tree\n"
        "  --tts.KEY VALUE           Override any C++ TTS setting\n"
        "  --concurrency N           Fire N identical requests at once and report the\n"
        "                            aggregate realtime factor and first-audio spread\n"
        "  --arrival-ms N            Stagger those requests N ms apart instead\n"
        "  --cancel-after-ms N       Stop reading audio after N ms, as a client hanging up\n"
        "  --rounds N                Repeat the concurrent burst N times in one process\n"
        "  --no-warmup               Skip warmup\n"
        "  --force                   Replace an existing WAV\n",
        program);
}

int
command_synthesize(int argc, char** argv) {
    try {
        if (argc > 0 && is_help_argument(argv[0])) {
            print_synthesize_help("nemo-speech");
            return 0;
        }
        nemo_speech::tts::MagpieTtsServerConfig parsed;
        nemo_speech::common::ParameterParser parser;
        parser.Register("tts", parsed);
        std::string config_file;
        for (int i = 0; i < argc; ++i) {
            if (std::string(argv[i]) == "--config") {
                if (++i >= argc)
                    throw std::invalid_argument("--config requires a value");
                config_file = argv[i];
            }
        }
        if (!config_file.empty())
            parser.ApplyYaml(config_file);
        parser.ApplyEnv("NEMO_SPEECH");

        std::string text, input_path, output_path = "speech.wav", language, voice;
        std::string format = "wav";
        int concurrency = 1;
        int arrival_ms = 0;
        int cancel_after_ms = 0;
        int rounds = 1;
        bool force = false;
        bool warmup = true;
        int output_rate = 0;
        bool device_set = false;
        std::string device_name = "auto";
        int gpu = default_gpu_index();
        nemo_speech::tts::MagpieSynthesisOptions request_options;
        auto value = [&](int& i, const std::string& option) {
            if (++i >= argc)
                throw std::invalid_argument(option + " requires a value");
            return std::string(argv[i]);
        };
        for (int i = 0; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--config")
                ++i;
            else if (arg == "--concurrency")
                concurrency = std::stoi(value(i, arg));
            else if (arg == "--arrival-ms")
                arrival_ms = std::stoi(value(i, arg));
            else if (arg == "--cancel-after-ms")
                cancel_after_ms = std::stoi(value(i, arg));
            else if (arg == "--rounds")
                rounds = std::stoi(value(i, arg));
            else if (arg == "--magpie-model")
                parsed.runtime.magpie_model = value(i, arg);
            else if (arg == "--codec-model")
                parsed.runtime.codec_model = value(i, arg);
            else if (arg == "--tokenizer-dir")
                parsed.tokenizer_model_dir = value(i, arg);
            else if (arg == "--tn-model-dir")
                parsed.tn_model_dir = value(i, arg);
            else if (arg == "--input" || arg == "-i")
                input_path = value(i, arg);
            else if (arg == "--output" || arg == "-o")
                output_path = value(i, arg);
            else if (arg == "--format")
                format = value(i, arg);
            else if (arg == "--language")
                language = value(i, arg);
            else if (arg == "--voice")
                voice = value(i, arg);
            else if (arg == "--speaker")
                request_options.speaker = parse_int(value(i, arg), arg, 0, 100000);
            else if (arg == "--sample-rate")
                output_rate = parse_int(value(i, arg), arg, 8000, 192000);
            else if (arg == "--device" || arg == "--backend") {
                device_name = value(i, arg);
                gpu = parse_device(device_name, arg);
                device_set = true;
            } else if (arg == "--seed")
                request_options.seed = parse_int(value(i, arg), arg, -1, 2147483647);
            else if (arg == "--steps")
                request_options.steps = parse_int(value(i, arg), arg, 1, 1000000);
            else if (arg == "--top-k")
                request_options.top_k = parse_int(value(i, arg), arg, 1, 1000000);
            else if (arg == "--temperature") {
                request_options.temperature = static_cast<float>(parse_double(value(i, arg), arg));
                request_options.override_temperature = true;
            } else if (arg == "--cfg-scale") {
                request_options.cfg_scale = static_cast<float>(parse_double(value(i, arg), arg));
                request_options.override_cfg_scale = true;
            } else if (arg == "--no-warmup")
                warmup = false;
            else if (arg == "--force")
                force = true;
            else if (!arg.empty() && arg[0] == '-') {
                bool consumed = false;
                if (!parser.ParseCliArg(arg, i + 1 < argc ? argv[i + 1] : nullptr, &consumed))
                    throw std::invalid_argument("unknown option: " + arg);
                if (consumed)
                    ++i;
            } else if (text.empty())
                text = arg;
            else
                text += " " + arg;
        }
        if (!input_path.empty()) {
            if (!text.empty())
                throw std::invalid_argument("TEXT and --input cannot be used together");
            text = read_text_file(input_path);
        }
        if (text.empty())
            throw std::invalid_argument("TEXT is required");
        if (format != "wav" && format != "pcm")
            throw std::invalid_argument("--format must be wav or pcm");
        if (cli_json() && output_path == "-")
            throw std::invalid_argument("--json cannot be combined with --output -");

        parsed.runtime.magpie_model =
            resolve_model_file(parsed.runtime.magpie_model, "tts", "MagpieTTS model").string();
        parsed.runtime.codec_model =
            resolve_model_file(parsed.runtime.codec_model, "codec", "NanoCodec model").string();
        parsed.tokenizer_model_dir =
            resolve_model_directory(parsed.tokenizer_model_dir, "tokenizer", "tokenizer model")
                .string();
        if (!parsed.tn_model_dir.empty())
            parsed.tn_model_dir =
                require_model_directory(parsed.tn_model_dir, "text normalization model").string();
        if (device_set) {
            bool cuda_device = device_name == "cuda" || device_name.rfind("cuda:", 0) == 0;
#if defined(NEMO_SPEECH_CLI_CUDA)
            cuda_device = cuda_device || device_name == "auto" || device_name == "gpu" ||
                          device_name.rfind("gpu:", 0) == 0;
#endif
            if (gpu < 0) {
                parsed.runtime.lt_backend = nemo_speech::tts::MagpieBackendPreference::Cpu;
                parsed.runtime.sampling_backend = nemo_speech::tts::MagpieBackendPreference::Cpu;
                parsed.runtime.magpie_cpu = true;
                parsed.runtime.codec_cpu = true;
            } else if (cuda_device) {
                parsed.runtime.lt_backend = nemo_speech::tts::MagpieBackendPreference::Cuda;
                parsed.runtime.codec_cpu = false;
            } else {
                parsed.runtime.lt_backend = nemo_speech::tts::MagpieBackendPreference::Cpu;
                parsed.runtime.sampling_backend = nemo_speech::tts::MagpieBackendPreference::Cpu;
                parsed.runtime.codec_cpu = false;
            }
        }
        parsed.runtime.verbose = cli_verbose();

        nemo_speech::tts::SynthesizerConfig config;
        config.runtime = parsed.runtime;
        config.tokenizer_model_dir = parsed.tokenizer_model_dir;
        config.text_normalizer_model_dir = parsed.tn_model_dir;
        config.tokenizer = parsed.tokenizer_config;
        config.default_language_code = parsed.default_language_code;
        config.default_voice_name = parsed.default_voice_name;
        nemo_speech::EngineRegistry engines;
        auto synthesizer = engines.load_tts(std::move(config));
        if (warmup)
            synthesizer->warmup("Hello", 1);

        nemo_speech::tts::SynthesisRequest request;
        request.text = text;
        request.language_code = language;
        request.voice_name = voice;
        request.output_sample_rate = output_rate;
        request.options = request_options;
        std::string pcm;
        nemo_speech::tts::SynthesisResult result;
        if (concurrency > 1) {
            // Fire the same request from N threads at one synthesizer. What it
            // measures is whether they share the wave: aggregate realtime
            // factor against a single request's, at the same time to first
            // audio.
            struct load_result {
                std::string pcm;
                double ttfa_ms = 0.0;
                double wall_s = 0.0;
                nemo_speech::tts::SynthesisResult result;
                std::string error;
            };
          // A warm-up burst first: the codec builds a graph per channel on its
          // first use, and those land on the first burst's first audio. Its
          // numbers are discarded.
          const int total_rounds = std::max(1, rounds) + (warmup ? 1 : 0);
          for (int round = (warmup ? -1 : 0); round < total_rounds - (warmup ? 1 : 0);
               ++round) {
            std::vector<load_result> runs((size_t)concurrency);
            std::vector<std::thread> threads;
            const auto load_start = std::chrono::steady_clock::now();
            for (int i = 0; i < concurrency; ++i) {
                threads.emplace_back([&, i] {
                    load_result& run = runs[(size_t)i];
                    // Requests arriving at a rate, rather than all at once.
                    // Time to first audio is measured from this request's own
                    // arrival, not from the start of the run.
                    if (arrival_ms > 0)
                        std::this_thread::sleep_for(std::chrono::milliseconds(arrival_ms * i));
                    const auto started = std::chrono::steady_clock::now();
                    // With --cancel-after-ms, every fourth request hangs up
                    // mid-stream. The rest must be unaffected, which is what
                    // says a cancelled session releases its lanes cleanly.
                    const bool hangs_up = cancel_after_ms > 0 && (i % 4) == 1;
                    try {
                        run.result = synthesizer->synthesize(
                            request, [&](const auto&, const std::string& chunk) {
                                if (hangs_up &&
                                    std::chrono::duration<double, std::milli>(
                                        std::chrono::steady_clock::now() - started)
                                            .count() > cancel_after_ms) {
                                    return false;
                                }
                                if (run.pcm.empty() && !chunk.empty()) {
                                    run.ttfa_ms = std::chrono::duration<double, std::milli>(
                                                      std::chrono::steady_clock::now() - started)
                                                      .count();
                                }
                                run.pcm += chunk;
                                return true;
                            });
                    }
                    catch (const std::exception& error) {
                        run.error = error.what();
                    }
                    run.wall_s =
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
                            .count();
                });
            }
            for (auto& thread : threads)
                thread.join();
            const double load_wall_s =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - load_start)
                    .count();

            double audio_s = 0.0;
            std::vector<double> ttfa;
            std::vector<double> durations;
            int failures = 0;
            int cancelled = 0;
            for (const load_result& run : runs) {
                if (!run.error.empty()) {
                    ++failures;
                    std::fprintf(stderr, "request failed: %s\n", run.error.c_str());
                    continue;
                }
                const double seconds =
                    (double)run.result.output_samples / run.result.metadata.sample_rate;
                audio_s += seconds;
                // A request that hung up before its first frame has no time to
                // first audio; counting it as zero would flatter the spread.
                if (run.ttfa_ms > 0.0) {
                    ttfa.push_back(run.ttfa_ms);
                }
                if (run.result.cancelled) {
                    ++cancelled;
                } else {
                    durations.push_back(seconds);
                }
            }
            std::sort(ttfa.begin(), ttfa.end());
            if (!cli_quiet()) {
                if (cancelled > 0) {
                    // Every request that ran to completion must still produce
                    // the same audio as it would have alone.
                    std::sort(durations.begin(), durations.end());
                    std::fprintf(
                        stderr, "%d cancelled, %zu completed spanning %.2f-%.2f s of audio\n",
                        cancelled, durations.size(), durations.empty() ? 0.0 : durations.front(),
                        durations.empty() ? 0.0 : durations.back());
                }
                std::fprintf(
                    stderr,
                    "%s: %d concurrent requests, %d failed: %.1f s of audio in %.2f s = "
                    "%.1fx realtime; first audio min/median/max %.0f/%.0f/%.0f ms\n",
                    round < 0 ? "warmup" : ("round " + std::to_string(round + 1)).c_str(),
                    concurrency, failures, audio_s, load_wall_s,
                    load_wall_s > 0.0 ? audio_s / load_wall_s : 0.0,
                    ttfa.empty() ? 0.0 : ttfa.front(),
                    ttfa.empty() ? 0.0 : ttfa[ttfa.size() / 2],
                    ttfa.empty() ? 0.0 : ttfa.back());
            }
            if (failures > 0)
                throw std::runtime_error("one or more concurrent requests failed");
            pcm = std::move(runs.front().pcm);
            result = runs.front().result;
          }
        } else {
            const auto started = std::chrono::steady_clock::now();
            if (cancel_after_ms > 0) {
                // Asked even before any audio exists, which is the case the PCM
                // callback below cannot cover.
                request.options.should_cancel = [&] {
                    return std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - started)
                               .count() > cancel_after_ms;
                };
            }
            result = synthesizer->synthesize(request, [&](const auto&, const std::string& chunk) {
                if (cancel_after_ms > 0 &&
                    std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - started)
                            .count() > cancel_after_ms) {
                    return false;
                }
                pcm += chunk;
                return true;
            });
        }
        if (result.cancelled) {
            // A caller that hangs up gets the audio produced so far, not an
            // error: the run was stopped, not broken.
            if (!cli_quiet()) {
                std::fprintf(
                    stderr, "cancelled after %llu samples (%.2f s)\n",
                    static_cast<unsigned long long>(result.output_samples),
                    result.metadata.sample_rate > 0
                        ? (double)result.output_samples / result.metadata.sample_rate
                        : 0.0);
            }
            return 0;
        }
        if (pcm.empty())
            throw std::runtime_error("synthesizer returned no audio");
        const std::string audio =
            format == "wav" ? nemo_speech::audio::pcm16_wav(pcm, result.metadata.sample_rate) : pcm;
        write_audio(output_path, audio, force);
        if (cli_json()) {
            nemo_speech::json::Value report(nemo_speech::json::Value::Object{});
            report["output"] = output_path;
            report["format"] = format;
            report["language"] = result.metadata.language_code;
            report["speaker"] = result.metadata.speaker;
            report["sample_rate"] = result.metadata.sample_rate;
            report["samples"] = static_cast<double>(result.output_samples);
            report["duration"] =
                static_cast<double>(result.output_samples) / result.metadata.sample_rate;
            report["rtfx"] = result.stats.e2e_rtfx;
            std::printf("%s\n", report.dump(2).c_str());
        } else if (!cli_quiet()) {
            std::fprintf(
                stderr, "wrote %s (%llu samples at %d Hz, %.2fx realtime)\n", output_path.c_str(),
                static_cast<unsigned long long>(result.output_samples), result.metadata.sample_rate,
                result.stats.e2e_rtfx);
        }
        return 0;
    }
    catch (const std::invalid_argument& error) {
        return print_cli_error(
            "synthesize", error.what(), kCliExitInvalidArgument, "invalid_argument");
    }
    catch (const std::exception& error) {
        return print_cli_exception("synthesize", error);
    }
}
