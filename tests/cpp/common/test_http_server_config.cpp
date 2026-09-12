// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <iostream>
#include <stdexcept>
#include <string>

#include "engine_registry.h"
#include "http_server.h"

// tts.preempt used to require two HTTP workers, because one request could only
// start by cancelling the one holding the runtime. Synthesis is no longer
// serialized, so the setting is ignored and a single-worker server carrying it
// has to start rather than be rejected.
int
main() {
    nemo_speech::EngineRegistry engines;
    nemo_speech::http::ServerConfig config;
    config.threads = 1;
    config.preempt_tts = true;

    try {
        nemo_speech::http::Server server(engines, config);
    }
    catch (const std::exception& error) {
        std::cerr << "FAIL: a single-worker server with tts.preempt was rejected: " << error.what()
                  << '\n';
        return 1;
    }

    config.threads = 2;
    config.preempt_tts = false;
    try {
        nemo_speech::http::Server server(engines, config);
    }
    catch (const std::exception& error) {
        std::cerr << "FAIL: an ordinary two-worker server was rejected: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
