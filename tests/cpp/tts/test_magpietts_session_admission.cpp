// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// plan_session_admission decides which request fills each idle lane when one
// wave carries chunks from several. Both ways of getting it wrong are invisible
// at runtime: starve a new request and its time to first audio collapses into
// whatever the busy ones are doing, or let one request take every lane and the
// others stall behind it for the length of a script. Neither fails a run.
#include <cstdio>
#include <vector>

#include "tts/magpietts/magpietts.h"

namespace tts = nemo_speech::tts;

namespace {

int failures = 0;

void
expect(const std::vector<int>& got, const std::vector<int>& want, const char* msg) {
    if (got != want) {
        std::fprintf(stderr, "FAIL %s: got [", msg);
        for (int v : got) std::fprintf(stderr, "%d ", v);
        std::fprintf(stderr, "], want [");
        for (int v : want) std::fprintf(stderr, "%d ", v);
        std::fprintf(stderr, "]\n");
        ++failures;
    }
}

// "3+" is a session with 3 chunks pending that already holds a lane; "3-" is one
// with 3 pending and nothing in flight.
std::vector<tts::MagpieSessionDemand>
demand(std::initializer_list<std::pair<size_t, bool>> spec) {
    std::vector<tts::MagpieSessionDemand> out;
    for (const auto& s : spec) out.push_back({s.first, s.second});
    return out;
}

std::vector<int>
lanes(int n) {
    std::vector<int> out;
    for (int i = 0; i < n; ++i) out.push_back(i);
    return out;
}

}  // namespace

int
main() {
    size_t turn = 0;

    // A request with nothing in flight is served before one that is already
    // decoding, however much the busy one has queued. This is the whole latency
    // rule: it only needs one lane to start producing audio.
    turn = 0;
    expect(
        tts::plan_session_admission(demand({{9, true}, {1, false}}), lanes(2), turn), {1, 0},
        "unoccupied session jumps the queue");

    // But only one lane each -- enough to start, not enough to take the wave.
    turn = 0;
    expect(
        tts::plan_session_admission(demand({{4, false}, {4, false}}), lanes(4), turn),
        {0, 1, 0, 1}, "new sessions take one lane each, then round robin");

    // With everyone occupied it is pure round robin, and the cursor persists so
    // the same session does not win every burst.
    turn = 0;
    std::vector<tts::MagpieSessionDemand> busy = demand({{2, true}, {2, true}, {2, true}});
    expect(tts::plan_session_admission(busy, lanes(3), turn), {0, 1, 2}, "round robin");
    expect(tts::plan_session_admission(busy, lanes(3), turn), {0, 1, 2}, "cursor wrapped");

    turn = 1;
    expect(
        tts::plan_session_admission(busy, lanes(2), turn), {1, 2}, "cursor resumes where it left");

    // A session with nothing pending is skipped even when unoccupied -- it has
    // finished, not stalled.
    turn = 0;
    expect(
        tts::plan_session_admission(demand({{0, false}, {2, true}}), lanes(2), turn), {1, 1},
        "nothing pending is skipped");

    // More lanes than work leaves the tail unclaimed rather than inventing it.
    turn = 0;
    expect(
        tts::plan_session_admission(demand({{1, true}, {1, true}}), lanes(4), turn),
        {0, 1, -1, -1}, "more lanes than chunks");

    // Degenerate inputs must not fill anything.
    turn = 0;
    expect(tts::plan_session_admission({}, lanes(2), turn), {-1, -1}, "no sessions");
    expect(tts::plan_session_admission(demand({{5, true}}), {}, turn), {}, "no lanes");

    // One session is the single-request case, and must behave exactly as
    // plan_wave_admission already did: take everything it is offered.
    turn = 0;
    expect(
        tts::plan_session_admission(demand({{8, true}}), lanes(3), turn), {0, 0, 0},
        "one session takes every lane");

    if (failures == 0) std::fprintf(stderr, "OK magpietts_session_admission\n");
    return failures == 0 ? 0 : 1;
}
