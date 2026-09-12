// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// plan_wave_admission is the wave scheduler's whole admission policy: when a
// lane that has finished its chunk gets the next one, and which lanes go
// together in a burst. It is pure -- no model, no device -- and getting it
// wrong is expensive in both directions. Admit too eagerly and the run spends
// more on prefills than refilling saves: at width 32 over 150 chunks of prose,
// a threshold of one lane costs 112x realtime against 134x at four. Admit too
// lazily and the lanes sit idle: twenty-four costs 124x. Neither shows up as a
// failure, only as a slower run, which is exactly the kind of thing a test
// should pin.
#include <cstdio>
#include <vector>

#include "tts/magpietts/magpietts.h"

namespace tts = nemo_speech::tts;

namespace {

int failures = 0;

void
expect_lanes(
    const std::vector<int>& got, const std::vector<int>& want, const char* msg) {
    if (got != want) {
        std::fprintf(stderr, "FAIL %s: got [", msg);
        for (int lane : got) std::fprintf(stderr, "%d ", lane);
        std::fprintf(stderr, "], want [");
        for (int lane : want) std::fprintf(stderr, "%d ", lane);
        std::fprintf(stderr, "]\n");
        ++failures;
    }
}

// `....` is four live lanes; `x` marks one whose chunk has finished.
std::vector<char>
lanes(const char* pattern) {
    std::vector<char> out;
    for (const char* c = pattern; *c; ++c) out.push_back(*c == 'x' ? 1 : 0);
    return out;
}

}  // namespace

int
main() {
    // Below the threshold the wave keeps decoding: a burst of one would pay a
    // whole prefill to fill a single lane.
    expect_lanes(tts::plan_wave_admission(lanes("x..."), 10, 2), {}, "below threshold");
    expect_lanes(tts::plan_wave_admission(lanes("...."), 10, 2), {}, "nothing idle");

    // At the threshold every idle lane is filled, not just the threshold's
    // worth: the prefill is already being paid for.
    expect_lanes(tts::plan_wave_admission(lanes("x.x."), 10, 2), {0, 2}, "at threshold");
    expect_lanes(tts::plan_wave_admission(lanes("xxx."), 10, 2), {0, 1, 2}, "over threshold");

    // Lanes come back ascending, and scattered lanes are fine -- the prefill
    // opens any set, so survivors never have to be moved to make room.
    expect_lanes(
        tts::plan_wave_admission(lanes(".x..x...x."), 10, 3), {1, 4, 8}, "scattered, ascending");

    // A wave with nothing live would otherwise sit doing nothing until a
    // threshold it can no longer reach, so it admits whatever it has.
    expect_lanes(tts::plan_wave_admission(lanes("xxxx"), 10, 8), {0, 1, 2, 3}, "stalled");
    expect_lanes(tts::plan_wave_admission(lanes("x..."), 10, 99), {}, "not stalled, high bar");

    // Never more chunks than there are left. This is the tail of a run: the
    // remaining lanes keep their finished chunks and idle, which costs a fixed
    // width of decode but no correctness.
    expect_lanes(tts::plan_wave_admission(lanes("xxxx"), 2, 2), {0, 1}, "fewer pending than idle");
    expect_lanes(tts::plan_wave_admission(lanes("xxxx"), 0, 2), {}, "nothing pending");
    expect_lanes(tts::plan_wave_admission(lanes("x..."), 0, 1), {}, "nothing pending, stalled bar");

    // A threshold of zero or less must not mean "admit on every idle lane
    // regardless", which would be the singly-admitting worst case.
    expect_lanes(tts::plan_wave_admission(lanes("x..."), 10, 0), {0}, "threshold floors at one");

    // Width one is the degenerate wave: its only lane is either live or done.
    expect_lanes(tts::plan_wave_admission(lanes("x"), 5, 4), {0}, "single lane, stalled");
    expect_lanes(tts::plan_wave_admission(lanes("."), 5, 1), {}, "single lane, live");

    if (failures == 0) std::fprintf(stderr, "OK magpietts_wave_admission\n");
    return failures == 0 ? 0 : 1;
}
