// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Where the tokenizer splits a paragraph into chunks.
//
// Each chunk becomes an independent decode: its own BOS, its own attention
// prior, its own boundary silence. A split inside a sentence is therefore
// audible, and the naive rule -- period followed by a space -- puts one inside
// every abbreviation. These cases pin the guard against that.
//
// Requires the tokenizer assets; skips (exit 0) without them.
#include <cstdio>
#include <string>
#include <vector>

#include "tts/tokenizer/tokenizer.h"

namespace {

int failures = 0;

std::vector<std::string>
chunks_of(const nemo_speech::tts::MagpieNativeTokenizer& tok, const std::string& text) {
    std::vector<std::string> out;
    for (const auto& c : tok.tokenize(text, "en-US").chunks) {
        out.push_back(c.text);
    }
    return out;
}

void
expect_chunks(
    const nemo_speech::tts::MagpieNativeTokenizer& tok, const std::string& text, size_t expected,
    const char* why) {
    const std::vector<std::string> got = chunks_of(tok, text);
    if (got.size() == expected) {
        return;
    }
    std::fprintf(
        stderr, "FAIL: %s\n  text:     %s\n  expected: %zu chunk(s), got %zu\n", why, text.c_str(),
        expected, got.size());
    for (const std::string& c : got) {
        std::fprintf(stderr, "    | %s\n", c.c_str());
    }
    ++failures;
}

}  // namespace

int
main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s TOKENIZER_MODEL_DIR\n", argv[0]);
        return 2;
    }
    nemo_speech::tts::MagpieNativeTokenizer tok(argv[1]);

    // Sentence chunking is gated on the *word count of the whole text* --
    // should_tokenize_by_sentence measures words, not characters, and English
    // needs 45 of them. Below that the text stays one chunk however it is
    // punctuated, so every case here is padded past the gate; otherwise the
    // test passes whether or not the splitter works.
    // Two filler sentences, 48 words, so every case clears the 45-word gate
    // with margin and the counts below are about the splitter, not the gate.
    const std::string filler =
        "Everyone in the surrounding villages had been waiting for news about the "
        "harvest for several weeks by then, and the roads were still difficult. "
        "The old bridge had been closed since the middle of the previous winter, "
        "which meant the only route left was the long one around the valley. ";

    // A real boundary splits once the gate is cleared.
    expect_chunks(
        tok,
        filler +
            "The rain continued steadily throughout the entire afternoon. "
            "Everyone waited inside until the sky finally cleared again.",
        4, "real sentence boundaries split");

    // Abbreviations do not add a boundary.
    expect_chunks(
        tok,
        filler +
            "The committee agreed that the U.S. government should be told "
            "about the decision before any of it becomes public.",
        3, "U.S. mid-sentence does not split");
    expect_chunks(
        tok,
        filler +
            "After a long and tedious meeting Dr. Smith agreed to review "
            "all of the remaining proposals himself before Friday.",
        3, "Dr. mid-sentence does not split");
    expect_chunks(
        tok,
        filler +
            "He recommended bringing something warm for the evening, e.g. "
            "a heavy coat, because the weather here turns very quickly.",
        3, "e.g. mid-sentence does not split");
    expect_chunks(
        tok,
        filler +
            "The visitors gathered outside before walking on to St. Mary "
            "and continuing along the river path together until dusk.",
        3, "St. mid-sentence does not split");

    // Both behaviours together.
    expect_chunks(
        tok,
        filler +
            "After a long meeting Dr. Smith agreed to review the proposals. "
            "Everyone else went home for the evening without waiting.",
        4, "abbreviation held, real boundary split");

    if (failures == 0) {
        std::printf("test_sentence_splitting: all checks passed\n");
    }
    return failures == 0 ? 0 : 1;
}
