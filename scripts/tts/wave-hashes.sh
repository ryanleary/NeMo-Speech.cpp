#!/usr/bin/env bash
# Greedy output hashes for the WAVE path, for byte-identity checks.
#
#   [CASES="strict:17 script:20"] scripts/tts/wave-hashes.sh <worktree> <label> [widths...]
#
# greedy-hashes.sh covers the sequential path only -- it passes no
# --tts.batch-size -- so it cannot see a wave-scheduler regression.
#
# Greedy output IS width-dependent (batched matmul numerics), so only ever
# compare like width with like. It is also LANE-dependent: permuting which chunk
# occupies which lane moves the hash, so continuous batching cannot be gated on
# byte-identity in general. `strict:17` at width 8 is the case that can be --
# 1 + 8 + 8 chunks, every cohort full, so every chunk sees the same batch width
# it would have seen before. See docs/development/tts-wave-scheduler/.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/bench-env.sh"
WT="${1:?usage: wave-hashes.sh <worktree> <label> [widths...]}"; LABEL="${2:?}"; shift 2
WIDTHS=("${@:-8 32}")
bench_resolve "$WT"
for spec in ${CASES:-strict:17 script:20 long:48}; do
    name=${spec%%:*}; n=${spec##*:}
    for w in ${WIDTHS[@]}; do
        out="/tmp/wave-hash-$name-w$w.wav"
        if "$BENCH_BIN" synthesize "$(bench_text_for "$n")" --device cuda --voice John \
            --seed 7 --top-k 1 --output "$out" --force "${BENCH_ARGS[@]}" \
            --tts.batch-size "$w" --tts.longform-history-tokens 20 --tts.chunk-frames 32 \
            >/dev/null 2>"/tmp/wave-hash-$name-w$w.log"; then
            echo "$LABEL $name w$w $(sha256sum "$out" | cut -c1-16) $(stat -c%s "$out")"
        else
            echo "$LABEL $name w$w FAILED $(tail -2 "/tmp/wave-hash-$name-w$w.log" | tr '\n' ' ')"
        fi
    done
done
