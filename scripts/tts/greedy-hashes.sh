#!/usr/bin/env bash
# Greedy output hashes for the three inlined cases, for byte-identity checks.
#
#   scripts/tts/greedy-hashes.sh <worktree> <label>
#
# This runs the SEQUENTIAL path: it passes no --tts.batch-size, so it cannot see
# a wave-scheduler regression. Use wave-hashes.sh for that.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/bench-env.sh"
WT="${1:?usage: greedy-hashes.sh <worktree> <label>}"; LABEL="${2:?}"
bench_resolve "$WT"
for spec in line:1 paragraph:5 script:20; do
    name=${spec%%:*}; n=${spec##*:}
    "$BENCH_BIN" synthesize "$(bench_text_for "$n")" --device cuda --voice John \
        --seed 7 --top-k 1 --output "/tmp/hash-$name.wav" --force \
        "${BENCH_ARGS[@]}" >/dev/null 2>&1
    echo "$LABEL $name $(sha256sum "/tmp/hash-$name.wav" | cut -c1-16) $(stat -c%s "/tmp/hash-$name.wav")"
done
