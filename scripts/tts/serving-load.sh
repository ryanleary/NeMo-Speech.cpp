#!/usr/bin/env bash
# Aggregate throughput and first-audio latency against concurrent requests.
#
#   [SENTENCES=5] [CONCURRENCY="1 4 16 32"] scripts/tts/serving-load.sh <worktree> [width]
#
# Every request is the same text, submitted at the same moment, through one
# synthesizer -- which is the point: they can only go faster together if they
# share the wave. Compare the aggregate realtime factor against the first row,
# which is one request on its own.
#
# The wave only ever grows, and it is sized from everything queued at the time
# it opens, so a run that starts with a single request keeps a narrow wave until
# it drains. Each row is its own process for that reason.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/bench-env.sh"
WT="${1:?usage: serving-load.sh <worktree> [width]}"; WIDTH="${2:-32}"
bench_resolve "$WT"
TEXT="$(bench_text_for "${SENTENCES:-5}")"
printf '%-8s %-10s %-12s %s\n' requests aggregate "ttfa median" "ttfa min/max"
for c in ${CONCURRENCY:-1 4 16 32}; do
    log="/tmp/serving-load-w$WIDTH-c$c.log"
    "$BENCH_BIN" synthesize "$TEXT" --device cuda --voice John --seed 7 --top-k 1 \
        --output "/tmp/serving-load.wav" --force "${BENCH_ARGS[@]}" \
        --tts.batch-size "$WIDTH" --tts.longform-history-tokens 20 --tts.chunk-frames 32 \
        --concurrency "$c" >/dev/null 2>"$log" || { echo "$c FAILED $(tail -2 "$log")"; continue; }
    if [ "$c" = 1 ]; then
        # A lone request reports itself rather than an aggregate.
        rtfx=$(grep -oE '[0-9.]+x realtime' "$log" | tail -1 | cut -d' ' -f1)
        printf '%-8s %-10s %-12s %s\n' "$c" "$rtfx" "-" "-"
    else
        line=$(grep 'concurrent requests' "$log")
        rtfx=$(echo "$line" | grep -oE '= [0-9.]+x' | tr -d '= ')
        spread=$(echo "$line" | grep -oE '[0-9]+/[0-9]+/[0-9]+ ms')
        printf '%-8s %-10s %-12s %s\n' "$c" "$rtfx" \
            "$(echo "$spread" | cut -d/ -f2) ms" "$(echo "$spread" | cut -d/ -f1)/$(echo "$spread" | cut -d/ -f3)"
    fi
done
