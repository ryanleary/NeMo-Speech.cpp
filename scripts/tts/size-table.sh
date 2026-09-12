#!/usr/bin/env bash
# One binary, eight inputs: the three inlined cases the wave has always been
# gated on, plus five sizes of book-corpus prose.
#
#   [WIDTH=128] scripts/tts/size-table.sh <worktree> <label> [reps] [wave]
#
# wave=0 runs the stock configuration -- no batching flags, adaptive history --
# matching bench_longform_prose.py's --baseline arm. Needs the prose corpora:
# run make-prose-corpora.sh first.
#
# Prove the arm before trusting a number. nemo-speech is a thin CLI over
# libnemo_speech_tts.so.1, so a stale library is invisible:
#   nm -DC --defined-only <build>/bin/libnemo_speech_tts.so.1 | grep -c plan_wave_admission
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/bench-env.sh"
WT="${1:?usage: size-table.sh <worktree> <label> [reps] [wave]}"; LABEL="${2:?}"
REPS="${3:-3}"; WAVE="${4:-1}"
bench_resolve "$WT" || exit 1
EXTRA=(--tts.chunk-frames 32)
[ "$WAVE" = "1" ] && EXTRA+=(--tts.batch-size "${WIDTH:-32}" --tts.longform-history-tokens 20)
median() { sort -n | awk '{a[NR]=$1} END{print (NR%2)?a[(NR+1)/2]:(a[NR/2]+a[NR/2+1])/2}'; }
run_case() {
    local name="$1" txt="$2" xs=() cs="" a="" out
    for ((r = 0; r < REPS; r++)); do
        out=$("$BENCH_BIN" synthesize "$txt" --device cuda --voice John --seed 7 --top-k 1 \
            --output /tmp/size-table.wav --force --verbose "${BENCH_ARGS[@]}" "${EXTRA[@]}" \
            2>&1 >/dev/null) || { echo "$LABEL $name FAILED: $(echo "$out" | tail -1)"; return; }
        xs+=("$(echo "$out" | grep -oE 'e2e_rtfx=[0-9.]+' | tail -1 | cut -d= -f2)")
        cs=$(echo "$out" | grep -oE 'in [0-9]+ chunk' | tail -1 | grep -oE '[0-9]+')
        a=$(echo "$out" | grep -oE 'e2e_audio_s=[0-9.]+' | tail -1 | cut -d= -f2)
    done
    printf "%s %-10s chunks=%-5s audio=%-9s xRT=%s\n" "$LABEL" "$name" "${cs:-1}" "$a" \
        "$(printf '%s\n' "${xs[@]}" | median)"
}
run_case line      "$(bench_text_for 1)"
run_case paragraph "$(bench_text_for 5)"
run_case script    "$(bench_text_for 20)"
for n in 16 32 64 128 256; do
    [ -f "/tmp/prose$n.txt" ] || { echo "$LABEL prose$n SKIPPED (run make-prose-corpora.sh)"; continue; }
    run_case "prose$n" "$(cat "/tmp/prose$n.txt")"
done
