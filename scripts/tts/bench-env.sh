#!/usr/bin/env bash
# Shared locations for the long-form TTS benchmark and gate scripts. Source it;
# do not run it. Everything is overridable from the environment, so the scripts
# carry no one's home directory:
#
#   NEMO_SPEECH_BIN        nemo-speech binary (default: $WT/build/cuda-speech/bin)
#   NEMO_SPEECH_MAGPIE     MagpieTTS GGUF
#   NEMO_SPEECH_CODEC      NanoCodec decoder GGUF
#   NEMO_SPEECH_TOKENIZER  extracted tokenizer directory
#   NEMO_SPEECH_MODEL_DIR  root the two GGUF defaults hang off
#
# The defaults match the layout `nemo-speech download` leaves behind, which is
# what the numbers in docs/development/tts-wave-scheduler/ were measured against.
: "${NEMO_SPEECH_MODEL_DIR:=$HOME/.cache/nemo-speech/models/nvidia}"
: "${NEMO_SPEECH_MAGPIE:=$NEMO_SPEECH_MODEL_DIR/magpie_tts_multilingual_357m/452ef560f972c38d5fc16476259aac9456453547/magpie_tts_multilingual_357m.v2602.f16.gguf}"
: "${NEMO_SPEECH_CODEC:=$NEMO_SPEECH_MODEL_DIR/nemo-nano-codec-22khz-1.89kbps-21.5fps/fc00890b604aa2de298d2641ffc6c5f6caf8c4d7/nemo_nano_codec_22khz_1.89kbps_21.5fps.decoder.f16.gguf}"
: "${NEMO_SPEECH_TOKENIZER:=$HOME/nemo-bench-assets/tokenizer-v2602}"

# The three inlined cases the wave has always been gated on. Repeating five
# sentences keeps chunk count proportional to length with no sampling noise --
# which also means a low straggler ratio, so prose is what the scheduler work is
# actually measured on. See make-prose-corpora.sh.
bench_pool=(
    "Magpie is a text to speech model."
    "It generates audio codes autoregressively and then decodes them with a neural codec."
    "This paragraph is long enough that the server treats it as long form input."
    "We want to know how the wall clock time scales with the number of sentences."
    "Here is more filler text to push us well past the language threshold."
)
bench_text_for() {
    local n=$1 out="" i
    for ((i = 0; i < n; i++)); do out+="${bench_pool[$((i % 5))]} "; done
    echo "${out% }"
}

# Resolve the binary for a worktree, and fail loudly rather than measuring a
# model that is not there.
bench_resolve() {
    local wt="${1:-.}"
    BENCH_BIN="${NEMO_SPEECH_BIN:-$wt/build/cuda-speech/bin/nemo-speech}"
    local missing=0 p
    for p in "$BENCH_BIN" "$NEMO_SPEECH_MAGPIE" "$NEMO_SPEECH_CODEC" "$NEMO_SPEECH_TOKENIZER"; do
        [ -e "$p" ] || { echo "missing: $p" >&2; missing=1; }
    done
    [ "$missing" = 0 ] || return 1
    BENCH_ARGS=(
        --tts.magpie-model "$NEMO_SPEECH_MAGPIE"
        --tts.codec-model "$NEMO_SPEECH_CODEC"
        --tts.tokenizer-model-dir "$NEMO_SPEECH_TOKENIZER"
    )
}
