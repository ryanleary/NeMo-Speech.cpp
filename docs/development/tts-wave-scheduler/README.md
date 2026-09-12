# The MagpieTTS wave scheduler

Long-form synthesis decodes a script as a sequence of chunks. The wave decodes
several at once through one graph; continuous batching keeps those lanes full
rather than retiring them together. These are the working notes behind that
work -- what was measured, what the measurements ruled out, and where the
numbers came from.

| | |
|---|---|
| [continuous-batching.md](continuous-batching.md) | The design, what shipped, and the results. **Start here.** |
| [stragglers.md](stragglers.md) | Why a fixed group pays for its slowest member, and what that costs. |
| [serving.md](serving.md) | Many concurrent requests through one wave: what is done, the numbers, and the decision that gates the rest. |
| [handoff.md](handoff.md) | What the wave batches, and the state of the two benchmark machines. |
| [joint-tuning.md](joint-tuning.md) | Width and chunk-frames interact; tuning one alone misleads. |
| [pr-body.md](pr-body.md), [pr-table.md](pr-table.md) | Review-facing summaries. |
| [data/](data/) | Per-run metric dumps, per-chunk step counts, and the payoff projection. |

Reproducing anything here needs `scripts/tts/`: `greedy-hashes.sh` and
`wave-hashes.sh` are the gates, `size-table.sh` and `make-prose-corpora.sh`
produce the tables, and `bench-env.sh` holds the model locations they share.

Two traps worth knowing before trusting a number, both of which cost time here:

- `greedy-hashes.sh` runs the **sequential** path -- it passes no
  `--tts.batch-size` -- so it cannot see a wave regression at all.
- `nemo-speech` is a thin CLI over `libnemo_speech_tts.so.1`. Prove which arm
  you built by symbol, not by which directory you are standing in.
