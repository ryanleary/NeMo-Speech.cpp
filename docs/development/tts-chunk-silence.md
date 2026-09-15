# A chunk that decodes to silence

Standalone handoff. The defect is open; this is what is known, what has been
ruled out, and what to try next.

## The defect

A long-form chunk sometimes returns silence. The decoder runs its full step
budget emitting quiet frames (RMS ~12 against ~2900 for speech) while its
cross-attention stays pinned to the chunk's first token, and the words in that
chunk never reach the audio. On nine and a half hours it costs about 0.2% of
words, in two to eleven runs of three or more words.

Instrumented per step, the model's attention peak is *sharp* -- peak/mean ~94,
the same as a healthy chunk -- but frozen at position 1, while the tracked
position walks to 207. Our search window is forward-only and the tracker
monotonic, so once it passes the model it can never resynchronise, and every
guard we have (sink penalty, forceful chunk end, EOS masking) reads the tracker
rather than the model. Upstream shares that blind spot:
`get_most_attended_text_timestep` argmaxes `alignment_attention_scores[last:
window_end]`, and a peak behind the window is outside the slice.

## Reproducing it

From the cleaned Gutenberg text, take ±300/1200 characters around "I have an
excessive regard", and replace the `,--` before it with `,-`: a single hyphen
still flattens to whitespace, so it defeats the dash fix and restores the
original chunking. About 1,500 characters. It fails in **both** decode paths and
is deterministic at that size, which is what makes it usable -- everything else
that looked like a dropped phrase moved between runs.

Check with `build/cuda-asr/bin/nemo-speech transcribe` and grep for the phrase.

## Eliminated

| hypothesis | test | result |
|---|---|---|
| our attention prior causes the pin | `apply_attention_prior: false` | still silent |
| the spliced history token | `--tts.longform-history-tokens 0` | still silent |
| the model cannot say this text | same passage through the reference | reference speaks it |
| text normalization resolves the dash | ran the WFST normalizer alone | `,--` untouched |
| the chunk is too long | reference speaks a 403-token chunk of the same words; ours fails at 207 | length is not the variable |
| a dangling opening quote opens the chunk | removed it, and closed it | still silent |
| the text is out of distribution | rendered the sentence alone | speaks, 6.0 s |
| it needs a preceding sentence for prosody | rendered with one | speaks, 13.0 s |
| the wave causes it | sequential has *more* multi-word drops than the wave on the same slice (10 against 5) | not the wave |

## Leading hypotheses

**1. Decoder state carried between chunks.** The strongest, because it is the
one thing that distinguishes the failing case from every passing one: identical
words with identical chunk boundaries are spoken when the request holds two
chunks and lost when they fall fourth of thirteen. Both paths reuse state -- the
sequential path keeps `persistent_runtime_` and reseeds its K/V per chunk, a wave
lane is reused with its ring rotated. It also explains why eight of nine book
deletions evaporated in isolation, and why the sites move between runs at book
scale but not at 100k.

*Experiment:* force a fresh decoder for every chunk (`resetWave` between
admissions; defeat the `persistent_runtime_` reuse) and see whether the silence
goes. If it does, bisect which piece of carried state is wrong -- the reseed path
and the ring rotation are the candidates.

**2. Where a chunk may begin.** Secondary, and possibly the same thing seen from
the text side. The reference produces two chunks for the passage where we produce
eight, and never creates one that opens mid-clause. Note this cuts against the
closing-quote rule in `tts-longform-chunking.md`: that rule exists because run-on
sentences were being cut at commas, but the reference solves the same problem by
not cutting at all.

*Experiment:* remove the closing-quote rule and the comma split together -- that
reproduces the reference's boundaries, 124 chunks against its 125. It measured
worse when tried, but before the mechanism was understood, and its two
step-budget exhaustions suggest the prerequisite is a larger `max_decoder_steps`
rather than shorter chunks.

**3. Detection, whatever the cause.** The signal is free: the model's global
attention argmax sitting far behind the tracked position for many steps. Median
|peak - tracked| is 0 in a healthy chunk and 120 in a stalling one, held for 120
consecutive steps. We compute the full alignment vector every step and discard
everything outside the window.

*Caution:* what to do on detection -- resync the tracker backward, end the chunk,
retry it -- has no upstream precedent, so measure rather than assume.

## Measurement traps

- **The book is not an instrument.** At 6,000+ chunks the scheduler makes runs
  non-repeatable: the multi-word deletion count swings 2 to 11 on identical
  input, with no site in common between runs. Measure on a fixed ~100,000
  character slice, where four renders across two decode paths were byte-identical.
- **Two disjoint passages, both widths, every time.** Tuning the attention
  prior's advance threshold looked best at 4 on one passage and worst at 4 on
  another. Both readings were noise.
- **Word error rate alone hides the defect.** Substitution rate is the ASR's own
  floor and barely moves; deletions are the signal. Split them.
- **Use an aligner, not `difflib`.** `SequenceMatcher` finds longest matching
  blocks, so a one-word substitution inside a long run is reported as a delete
  plus an insert, inventing missing phrases. Use a banded Levenshtein with
  backtrace.
- **Transcribe with nemo-speech's own ASR.** Whisper in fixed windows loses words
  at every boundary and invented several "deletions" that were not in the audio.
