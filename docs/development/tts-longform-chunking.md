# Long-form chunking against the reference

Everything else in the long-form path follows NeMo: the attention prior's
weights and constants, how a chunk ends, how much history it carries, the pause
between chunks. Two things in the sentence splitter deliberately do not, and
this is the measurement that says why.

## What NeMo does

The long-form path splits text per sentence with
`chunk_and_tokenize_text_by_sentence` (`tts_dataset_utils.py`). Its rule is
narrow: a terminal mark ends a sentence only when the very next character is
whitespace or end of string, with an exemption for title abbreviations. There is
no comma splitting. `magpie_serve`'s `split_text` does pack to
`max_chunk_chars = 300` and reach for a sentence's commas past that, but that
path serves batched synthesis, not long form.

`longform_group = 4` groups sentences for parallel decode. The group is not the
decode unit; the sentence is.

## Where we differ, and the two reasons

| | NeMo | here |
|---|---|---|
| closing quote after the terminal mark | not a sentence end | ends the sentence |
| long sentence | never split | split on commas past ~50 words |

On 20,000 characters of Pride and Prejudice, NeMo's rule gives **125 sentences,
median 125 characters, longest 692 characters / 760 tokens**. Ours gives 199.

760 tokens does not fit. The decoder's budget is `max_decoder_steps / stacking`
= 250 steps, which at two frames a step and 21.5 fps is 23 seconds of audio; a
692-character sentence needs about 45. Rendering the same two passages with
NeMo's rule instead of ours, at two wave widths:

| | ours | NeMo's rule |
|---|---|---|
| ref20k w8 | 1 stall 1.2 s, WER 2.30% | 2 stalls 5.7 s, 3.24% |
| ref20k w128 | 0 stalls, 2.27% | 1 stall 1.7 s, 2.79% |
| hold20k w8 | 3 stalls 16.2 s, 3.40% | 4 stalls 18.5 s, 4.55% |
| hold20k w128 | 4 stalls 9.7 s, 3.07% | 3 stalls 19.7 s, 4.86% |

and **two chunks per run run out of decoder steps**, which never happens with
ours. Word error rate is worse on all four. So the deviation is kept, and it is
the only place in the long-form path where matching upstream is measurably worse.

A third, smaller one: NeMo compares the raw whitespace-delimited word before a
period against its title list, so an opening quote defeats it and `"Mr.` ends a
sentence -- leaving the next chunk to open on a bare surname, which is audible.
Trimming non-letters first can only ever prevent a split, so it is trimmed here.

## Measuring this

Chunk counts come from the tokenizer alone and need no decode, which makes them
the cheapest comparison available:

```bash
# NeMo's sentences
python3 - <<'PY'
from magpie_serve.engine import load_model, DEFAULT_TOKENIZER
from nemo.collections.tts.parts.utils.tts_dataset_utils import (
    chunk_and_tokenize_text_by_sentence)
model, _ = load_model()
_, lens, texts = chunk_and_tokenize_text_by_sentence(
    text=open("sample.txt").read(), language="en",
    tokenizer_name=DEFAULT_TOKENIZER, text_tokenizer=model.tokenizer,
    eos_token_id=model.eos_id)
print(len(texts), max(int(l) for l in lens))
PY
```

Anything measured by rendering needs two disjoint passages and both wave widths.
A single passage gives opposite answers to the same question: tuning the
attention prior's advance threshold looked best at 4 on one and worst at 4 on
the other, and both readings were noise.

## The stall, and what it is really about

A chunk can come back as silence: the decoder emits quiet frames for its whole
length while its cross-attention stays pinned to the chunk's first token. Dumped
per step, the model's attention peak is sharp (peak/mean ~94) but frozen at
position 1, while the tracked position marches to 207 -- because the search
window is forward-only and the tracker monotonic, so once it passes the model it
can never come back. Every guard reads the tracker, not the model, so nothing
notices. That is true upstream too: `get_most_attended_text_timestep` slices
`alignment_attention_scores[last_attended:window_end]`, and a peak behind the
window is outside the slice.

What has been ruled out, on a 1,500-character reproducer that fails in both
decode paths deterministically:

| | |
|---|---|
| our attention prior causes it | no -- `apply_attention_prior: false`, still silent |
| the spliced history token | no -- `longform-history-tokens 0`, still silent |
| the model cannot say this text | no -- the reference speaks it |
| text normalization resolves the dash | no -- the WFST normalizer leaves `,--` untouched |
| the chunk is too long | no -- the reference speaks a 403-token chunk containing the same words |

What is left is where a chunk *starts*. On that passage the reference produces
two chunks, the first 403 tokens long and beginning at a sentence; we produced
eight, one of which began mid-clause at the dash, and that one went silent.

So the lever is chunk boundaries, and the question for the next pass is not "how
long may a chunk be" but "where may one begin". Note this cuts against the
closing-quote rule above: it was added because run-on sentences were being cut at
commas, but the reference solves the same problem by not cutting at all. Removing
both the quote rule and the comma split reproduces the reference's boundaries
(124 chunks against its 125) and measured worse -- but that was measured before
the mechanism was understood, and its two step-budget exhaustions suggest the
real prerequisite is a larger `max_decoder_steps`, not shorter chunks.

### Narrowed further: it depends on position in the request

Three more eliminations on the same reproducer, none of which fixed it:

| | |
|---|---|
| a dangling opening quote starts the chunk | no -- removing it, or closing it, still silent |
| the sentence is out of distribution | no -- it speaks alone in 6.0 s |
| it needs a preceding sentence for prosody | no -- it speaks with one in 13.0 s |

The same words, with the same chunk boundaries around them, are spoken when the
request holds two chunks and lost when it holds thirteen and they fall at the
fourth. So the trigger is not the text, its shape, or where the chunk begins in
the sentence -- it is *where the chunk falls in the request*.

That points at state carried between chunks rather than anything about the chunk
itself. Both paths reuse decoder state across chunks: the sequential path keeps
`persistent_runtime_` and reseeds its K/V per chunk, and a wave lane is reused
with its ring rotated. The next experiment is to force a fresh decoder for every
chunk and see whether the silence goes away; if it does, the question becomes
which piece of carried state is wrong, and `resetWave` / the persistent runtime's
reseed path are where to look.
