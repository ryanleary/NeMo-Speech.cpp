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
