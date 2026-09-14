#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capture reference tensors from inside a real NeMo synthesis call.

This *captures* rather than reconstructs. An earlier version of this tooling
rebuilt the context embedding by calling `embed_audio_tokens` and the context
encoder directly, which reproduced the same misunderstanding the C++ had and
reported a false pass at cosine 0.99999763. Hooking the model's own modules
during `engine.synthesize()` is the only way to be sure the reference is what
the working implementation actually does.

Writes to --out:
  text-tokens.txt      text token IDs, in this checkpoint's vocabulary
  context-codes.txt    reference-audio codec codes, BOS/EOS stripped (the
                       native runtime adds its own)
  ce-in.npy            context-encoder input  (padded length x E)
  ce-out.npy           context-encoder output = the decoder's prefix
  prefix.npy           additional_decoder_input, should equal ce-out
  generated-codes.txt  the codec frames NeMo generated

Run from the magpie-tts-server checkout with its venv:

    .venv/bin/python .../capture-magpie-reference.py --out /tmp/ref
"""
import argparse
import sys, json, os, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.expanduser(
    os.environ.get("MAGPIE_REPO", "~/devel/magpie-tts-server")), "src"))
from magpie_serve import engine
_ap = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument("--out", default="/tmp/ref")
_ap.add_argument("--text", default="Magpie is a text to speech model.")
_ap.add_argument("--magpie-repo", default=os.environ.get(
    "MAGPIE_REPO", os.path.expanduser("~/devel/magpie-tts-server")))
_args = _ap.parse_args()
OUT = _args.out; os.makedirs(OUT, exist_ok=True)
model,_ = engine.load_model()
cap={}

orig_prep = model.prepare_context_tensors
def prep(batch):
    cap["text"]=batch["text"].detach().cpu().numpy()
    cap["text_lens"]=batch["text_lens"].detach().cpu().numpy()
    out = orig_prep(batch)
    # additional_decoder_input is the conditioning prefix the decoder consumes
    adi = getattr(out, "additional_decoder_input", None)
    adm = getattr(out, "additional_decoder_mask", None)
    if adi is not None:
        cap["prefix"]=adi.detach().float().cpu().numpy()
    if adm is not None:
        cap["prefix_mask"]=adm.detach().float().cpu().numpy()
    return out
model.prepare_context_tensors = prep

orig_ce = model.context_encoder.forward
def ce(x, mask, *a, **kw):
    cap["ce_in"]=x.detach().float().cpu().numpy()
    cap["ce_mask"]=mask.detach().float().cpu().numpy()
    r = orig_ce(x, mask, *a, **kw)
    cap["ce_out"]=r["output"].detach().float().cpu().numpy()
    return r
model.context_encoder.forward = ce

orig_gcac = model._get_context_audio_codes
def gcac(batch):
    codes, lens = orig_gcac(batch)
    cap["wrapped_codes"]=codes.detach().cpu().numpy(); cap["wrapped_lens"]=lens.detach().cpu().numpy()
    return codes, lens
model._get_context_audio_codes = gcac

orig_c2a = model._codec_helper.codes_to_audio
def c2a(codes, lens, *a, **kw):
    cap.setdefault("out_codes", codes.detach().cpu().numpy()); cap.setdefault("out_lens", lens.detach().cpu().numpy())
    return orig_c2a(codes, lens, *a, **kw)
model._codec_helper.codes_to_audio = c2a

res = engine.synthesize(_args.text, engine.resolve_voice(None),
                        engine.SynthParams(topk=1, temperature=0.01), seed=1)
print("audio", res.duration)
for k in ("ce_in","ce_mask","ce_out","prefix","prefix_mask","wrapped_lens"):
    v=cap.get(k)
    print(k, None if v is None else (v.shape if hasattr(v,'shape') else v))

t=cap["text"][0][:int(cap["text_lens"][0])]
open(f"{OUT}/text-tokens.txt","w").write(" ".join(str(int(x)) for x in t)+"\n")
w=cap["wrapped_codes"][0]; wl=int(cap["wrapped_lens"][0])
with open(f"{OUT}/context-codes.txt","w") as fh:
    for i in range(1, wl-1):
        fh.write(" ".join(str(int(w[c,i])) for c in range(w.shape[0]))+"\n")
np.save(f"{OUT}/ce-out.npy", cap["ce_out"][0])
np.save(f"{OUT}/ce-in.npy", cap["ce_in"][0])
np.save(f"{OUT}/prefix.npy", cap["prefix"][0])
oc=cap["out_codes"][0]; ol=int(cap["out_lens"][0])
with open(f"{OUT}/generated-codes.txt","w") as fh:
    for i in range(ol):
        fh.write(" ".join(str(int(oc[c,i])) for c in range(oc.shape[0]))+"\n")
print("context frames written:", wl-2, "generated frames:", ol, "text tokens:", len(t))
