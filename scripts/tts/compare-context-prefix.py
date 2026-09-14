#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare the native context-encoder prefix against the NeMo reference.

The conditioning prefix is the tightest place to check a zero-shot checkpoint:
it is upstream of sampling, so a mismatch here is a real numerical difference
rather than a different random draw.

    scripts/tts/dump-magpie-reference.py ref.wav --out /tmp/ref   # NeMo side
    MAGPIETTS_CONTEXT_DUMP=/tmp/cpp.bin nemo-speech synthesize ... \
        --context-codes /tmp/ref/context-codes.txt
    scripts/tts/compare-context-prefix.py /tmp/ref/context-prefix.npy /tmp/cpp.bin
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reference_npy", type=Path)
    ap.add_argument("native_bin", type=Path)
    ap.add_argument("--rtol", type=float, default=2e-2,
                    help="the native runtime stores weights as f16, so exact equality "
                         "is not the bar; 2%% relative is")
    args = ap.parse_args()

    ref = np.load(args.reference_npy).astype(np.float32)
    native = np.fromfile(args.native_bin, dtype=np.float32)
    if native.size != ref.size:
        print(f"size mismatch: reference {ref.shape} = {ref.size} values, "
              f"native {native.size} values", file=sys.stderr)
        if ref.size and native.size % ref.shape[1] == 0:
            print(f"native looks like ({native.size // ref.shape[1]}, {ref.shape[1]})",
                  file=sys.stderr)
        return 1
    native = native.reshape(ref.shape)

    diff = np.abs(ref - native)
    scale = np.maximum(np.abs(ref), 1e-6)
    rel = diff / scale
    cos = float(
        (ref.ravel() @ native.ravel())
        / (np.linalg.norm(ref.ravel()) * np.linalg.norm(native.ravel()) + 1e-12)
    )

    print(f"shape              {ref.shape}")
    print(f"max abs diff       {diff.max():.6f}")
    print(f"mean abs diff      {diff.mean():.6f}")
    print(f"max rel diff       {rel.max():.6f}")
    print(f"cosine similarity  {cos:.8f}")
    print(f"reference range    [{ref.min():.4f}, {ref.max():.4f}]")

    # Cosine similarity is the honest headline: f16 weights shift individual
    # values but must not rotate the vector the decoder conditions on.
    ok = cos > 0.999 and rel.mean() < args.rtol
    print(f"\n{'PASS' if ok else 'FAIL'} (cosine > 0.999 and mean rel < {args.rtol})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
