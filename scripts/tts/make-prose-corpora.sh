#!/usr/bin/env bash
# Write the seeded book-corpus prose inputs size-table.sh uses to /tmp. Same
# pool, seed and sampling as bench_longform_prose.py, so a given size is the
# same text every time and is comparable across arms.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 - "$HERE" <<'PY'
import pathlib, random, sys
sys.path.insert(0, sys.argv[1])
import bench_longform_prose as b
pool = b.pool(pathlib.Path.home() / ".cache/nemo-speech/prose")
print(f"pool: {len(pool)} sentences", file=sys.stderr)
for n in (16, 32, 64, 128, 256, 512, 1024, 2048, 4096):
    if n <= len(pool):
        pathlib.Path(f"/tmp/prose{n}.txt").write_text(" ".join(random.Random(7).sample(pool, n)))
        print(f"/tmp/prose{n}.txt", file=sys.stderr)
PY
