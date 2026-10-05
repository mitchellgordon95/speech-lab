"""Add deterministic timing variation on development speakers only."""

import json
import time
from collections import defaultdict

import numpy as np

from speechlab import phonetics

from .mandarin_expand_prepare import OUT, stable
from .mandarin_features import crop


def main():
    rows = json.loads((OUT / "manifest.json").read_text())
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        if r["split"] != "test":
            groups[(r["speaker"], r["phone"])].append(i)
    selected = []
    for indices in groups.values():
        for i in sorted(indices, key=lambda i: stable(rows[i]["id"] + ":jitter"))[:3]:
            shift = [-40, -20, 20, 40][int(stable(rows[i]["id"] + ":shift")[-2:], 16) % 4]
            selected.append(dict(index=i, id=rows[i]["id"], shift_ms=shift, split=rows[i]["split"]))
    selected.sort(key=lambda r: r["id"])
    (OUT / "jitter-manifest.json").write_text(json.dumps(selected, indent=2))
    partial = OUT / "jitter-partial.npz"
    path = OUT / "jitter.npz"
    if path.exists():
        np.testing.assert_array_equal(np.load(path)["ids"], [r["id"] for r in selected])
        print("Jitter complete")
        return
    features = list(np.load(partial)["x"]) if partial.exists() else []
    start = time.monotonic()
    for n in range(len(features), len(selected)):
        item = selected[n]
        features.append(
            phonetics.embedding(crop(rows[item["index"]], item["shift_ms"] / 1000), "qwen-asr-large")
        )
        if (n + 1) % 500 == 0:
            np.savez_compressed(partial, x=np.stack(features))
            print(n + 1, "/", len(selected), "seconds", round(time.monotonic() - start, 1), flush=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, x=np.stack(features), ids=[r["id"] for r in selected])
    temporary.replace(path)
    partial.unlink(missing_ok=True)
    print("Complete", len(selected), round(time.monotonic() - start, 1), flush=True)


if __name__ == "__main__":
    main()
