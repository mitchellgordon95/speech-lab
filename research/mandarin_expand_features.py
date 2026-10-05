"""Encode expanded 160 ms windows with the previously selected Qwen encoder."""

import json
import time

import numpy as np

from speechlab import models, phonetics

from .mandarin_expand_prepare import OUT
from .mandarin_features import crop


def extract():
    rows = json.loads((OUT / "manifest.json").read_text())
    path = OUT / "features.npz"
    partial = OUT / "features-partial.npz"
    if path.exists():
        np.testing.assert_array_equal(np.load(path)["ids"], [r["id"] for r in rows])
        print("Already complete")
        return
    features = list(np.load(partial)["x"]) if partial.exists() else []
    models.get_model("qwen-asr-large")
    started = time.monotonic()
    for i in range(len(features), len(rows)):
        features.append(phonetics.embedding(crop(rows[i]), "qwen-asr-large"))
        if (i + 1) % 1000 == 0:
            temp = partial.with_suffix(".tmp.npz")
            np.savez_compressed(temp, x=np.stack(features))
            temp.replace(partial)
            print(i + 1, "/", len(rows), round(time.monotonic() - started, 1), "seconds", flush=True)
    temp = path.with_suffix(".tmp.npz")
    np.savez_compressed(temp, x=np.stack(features), ids=[r["id"] for r in rows])
    temp.replace(path)
    partial.unlink(missing_ok=True)
    print("Complete", len(rows), round(time.monotonic() - started, 1), "seconds", flush=True)


if __name__ == "__main__":
    extract()
