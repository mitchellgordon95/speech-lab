"""Encode fixed 160 ms Mandarin windows using the same transform as live capture."""

import json
import time
from functools import lru_cache

import numpy as np
import soundfile as sf

from speechlab import models, phonetics

from .mandarin_prepare import BASE, PROTOCOL


@lru_cache(maxsize=128)
def utterance(ident):
    wave, sr = sf.read(BASE / "utterances" / f"{ident}.flac", dtype="float32")
    assert sr == 16000
    return wave


def crop(token, shift=0):
    wave = utterance(token["utterance"])
    start = round(((token["start"] + token["end"]) / 2 - 0.08 + shift) * 16000)
    assert start >= 0 and start + 2560 <= len(wave)
    return wave[start : start + 2560].copy()


def extract():
    samples = json.loads((BASE / "manifest.json").read_text())
    folder = BASE / "features"
    folder.mkdir(exist_ok=True)
    for name in ["acoustic"] + PROTOCOL["representations"]:
        path = folder / f"{name}.npz"
        if path.exists():
            print(name, "already complete", flush=True)
            continue
        checkpoint = folder / f"{name}-partial.npz"
        vectors = list(np.load(checkpoint)["x"]) if checkpoint.exists() else []
        if name != "acoustic":
            models.get_model(name)
        started = time.monotonic()
        for i in range(len(vectors), len(samples)):
            wave = crop(samples[i])
            vectors.append(
                phonetics.acoustic(wave) if name == "acoustic" else phonetics.embedding(wave, name)
            )
            if (i + 1) % 1000 == 0:
                temporary = checkpoint.with_suffix(".tmp.npz")
                np.savez_compressed(temporary, x=np.stack(vectors))
                temporary.replace(checkpoint)
                print(name, i + 1, "/", len(samples), round(time.monotonic() - started, 1), "s", flush=True)
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, x=np.stack(vectors), ids=np.array([s["id"] for s in samples]))
        temporary.replace(path)
        checkpoint.unlink(missing_ok=True)
        print(name, "complete", round(time.monotonic() - started, 1), "s", flush=True)
        models.unload()


if __name__ == "__main__":
    extract()
