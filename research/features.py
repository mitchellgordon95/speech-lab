"""Extract all benchmark representations, checkpointing progress after each block."""

import argparse
import json
import time
import warnings
from functools import lru_cache

import numpy as np
import soundfile as sf
from huggingface_hub import hf_hub_download
from sklearn.exceptions import InconsistentVersionWarning

from demos.articulation import LinearModelUnpickler
from speechlab import models, phonetics

from .prepare import BASE


@lru_cache(maxsize=128)
def utterance(ident):
    y, sr = sf.read(BASE / "utterances" / f"{ident}.flac", dtype="float32")
    assert sr == 16000
    return y


def crop(token, shift=0):
    y = utterance(token["utterance"])
    return y[max(0, round((token["start"] + shift) * 16000)) : round((token["end"] + shift) * 16000)].copy()


def extract(names):
    samples = json.loads((BASE / "manifest.json").read_text())
    out = BASE / "features"
    out.mkdir(exist_ok=True)
    for name in names:
        target = out / f"{name}.npz"
        if target.exists():
            print(name, "already complete", flush=True)
            continue
        linear = None
        if name == "sparc":
            path = hf_hub_download(
                "cheoljun95/Speech-Articulatory-Coding",
                "wavlm_large-9_cut-10_mngu_linear.pkl",
                revision="2e6a07d4b35022d4366a2896a6293fe16aaed78a",
            )
            with warnings.catch_warnings(), open(path, "rb") as f:
                warnings.simplefilter("ignore", InconsistentVersionWarning)
                linear = LinearModelUnpickler(f).load()
        if name in models.MODELS:
            models.get_model(name, progress=print)
        partial = out / f"{name}-partial.npz"
        vectors = []
        if partial.exists():
            vectors = list(np.load(partial)["x"])
        started = time.monotonic()
        for i in range(len(vectors), len(samples)):
            y = crop(samples[i])
            vector = (
                phonetics.acoustic(y)
                if name == "acoustic"
                else phonetics.articulation(y, linear)
                if name == "sparc"
                else phonetics.embedding(y, name)
            )
            vectors.append(vector)
            if (i + 1) % 200 == 0:
                np.savez_compressed(partial, x=np.stack(vectors))
                print(
                    name,
                    i + 1,
                    "/",
                    len(samples),
                    "seconds",
                    round(time.monotonic() - started, 1),
                    flush=True,
                )
        x = np.stack(vectors)
        np.savez_compressed(target, x=x, ids=np.array([s["id"] for s in samples]))
        if partial.exists():
            partial.unlink()
        print(name, "complete", x.shape, "seconds", round(time.monotonic() - started, 1), flush=True)
        models.unload()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "acoustic",
            "wavlm-base",
            "wavlm-large",
            "xls-r",
            "qwen-asr",
            "qwen-asr-large",
            "qwen-omni",
            "sparc",
        ],
    )
    extract(parser.parse_args().models)
