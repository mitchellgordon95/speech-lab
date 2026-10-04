"""SPARC's published WavLM layer-9 linear inversion, without its speech synthesizer."""

import pickle

import numpy as np
from huggingface_hub import hf_hub_download
from scipy.signal import butter, filtfilt

from speechlab.audio import load
from speechlab.models import extract_frames

CHANNELS = [
    "Tongue dorsum X",
    "Tongue dorsum Y",
    "Tongue blade X",
    "Tongue blade Y",
    "Tongue tip X",
    "Tongue tip Y",
    "Lower incisor X",
    "Lower incisor Y",
    "Upper lip X",
    "Upper lip Y",
    "Lower lip X",
    "Lower lip Y",
]


class LinearModelUnpickler(pickle.Unpickler):
    """Accept the upstream sklearn/numpy model only; never unpickle user uploads."""

    def find_class(self, module, name):
        allowed = {
            ("sklearn.linear_model._base", "LinearRegression"),
            ("numpy", "ndarray"),
            ("numpy", "dtype"),
            ("numpy.core.multiarray", "_reconstruct"),
            ("numpy._core.multiarray", "_reconstruct"),
            ("numpy.core.multiarray", "scalar"),
            ("numpy._core.multiarray", "scalar"),
        }
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f"Unexpected type in SPARC checkpoint: {module}.{name}")
        return super().find_class(module, name)


def analyze(clip_id, start=0, end=None, progress=lambda _: None):
    y, _ = load(clip_id, start, end, sr=16000)
    # Match upstream: waveform z-score, 10 ms zero pad each side, hidden_states[9], 10 Hz filtering.
    y = (y - y.mean()) / max(float(y.std()), 1e-8)
    frames, backend = extract_frames(
        np.pad(y, (160, 160)), "wavlm-large", layer=9, progress=progress, normalize=False
    )
    if len(frames) < 20:
        raise ValueError("SPARC needs at least 0.4 seconds for its temporal filter.")
    path = hf_hub_download(
        "cheoljun95/Speech-Articulatory-Coding",
        "wavlm_large-9_cut-10_mngu_linear.pkl",
        revision="2e6a07d4b35022d4366a2896a6293fe16aaed78a",
    )
    with open(path, "rb") as f:
        linear = LinearModelUnpickler(f).load()
    b, a = butter(5, 10, fs=50)
    ema = filtfilt(b, a, frames, axis=0) @ linear.coef_.T + linear.intercept_
    return {
        "kind": "articulation",
        "clip_id": clip_id,
        "device": backend,
        "times": (np.arange(len(ema)) / 50 + start).tolist(),
        "channels": CHANNELS,
        "trajectories": ema.round(5).tolist(),
        "note": "Estimated coordinates in SPARC's reference space, not millimeters or measured positions of your mouth. Speech coaching and clinical effectiveness are not established by this display.",
    }
