"""Shared phone-level transforms used by the corpus benchmark and guided demos."""

from functools import lru_cache

import numpy as np
import parselmouth
from scipy import signal

from . import models

SR = 16000
ACOUSTIC_NAMES = [
    "F1",
    "F2",
    "F3",
    "spectral_center",
    "spectral_spread",
    "spectral_skew",
    "spectral_flatness",
    "low_band_fraction",
    "mid_band_fraction",
    "high_band_fraction",
    "zero_crossing_rate",
    "pitch",
    "voiced_fraction",
    "duration",
    "log_rms",
]


def padded_phone(y):
    """Keep the phone's time scale; normalize level and center it in 0.5 s of silence."""
    y = np.asarray(y, dtype=np.float32).copy()
    if not 0.05 <= len(y) / SR <= 0.45:
        raise ValueError("Phone region must be between 50 and 450 ms")
    y -= y.mean()
    y *= 0.1 / max(float(np.sqrt(np.mean(y * y))), 1e-6)
    fade = min(80, len(y) // 8)
    y[:fade] *= np.linspace(0, 1, fade)
    y[-fade:] *= np.linspace(1, 0, fade)
    left = (8000 - len(y)) // 2
    return np.pad(y, (left, 8000 - len(y) - left))


def acoustic(y):
    y = np.asarray(y, dtype=np.float64)
    duration, rms = len(y) / SR, np.sqrt(np.mean(y * y))
    sound = parselmouth.Sound(y, SR)
    pitch = sound.to_pitch_ac(time_step=0.01, pitch_floor=65, pitch_ceiling=600)
    f0 = pitch.selected_array["frequency"]
    formant = sound.to_formant_burg(time_step=0.01, maximum_formant=5500, window_length=0.02)
    values = []
    for i in (1, 2, 3):
        v = [formant.get_value_at_time(i, float(t)) for t, hz in zip(pitch.xs(), f0) if hz]
        v = [x for x in v if np.isfinite(x)]
        values.append(float(np.median(v)) if v else np.nan)
    f, p = signal.welch(y, SR, nperseg=min(len(y), 400), nfft=512)
    f, p = f[f >= 1000], p[f >= 1000]
    p = p / max(p.sum(), 1e-20)
    center = np.sum(f * p)
    spread = np.sqrt(np.sum((f - center) ** 2 * p))
    values.extend(
        [
            center,
            spread,
            np.sum((f - center) ** 3 * p) / max(spread**3, 1),
            np.exp(np.mean(np.log(p + 1e-12))) / max(np.mean(p), 1e-12),
            p[f < 3000].sum(),
            p[(f >= 3000) & (f < 5000)].sum(),
            p[f >= 5000].sum(),
            np.mean(np.diff(np.signbit(y)) != 0),
            np.median(f0[f0 > 0]) if np.any(f0 > 0) else np.nan,
            np.mean(f0 > 0),
            duration,
            np.log(max(rms, 1e-8)),
        ]
    )
    return np.array(values, dtype=np.float32)


def pool(frames):
    """Same four-bin temporal pooling as the original workbench."""
    return np.concatenate([b.mean(0) for b in np.array_split(frames, 4)]).astype(np.float32)


def embedding(y, name):
    frames, _ = models.extract_frames(padded_phone(y), name)
    return pool(frames)


def articulation(y, linear):
    from scipy.signal import butter, filtfilt

    wave = padded_phone(y)
    wave = (wave - wave.mean()) / max(float(wave.std()), 1e-8)
    frames, _ = models.extract_frames(np.pad(wave, (160, 160)), "wavlm-large", layer=9, normalize=False)
    b, a = butter(5, 10, fs=50)
    trajectories = filtfilt(b, a, frames, axis=0) @ linear.coef_.T + linear.intercept_
    return pool(trajectories)


@lru_cache
def sparc_linear():
    import warnings

    from huggingface_hub import hf_hub_download
    from sklearn.exceptions import InconsistentVersionWarning

    from demos.articulation import LinearModelUnpickler

    path = hf_hub_download(
        "cheoljun95/Speech-Articulatory-Coding",
        "wavlm_large-9_cut-10_mngu_linear.pkl",
        revision="2e6a07d4b35022d4366a2896a6293fe16aaed78a",
    )
    with warnings.catch_warnings(), open(path, "rb") as f:
        warnings.simplefilter("ignore", InconsistentVersionWarning)
        return LinearModelUnpickler(f).load()
