"""Pre-fitted English contrasts. No user labels or model fitting required."""

import json
from functools import lru_cache

import numpy as np
import soundfile as sf
from scipy.special import expit

from . import audio, models, phonetics
from .config import ROOT

ASSETS = ROOT / "speechlab" / "guided_assets"


@lru_cache
def catalog():
    return json.loads((ASSETS / "catalog.json").read_text())


def contrast(ident):
    found = next((c for c in catalog()["contrasts"] if c["id"] == ident), None)
    if found is None:
        raise ValueError("Unknown sound contrast")
    return found


@lru_cache
def probe(ident):
    contrast(ident)
    return json.loads((ASSETS / f"{ident}.json").read_text())


def margin(model, x):
    x = np.asarray(x)
    filled = np.where(np.isnan(x), np.asarray(model["impute"]), x)
    return ((filled - model["mean"]) / model["scale"]) @ np.asarray(model["coef"]) + model["intercept"]


def represent(y, name):
    if name in models.MODELS:
        return phonetics.embedding(y, name)
    if name == "sparc":
        return phonetics.articulation(y, phonetics.sparc_linear())
    x = phonetics.acoustic(y)
    if name == "formants":
        return x[:3]
    if name == "spectrum":
        return x[3:11]
    if name == "acoustic":
        return x
    raise ValueError("Unsupported guided representation")


def active_windows(y):
    """Select by energy and time only, never by the requested sound or model score."""
    y = np.asarray(y, dtype=np.float32)
    y = y - y.mean()
    if np.mean(np.abs(y) >= 0.99) > 0.01:
        raise ValueError("The microphone is clipping. Move a little farther away and try again.")
    block = 160
    framed = y[: len(y) // block * block].reshape(-1, block)
    levels = np.sqrt(np.mean(framed**2, axis=1))
    peak = float(np.quantile(levels, 0.95))
    if peak < 0.002:
        raise ValueError("I couldn't pick up a clear sound. Move closer to the microphone and try again.")
    active = np.flatnonzero(levels > max(0.001, peak * 0.15))
    runs = np.split(active, np.flatnonzero(np.diff(active) > 3) + 1)
    run = max(runs, key=len)
    start, end = int(run[0] * block), int(min(len(y), (run[-1] + 1) * block))
    if end - start < 3840:
        raise ValueError("Hold just the sound for about a second, then try again.")
    width = 2560
    # Ignore the onset/offset when possible; sample across the central 70%.
    left = max(start, int(start + (end - start) * 0.15 - width / 2))
    right = min(end - width, int(start + (end - start) * 0.85 - width / 2))
    starts = np.unique(np.linspace(left, right, 7).astype(int))
    return [(float(s / phonetics.SR), y[s : s + width]) for s in starts]


def reference(ident, side, index, kind="word"):
    c = contrast(ident)
    if side not in (0, 1) or index not in range(len(c["examples"][side])) or kind not in ("word", "phone"):
        raise ValueError("Unknown speech example")
    item = c["examples"][side][index]
    return ASSETS / "audio" / item[f"{kind}_file"]


def analyze(ident, clip_id=None, example=None, progress=lambda _: None):
    c, fitted = contrast(ident), probe(ident)
    is_example = example is not None
    if is_example:
        y, sr = sf.read(reference(ident, *example, kind="phone"), dtype="float32")
        assert sr == phonetics.SR
        windows = [(0.0, y)]
    else:
        y, _ = audio.load(clip_id, sr=phonetics.SR)
        windows = active_windows(y)
    progress("Listening to the sound…")
    values, measurements = [], []
    for _, wave in windows:
        values.append(float(margin(fitted, represent(wave, fitted["representation"]))))
        measurements.append(phonetics.acoustic(wave))
    values = np.asarray(values)
    measured = np.asarray(measurements)
    acoustics = np.array([np.nanmedian(col) if np.isfinite(col).any() else np.nan for col in measured.T])
    center = float(np.median(values))
    positions = expit(values)
    agreement = float(np.mean((values > 0) == (center > 0)))
    # These are conservative UX checks, not a validated open-set speech detector.
    # NumPy comparisons yield np.bool_; keep flags native for JSON persistence/API responses.
    wrong_kind = bool(
        not is_example
        and ((ident == "s_sh" and acoustics[12] > 0.5) or (ident != "s_sh" and acoustics[12] < 0.2))
    )
    uncertain = abs(center) < 0.6 or agreement < 0.72 or wrong_kind
    side = int(center > 0)
    label = c["labels"][side]
    if wrong_kind:
        title = "Try holding just the sound"
        detail = c["instruction"]
    elif uncertain:
        title = "Between the two, or changing"
        detail = "The short slices don't give a clear, consistent match. Try one steady sound."
    else:
        title = f"Closer to {label}"
        detail = f"Your sound falls toward the {label} side of this English contrast."
    observation = None
    if ident == "s_sh":
        observation = {
            "label": "Center of the hiss",
            "value": round(float(acoustics[3]) / 1000, 2),
            "unit": "kHz",
            "description": "S examples usually concentrate their hiss at higher frequencies than SH examples.",
            "reference_medians": c["spectral_medians_khz"],
        }
    return {
        "kind": "guided",
        "contrast": ident,
        "clip_id": clip_id,
        "example": list(example) if is_example else None,
        "title": title,
        "detail": detail,
        "uncertain": uncertain,
        "position": None if wrong_kind else float(expit(center)),
        "windows": [
            {
                "start": round(start, 3),
                "end": round(start + len(wave) / phonetics.SR, 3),
                "position": float(p),
            }
            for (start, wave), p in zip(windows, positions)
        ],
        "observation": observation,
        "note": "Position on a learned sound contrast, not a pronunciation grade or a tongue-position measurement.",
    }
