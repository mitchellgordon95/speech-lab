"""Fixed reference maps for low-latency speech windows; microphone audio is not saved."""

import json
import time
from functools import lru_cache

import numpy as np

from . import phonetics
from .config import ROOT

ASSETS = ROOT / "speechlab" / "live_assets"
WINDOW_SAMPLES = 2560


@lru_cache
def catalog():
    return json.loads((ASSETS / "catalog.json").read_text())


def get_map(ident):
    found = next((m for m in catalog()["maps"] if m["id"] == ident), None)
    if found is None:
        raise ValueError("Unknown sound map")
    return found


@lru_cache
def projection(ident):
    get_map(ident)
    p = json.loads((ASSETS / f"{ident}.json").read_text())
    return {
        key: np.asarray(value) if key in ("mean", "scale", "basis", "shift", "center", "rotation") else value
        for key, value in p.items()
    }


def coordinates(vector, p):
    xy = ((np.asarray(vector) - p["mean"]) / p["scale"]) @ p["basis"] - p["shift"]
    return (xy - p["center"]) @ p["rotation"] / p["radius"]


def frame(ident, wave):
    started = time.perf_counter()
    m = get_map(ident)
    wave = np.asarray(wave, dtype=np.float32)
    if wave.shape != (WINDOW_SAMPLES,) or not np.isfinite(wave).all():
        raise ValueError("Send exactly 160 ms of finite 16 kHz mono audio")
    if np.max(np.abs(wave)) > 1.01:
        raise ValueError("Audio samples must be between -1 and 1")
    if np.mean(np.abs(wave) > 0.99) > 0.01:
        return {"active": False, "reason": "clipping"}
    if np.sqrt(np.mean((wave - wave.mean()) ** 2)) < 0.002:
        return {"active": False, "reason": "quiet"}
    p = projection(ident)
    vector = phonetics.embedding(wave, m["model"])
    point = coordinates(vector, p)
    if not np.isfinite(point).all():
        raise ValueError("Could not locate this sound on the map")
    return {
        "active": True,
        "x": float(point[0]),
        "y": float(point[1]),
        "processing_ms": round((time.perf_counter() - started) * 1000, 2),
    }


def reference(ident, phone, index):
    m = get_map(ident)
    category = next((c for c in m["categories"] if c["phone"] == phone), None)
    if category is None or index not in range(len(category["examples"])):
        raise ValueError("Unknown reference recording")
    return ASSETS / "audio" / category["examples"][index]["file"]
