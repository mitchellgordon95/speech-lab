"""Original synthetic test signals. These are NOT native pronunciation references."""

import numpy as np
from scipy.signal import iirpeak, lfilter

from speechlab import audio, store


def vowel(f1, f2, f0=140, duration=0.9, sr=48000, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(round(sr * duration)) / sr
    source = sum(np.sin(2 * np.pi * f0 * h * t) / h for h in range(1, min(70, int(sr / 2 / f0))))
    y = np.zeros_like(t)
    for freq, bandwidth in ((f1, 80), (f2, 110), (2800, 160)):
        b, a = iirpeak(freq, freq / bandwidth, fs=sr)
        y += lfilter(b, a, source)
    y += rng.normal(0, 0.001, len(t))
    envelope = np.minimum(1, t / 0.03) * np.minimum(1, (duration - t) / 0.05)
    y = y * envelope
    return (y / max(np.max(np.abs(y)), 1e-6) * 0.7).astype(np.float32)


def seed():
    existing = [c for c in store.records("clips") if c.get("source") == "synthetic-fixture-v1"]
    if existing:
        return existing
    clips = []
    for session in ("fixture-day-1", "fixture-day-2"):
        for side, f1, f2 in (("A", 350, 2200), ("B", 700, 1200)):
            for i in range(3):
                y = vowel(f1 + i * 8, f2 + i * 15, f0=125 + i * 20 + (session.endswith("2")) * 10, seed=i)
                clip = audio.add(
                    audio.wav_bytes(y, 48000),
                    {
                        "label": f"Synthetic {side} · {session[-1]}-{i + 1}",
                        "target": "synthetic vowel contrast",
                        "language": "synthetic",
                        "speaker": "synthetic-generator",
                        "session": session,
                        "role": "fixture",
                        "source": "synthetic-fixture-v1",
                        "contrast": side,
                        "notes": "Original generated signal for checking displays and plumbing. Not a human/native reference.",
                        "region_start": 0,
                        "region_end": None,
                    },
                )
                clips.append(clip)
    return clips
