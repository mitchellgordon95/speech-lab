"""Measured acoustic cues. No automatic tongue-position or VOT claims."""

import numpy as np
import parselmouth
from scipy import signal

from speechlab.audio import load

FEATURE_NAMES = [
    "F1 (Hz)",
    "F2 (Hz)",
    "F3 (Hz)",
    "Spectral center (Hz)",
    "Spectral spread (Hz)",
    "Spectral skewness",
    "Voiced fraction",
    "Duration (s)",
]


def analyze(clip_id, start=0, end=None, ceiling=5500):
    y, sr = load(clip_id, start, end)
    sound = parselmouth.Sound(y, sampling_frequency=sr)
    pitch = sound.to_pitch_ac(time_step=0.01, pitch_floor=65, pitch_ceiling=600)
    f0 = pitch.selected_array["frequency"]
    formant = sound.to_formant_burg(
        time_step=0.01, max_number_of_formants=5, maximum_formant=ceiling, window_length=0.025
    )
    times = pitch.xs()
    tracks = []
    for t, hz in zip(times, f0):
        row = [formant.get_value_at_time(i, float(t)) if hz else float("nan") for i in (1, 2, 3)]
        tracks.append([float(v) if np.isfinite(v) else None for v in row])
    valid = np.array([[np.nan if v is None else v for v in r] for r in tracks])
    medians = [float(np.nanmedian(valid[:, i])) if np.isfinite(valid[:, i]).any() else None for i in range(3)]
    frequencies, power = signal.welch(y, sr, nperseg=min(len(y), round(sr * 0.025)))
    # Frication moments exclude DC/low-frequency voicing. Select a consonant region for this cue.
    mask = (frequencies >= 1000) & (frequencies <= min(12000, sr / 2))
    f, p = frequencies[mask], power[mask]
    p = p / max(float(p.sum()), 1e-20)
    center = float(np.sum(f * p))
    spread = float(np.sqrt(np.sum((f - center) ** 2 * p)))
    skew = float(np.sum((f - center) ** 3 * p) / max(spread**3, 1))
    freqs, st, spec = signal.spectrogram(
        y, sr, nperseg=round(0.025 * sr), noverlap=round(0.015 * sr), nfft=max(2048, round(0.025 * sr))
    )
    max_bin = np.searchsorted(freqs, min(8000, sr / 2))
    spec = 10 * np.log10(np.maximum(spec[:max_bin], 1e-14))
    spec -= spec.max()
    step = max(1, len(st) // 500)
    stride = max(1, len(y) // 1600)
    warnings = []
    if np.mean(f0 > 0) < 0.15:
        warnings.append(
            "Little voiced audio: vowel formants may be unavailable. Frication measurements can still be useful."
        )
    if np.max(np.abs(y)) >= 0.999:
        warnings.append("Clipping detected; lower the recording level.")
    return {
        "kind": "acoustics",
        "clip_id": clip_id,
        "start": start,
        "end": start + len(y) / sr,
        "times": (times + start).round(4).tolist(),
        "pitch": [float(v) if v else None for v in f0],
        "formants": tracks,
        "formant_medians": medians,
        "ceiling": ceiling,
        "waveform": y[::stride].round(5).tolist(),
        "wave_times": (np.arange(0, len(y), stride) / sr + start).tolist(),
        "spectrum": {"hz": frequencies.tolist(), "db": (10 * np.log10(power + 1e-14)).tolist()},
        "spectrogram": {
            "times": (st[::step] + start).tolist(),
            "hz": freqs[:max_bin].tolist(),
            "db": np.maximum(spec[:, ::step], -80).round(1).tolist(),
        },
        "feature_names": FEATURE_NAMES,
        "features": medians + [center, spread, skew, float(np.mean(f0 > 0)), len(y) / sr],
        "warnings": warnings,
        "note": "Formants are estimates on voiced frames. Spectral moments describe the selected region; they do not diagnose articulation. Mark burst and voicing onset manually to measure VOT.",
    }
