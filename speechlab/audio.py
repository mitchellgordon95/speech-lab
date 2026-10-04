import io
import math
import shutil
import subprocess
import tempfile

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from . import store
from .config import DATA, MAX_SECONDS


def decode(blob: bytes):
    """Decode PCM directly; ffmpeg handles browser WebM/M4A uploads."""
    try:
        y, sr = sf.read(io.BytesIO(blob), dtype="float32", always_2d=True)
    except (sf.LibsndfileError, RuntimeError):
        if not shutil.which("ffmpeg"):
            raise ValueError(
                "This audio format needs ffmpeg. Install it with brew install ffmpeg, or upload WAV."
            )
        with tempfile.TemporaryDirectory(dir=DATA) as folder:
            from pathlib import Path

            src, dst = Path(folder) / "input", Path(folder) / "output.wav"
            src.write_bytes(blob)
            proc = subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-v",
                    "error",
                    "-i",
                    str(src),
                    "-t",
                    str(MAX_SECONDS + 1),
                    "-ac",
                    "1",
                    "-ar",
                    "48000",
                    "-f",
                    "wav",
                    str(dst),
                ],
                capture_output=True,
                timeout=30,
            )
            if proc.returncode:
                raise ValueError("Could not decode the audio. Try a WAV, MP3, M4A, or WebM recording.")
            y, sr = sf.read(dst, dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if not np.isfinite(y).all() or len(y) / sr < 0.15:
        raise ValueError("Record at least 0.15 seconds of valid audio.")
    if len(y) / sr > MAX_SECONDS:
        raise ValueError(f"Keep each clip to {MAX_SECONDS} seconds or less.")
    if np.max(np.abs(y)) < 1e-5:
        raise ValueError("The recording is silent. Check the microphone and try again.")
    return y, sr


def load(ident, start=0, end=None, sr=None):
    y, rate = sf.read(store.path_for("clips", ident, "wav"), dtype="float32")
    duration = len(y) / rate
    end = duration if end is None else float(end)
    if not (0 <= start < end <= duration + 0.001) or end - start < 0.15:
        raise ValueError("Select a region of at least 0.15 seconds within the recording.")
    y = y[round(start * rate) : round(end * rate)]
    if sr and sr != rate:
        g = math.gcd(sr, rate)
        y = resample_poly(y, sr // g, rate // g).astype(np.float32)
        rate = sr
    return y, rate


def wav_bytes(y, sr):
    output = io.BytesIO()
    sf.write(output, y, sr, format="WAV", subtype="PCM_16")
    return output.getvalue()


def add(blob, metadata):
    y, sr = decode(blob)
    ident = store.uid()
    sf.write(store.path_for("clips", ident, "wav"), y, sr, subtype="PCM_16")
    return store.save(
        "clips",
        {
            **metadata,
            "id": ident,
            "duration": round(len(y) / sr, 4),
            "sample_rate": sr,
            "peak": float(np.max(np.abs(y))),
            "clipped_fraction": float(np.mean(np.abs(y) >= 0.999)),
        },
    )
