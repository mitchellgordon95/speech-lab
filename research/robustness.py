"""Stress the frozen winning probes; do not select a replacement using test results."""

import json
from collections import defaultdict

import numpy as np

from speechlab import models, phonetics

from .evaluate import OUT, metrics, representations
from .features import crop
from .prepare import BASE, stable


def predict(probe, x):
    x = np.asarray(x)
    filled = np.where(np.isnan(x), np.array(probe["impute"]), x)
    return ((filled - np.array(probe["mean"])) / np.array(probe["scale"])) @ np.array(probe["coef"]) + probe[
        "intercept"
    ]


def represent(y, name):
    if name in models.MODELS:
        return phonetics.embedding(y, name)
    x = phonetics.acoustic(y)
    if name == "formants":
        return x[:3]
    if name == "spectrum":
        return x[3:11]
    if name == "acoustic":
        return x
    if name == "sparc":
        return phonetics.articulation(y, phonetics.sparc_linear())
    raise ValueError(name)


def run():
    report = json.loads((OUT / "benchmark.json").read_text())
    samples = json.loads((BASE / "manifest.json").read_text())
    reps = representations()
    grouped = defaultdict(list)
    for contrast, r in report["contrasts"].items():
        grouped[r["selected_representation"]].append(contrast)
    results = {}
    for name, contrasts in grouped.items():
        for contrast in contrasts:
            r = report["contrasts"][contrast]
            probe = json.loads((BASE / "probes" / f"{contrast}.json").read_text())
            chosen = []
            for phone in r["phones"]:
                candidates = [
                    i for i, s in enumerate(samples) if s["phone"] == phone and s["split"] == "test"
                ]
                chosen.extend(sorted(candidates, key=lambda i: stable(samples[i]["id"]))[:80])
            y = np.array([int(samples[i]["phone"] == r["phones"][1]) for i in chosen])
            groups = np.array([samples[i]["speaker"] for i in chosen])
            original = predict(probe, reps[name][chosen])
            scores = {"baseline_subset": metrics(y, original, groups)}
            for condition in (
                "gain_quarter",
                "noise_20db",
                "boundary_minus_20ms",
                "boundary_plus_20ms",
                "central_160ms",
            ):
                margins = []
                for j, i in enumerate(chosen):
                    token = samples[i]
                    shift = (
                        -0.02
                        if condition == "boundary_minus_20ms"
                        else 0.02
                        if condition == "boundary_plus_20ms"
                        else 0
                    )
                    wave = crop(token, shift)
                    if condition == "central_160ms":
                        length = min(len(wave), 2560)
                        start = (len(wave) - length) // 2
                        wave = wave[start : start + length]
                    elif condition == "gain_quarter":
                        wave *= 0.25
                    elif condition == "noise_20db":
                        rng = np.random.default_rng(int(stable(token["id"])[:8], 16))
                        wave += rng.normal(0, np.sqrt(np.mean(wave**2)) * 0.1, len(wave)).astype(np.float32)
                    margins.append(float(predict(probe, represent(wave, name))))
                margins = np.array(margins)
                scores[condition] = {
                    **metrics(y, margins, groups),
                    "prediction_agreement": float(np.mean((margins > 0) == (original > 0))),
                    "mean_absolute_margin_change": float(np.mean(np.abs(margins - original))),
                }
                print(contrast, name, condition, round(scores[condition]["balanced_accuracy"], 3), flush=True)
            robust = all(
                scores[c]["balanced_accuracy"] >= 0.8
                for c in ("noise_20db", "boundary_minus_20ms", "boundary_plus_20ms", "central_160ms")
            )
            results[contrast] = {
                "representation": name,
                "conditions": scores,
                "robustness_pass": robust,
                "demo_eligible": r["promoted"] and robust,
            }
            (OUT / "robustness.json").write_text(json.dumps(results, indent=2))
        models.unload()


if __name__ == "__main__":
    run()
