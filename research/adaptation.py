"""Exploratory matched-size control, added after the main test without changing selection.

The original two-personal-anchor vs many-population-anchor comparison confounds
speaker matching with reference-set size. This control uses exactly two anchors
per category on both sides, paired queries, and repeated random anchor draws.
"""

import json

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from speechlab import models

from .evaluate import OUT, centroid_margin, representations
from .prepare import BASE, PROTOCOL


def run():
    report = json.loads((OUT / "benchmark.json").read_text())
    samples = json.loads((BASE / "manifest.json").read_text())
    reps = representations()
    results = {}
    for contrast, r in report["contrasts"].items():
        name = r["selected_representation"]
        chosen = [i for i, s in enumerate(samples) if s["phone"] in r["phones"]]
        x = reps[name][chosen]
        meta = [samples[i] for i in chosen]
        dev = np.array([s["split"] != "test" for s in meta])
        x = SimpleImputer(keep_empty_features=True).fit(x[dev]).transform(x)
        if name not in models.MODELS and name != "sparc":
            x = StandardScaler().fit(x[dev]).transform(x)
        x = x[~dev]
        meta = [s for s, keep in zip(meta, ~dev) if keep]
        y = np.array([int(s["phone"] == r["phones"][1]) for s in meta])
        speakers = np.array([s["speaker"] for s in meta])
        utterances = np.array([s["utterance"] for s in meta])
        rng = np.random.default_rng(PROTOCOL["seed"])
        per_speaker = []
        for speaker in sorted(set(speakers)):
            own = [np.flatnonzero((speakers == speaker) & (y == side)) for side in (0, 1)]
            other = [np.flatnonzero((speakers != speaker) & (y == side)) for side in (0, 1)]
            if min(map(len, own)) < 5:
                continue
            trials = []
            for _ in range(30):
                personal = np.concatenate([rng.choice(ids, 2, replace=False) for ids in own])
                population = np.concatenate([rng.choice(ids, 2, replace=False) for ids in other])
                query = np.flatnonzero((speakers == speaker) & ~np.isin(utterances, utterances[personal]))
                if not len(query) or len(set(y[query])) < 2:
                    continue
                scores = [
                    balanced_accuracy_score(y[query], centroid_margin(x[ids], y[ids], x[query]) > 0)
                    for ids in (personal, population)
                ]
                trials.append(scores)
            if trials:
                per_speaker.append(np.mean(trials, axis=0))
        values = np.asarray(per_speaker)
        delta = values[:, 0] - values[:, 1]
        draws = rng.integers(0, len(values), size=(1000, len(values)))
        results[contrast] = {
            "representation": name,
            "speakers": len(values),
            "personal_mean_balanced_accuracy": float(values[:, 0].mean()),
            "equal_size_other_speakers_mean_balanced_accuracy": float(values[:, 1].mean()),
            "paired_personal_minus_other": float(delta.mean()),
            "paired_speaker_bootstrap_95_ci": np.quantile(delta[draws].mean(axis=1), [0.025, 0.975]).tolist(),
        }
        print(contrast, results[contrast], flush=True)
    (OUT / "adaptation-control.json").write_text(
        json.dumps(
            {
                "status": "Exploratory control added after main test results; no effect on model selection or demo gates.",
                "method": "30 random draws per eligible test speaker; two known-correct anchors per category on each side; paired same-speaker queries from utterances distinct from personal anchors. Other-speaker anchors are drawn from the remaining test speakers. Mean within speaker, then across speakers; 1,000 paired speaker bootstraps.",
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        run()
