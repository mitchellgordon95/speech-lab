"""Freeze development-set choices before evaluating final test speakers."""

import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from speechlab import models

from .prepare import BASE, PROTOCOL

OUT = Path(__file__).resolve().parent / "results"


def pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000),
    )


def representations():
    manifest = json.loads((BASE / "manifest.json").read_text())
    ids = np.array([s["id"] for s in manifest])

    def read(name):
        with np.load(BASE / "features" / f"{name}.npz") as saved:
            if not np.array_equal(saved["ids"], ids):
                raise ValueError(f"Feature/label order mismatch: {name}")
            return saved["x"]

    acoustic = read("acoustic")
    reps = {
        "formants": acoustic[:, :3],
        "spectrum": acoustic[:, 3:11],
        "acoustic": acoustic,
        "timing_pitch_level": acoustic[:, 11:],
    }
    for name in list(models.MODELS) + ["sparc"]:
        path = BASE / "features" / f"{name}.npz"
        if not path.exists():
            raise ValueError(f"Features incomplete: {name}")
        reps[name] = read(name)
    return reps


def metrics(y, margin, speakers, bootstrap=False):
    pred = (margin > 0).astype(int)
    result = {
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, margin)),
        "n": len(y),
        "speakers": len(set(speakers)),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }
    if bootstrap:
        rng = np.random.default_rng(PROTOCOL["seed"])
        unique = sorted(set(speakers))
        # Bootstrap complete speakers rather than treating their tokens as independent.
        stats = []
        for _ in range(500):
            ids = np.concatenate(
                [np.flatnonzero(speakers == s) for s in rng.choice(unique, len(unique), replace=True)]
            )
            if len(set(y[ids])) == 2:
                stats.append(balanced_accuracy_score(y[ids], pred[ids]))
        result["speaker_bootstrap_95_ci"] = np.quantile(stats, [0.025, 0.975]).tolist()
    return result


def centroid_margin(x, y, q):
    centers = np.stack([x[y == v].mean(0) for v in (0, 1)])
    distances = cdist(q, centers, metric="cosine")
    return np.nan_to_num(distances[:, 0] - distances[:, 1])


def personalization(x, y, metadata):
    scores = []
    for speaker in sorted({m["speaker"] for m in metadata}):
        indices = np.array([i for i, m in enumerate(metadata) if m["speaker"] == speaker])
        side0 = [i for i in indices if y[i] == 0]
        side1 = [i for i in indices if y[i] == 1]
        if min(len(side0), len(side1)) < 5:
            continue
        # Two accepted anchors per class; query clips are separate utterances as well as tokens.
        anchor = side0[:2] + side1[:2]
        anchor_utts = {metadata[i]["utterance"] for i in anchor}
        query = np.array(
            [i for i in indices if i not in anchor and metadata[i]["utterance"] not in anchor_utts]
        )
        if not len(query) or len(set(y[query])) < 2:
            continue
        other = np.array([i for i, m in enumerate(metadata) if m["speaker"] != speaker])
        pm = centroid_margin(x[anchor], y[anchor], x[query])
        gm = centroid_margin(x[other], y[other], x[query])
        scores.append(
            {
                "speaker": speaker,
                "n": len(query),
                "personal": float(balanced_accuracy_score(y[query], pm > 0)),
                "other_speakers": float(balanced_accuracy_score(y[query], gm > 0)),
            }
        )
    return {
        "speakers": len(scores),
        "queries": sum(s["n"] for s in scores),
        "personal_mean_balanced_accuracy": float(np.mean([s["personal"] for s in scores]))
        if scores
        else None,
        "other_speakers_mean_balanced_accuracy": float(np.mean([s["other_speakers"] for s in scores]))
        if scores
        else None,
    }


def triplets(x, y, speakers):
    rng = np.random.default_rng(PROTOCOL["seed"])
    correct = []
    for a in rng.integers(0, len(y), 1000):
        pos = np.flatnonzero((y == y[a]) & (speakers != speakers[a]))
        neg = np.flatnonzero((y != y[a]) & (speakers == speakers[a]))
        if not len(pos) or not len(neg):
            continue
        p, n = rng.choice(pos), rng.choice(neg)
        distance = cdist(x[a : a + 1], x[[p, n]], metric="cosine")[0]
        correct.append(float(distance[0] < distance[1]))
    return {"n": len(correct), "same_phone_other_speaker_closer": float(np.mean(correct))}


def export_probe(probe, name, contrast, phones, report):
    imputer, scaler, clf = [step[1] for step in probe.steps]
    return {
        "version": 1,
        "representation": name,
        "contrast": contrast,
        "phones": phones,
        "impute": imputer.statistics_.tolist(),
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coef": clf.coef_[0].tolist(),
        "intercept": float(clf.intercept_[0]),
        "test": report,
        "preprocessing": "phonetics.padded_phone v1; four-bin means for neural encoders",
    }


def evaluate():
    OUT.mkdir(exist_ok=True)
    samples = json.loads((BASE / "manifest.json").read_text())
    reps = representations()
    labels = np.array([s["phone"] for s in samples])
    split = np.array([s["split"] for s in samples])
    speakers = np.array([s["speaker"] for s in samples])
    selection_path = OUT / "selection.json"
    development = {}
    selected = {}
    for contrast, phones in PROTOCOL["contrasts"].items():
        rows = np.isin(labels, phones)
        tr = np.flatnonzero(rows & (split == "train"))
        va = np.flatnonzero(rows & (split == "validation"))
        y = (labels == phones[1]).astype(int)
        scores = {}
        for name, x in reps.items():
            probe = pipeline().fit(x[tr], y[tr])
            scores[name] = metrics(y[va], probe.decision_function(x[va]), speakers[va])
        # The diagnostic timing/pitch/level baseline is not a candidate coaching feature.
        candidates = [n for n in scores if n != "timing_pitch_level"]
        winner = max(candidates, key=lambda n: (scores[n]["balanced_accuracy"], scores[n]["roc_auc"], n))
        selected[contrast] = {"representation": winner, "validation": scores[winner], "phones": phones}
        development[contrast] = scores
        print("Development", contrast, winner, round(scores[winner]["balanced_accuracy"], 3), flush=True)
    selection = {
        "protocol": PROTOCOL,
        "selected": selected,
        "development": development,
        "note": "Chosen using validation speakers only; written before test evaluation.",
    }
    if selection_path.exists():
        old = json.loads(selection_path.read_text())
        assert old["selected"] == selected, (
            "Development selection changed; do not silently revise the test protocol."
        )
    else:
        selection_path.write_text(json.dumps(selection, indent=2))
    reports = {}
    probe_dir = BASE / "probes"
    probe_dir.mkdir(exist_ok=True)
    for contrast, phones in PROTOCOL["contrasts"].items():
        rows = np.isin(labels, phones)
        fit = np.flatnonzero(rows & (split != "test"))
        te = np.flatnonzero(rows & (split == "test"))
        y = (labels == phones[1]).astype(int)
        train_words = {samples[i]["word"] for i in fit}
        unseen = np.array([j for j, i in enumerate(te) if samples[i]["word"] not in train_words])
        testmeta = [samples[i] for i in te]
        scores = {}
        for name, x in reps.items():
            probe = pipeline().fit(x[fit], y[fit])
            margin = probe.decision_function(x[te])
            scores[name] = metrics(y[te], margin, speakers[te], bootstrap=True)
            if len(unseen) and len(set(y[te][unseen])) == 2:
                scores[name]["unseen_words"] = metrics(y[te][unseen], margin[unseen], speakers[te][unseen])
            for sex in ("F", "M"):
                ids = np.array([j for j, i in enumerate(te) if samples[i]["sex"] == sex])
                if len(ids) and len(set(y[te][ids])) == 2:
                    scores[name][sex] = metrics(y[te][ids], margin[ids], speakers[te][ids])
            base = probe.steps[0][1].transform(x)
            if name not in models.MODELS and name != "sparc":
                base = probe.steps[1][1].transform(base)
            scores[name]["centroid"] = metrics(
                y[te], centroid_margin(base[fit], y[fit], base[te]), speakers[te]
            )
            scores[name]["triplets"] = triplets(base[te], y[te], speakers[te])
            scores[name]["personalization"] = personalization(base[te], y[te], testmeta)
            if name == selected[contrast]["representation"]:
                selected_report = scores[name]
                model = export_probe(probe, name, contrast, phones, selected_report)
                (probe_dir / f"{contrast}.json").write_text(json.dumps(model))
                np.savez_compressed(
                    BASE / f"predictions-{contrast}.npz",
                    ids=np.array([samples[i]["id"] for i in te]),
                    margin=margin,
                    labels=y[te],
                )
        best = selected[contrast]
        result = scores[best["representation"]]
        promoted = (
            best["validation"]["balanced_accuracy"] >= 0.85
            and result["balanced_accuracy"] >= 0.85
            and result["speaker_bootstrap_95_ci"][0] >= 0.8
        )
        reports[contrast] = {
            "phones": phones,
            "selected_representation": best["representation"],
            "promoted": promoted,
            "representations": scores,
        }
        print(
            "Final",
            contrast,
            best["representation"],
            round(result["balanced_accuracy"], 3),
            "promoted",
            promoted,
            flush=True,
        )
    report = {
        "protocol": PROTOCOL,
        "data": json.loads((BASE / "data-summary.json").read_text()),
        "contrasts": reports,
        "limitations": [
            "Automatic forced-alignment labels; not human-audited phonetic ground truth.",
            "Held-out for fitted probes, not a guarantee these recordings were unseen in encoder pretraining.",
            "Correct read speech supports phone discrimination, not detecting arbitrary mispronunciation or proving learner benefit.",
        ],
    }
    (OUT / "benchmark.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        evaluate()
