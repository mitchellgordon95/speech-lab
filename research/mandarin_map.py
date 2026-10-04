"""Fit fixed two-dimensional Mandarin maps; choose encoders before opening test scores."""

import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import Ridge
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

from speechlab import live, models, phonetics

from .mandarin_features import crop, utterance
from .mandarin_prepare import BASE, PROTOCOL, stable

RESULTS = Path(__file__).parent / "results"
LABELS = {"a": "a", "ɤ": "e", "i": "i", "o": "o", "u": "u", "y": "ü", "s": "s", "ʂ": "sh", "ɕ": "x"}
COLORS = ["#5a7bba", "#9b74a5", "#498e7a", "#cc8560", "#778c3e", "#b26683", "#477f9d", "#8e7543", "#8063a6"]


def target_coordinates(phones):
    order = PROTOCOL["shared_targets"]["order"]
    angles = np.array([order.index(p) for p in phones]) * 2 * np.pi / len(order)
    return np.column_stack([np.cos(angles), np.sin(angles)])


def fit_projection(x, y, method="lda", alpha=100):
    x = np.asarray(x, dtype=np.float64)
    scaler = StandardScaler().fit(x)
    z = scaler.transform(x)
    pca = PCA(
        n_components=min(96, x.shape[1], len(x) - len(set(y))),
        svd_solver="randomized",
        random_state=PROTOCOL["seed"],
    ).fit(z)
    reduced = pca.transform(z)
    if method == "lda":
        lda = LinearDiscriminantAnalysis(
            n_components=2, solver="eigen", shrinkage="auto", priors=np.ones(len(set(y))) / len(set(y))
        ).fit(reduced, y)
        basis = pca.components_.T @ lda.scalings_[:, :2]
        shift = pca.mean_ @ basis
        expected = lda.transform(reduced)
    else:
        weights = np.array([1 / np.mean(y == label) for label in y])
        weights /= weights.mean()
        ridge = Ridge(alpha=alpha).fit(reduced, target_coordinates(y), sample_weight=weights)
        basis = pca.components_.T @ ridge.coef_.T
        shift = pca.mean_ @ basis - ridge.intercept_
        expected = ridge.predict(reduced)
    p = dict(
        mean=scaler.mean_,
        scale=scaler.scale_,
        basis=basis,
        shift=shift,
        center=np.zeros(2),
        rotation=np.eye(2),
        radius=1.0,
    )
    p.update(method=method, alpha=alpha if method == "ridge" else None)
    np.testing.assert_allclose(live.coordinates(x, p), expected, atol=1e-8)
    return p


def centers(points, labels, phones):
    return np.array([points[labels == phone].mean(axis=0) for phone in phones])


def classify(points, means, phones):
    return np.array(phones)[np.square(points[:, None] - means).sum(axis=2).argmin(axis=1)]


def score(labels, predicted, speakers, phones):
    matrix = confusion_matrix(labels, predicted, labels=phones)
    unique = sorted(set(speakers))
    rng = np.random.default_rng(PROTOCOL["seed"])
    trials = []
    for _ in range(1000):
        indices = np.concatenate([np.flatnonzero(speakers == s) for s in rng.choice(unique, len(unique))])
        trials.append(balanced_accuracy_score(labels[indices], predicted[indices]))
    return {
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "speaker_bootstrap_95": np.percentile(trials, [2.5, 97.5]).tolist(),
        "recall": dict(zip(phones, (matrix.diagonal() / matrix.sum(axis=1)).tolist())),
        "confusion": matrix.tolist(),
        "phone_order": phones,
        "tokens": len(labels),
        "speakers": len(unique),
    }


def serializable(p):
    return {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in p.items()}


def main():
    samples = json.loads((BASE / "manifest.json").read_text())
    labels = np.array([s["phone"] for s in samples])
    splits = np.array([s["split"] for s in samples])
    speakers = np.array([s["speaker"] for s in samples])
    ids = np.array([s["id"] for s in samples])
    by_model = {}
    for model in PROTOCOL["representations"]:
        saved = np.load(BASE / "features" / f"{model}.npz")
        np.testing.assert_array_equal(ids, saved["ids"])
        by_model[model] = saved["x"]
    selection = {}
    for ident, phones in PROTOCOL["maps"].items():
        allowed = np.isin(labels, phones)
        train, val = allowed & (splits == "train"), allowed & (splits == "validation")
        scores = {}
        candidates = {}
        for model, x in by_model.items():
            methods = [("lda", 100)] + (
                [("ridge", a) for a in [10, 100, 1000]] if ident == "mandarin" else []
            )
            for method, alpha in methods:
                p = fit_projection(x[train], labels[train], method, alpha)
                means = centers(live.coordinates(x[train], p), labels[train], phones)
                predicted = classify(live.coordinates(x[val], p), means, phones)
                key = f"{model}/{method}/{alpha}"
                scores[key] = float(balanced_accuracy_score(labels[val], predicted))
                candidates[key] = {"model": model, "method": method, "alpha": alpha}
                print(ident, key, "validation", scores[key], flush=True)
        selection[ident] = {"validation": scores, **candidates[max(scores, key=scores.get)]}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "mandarin-selection.json").write_text(json.dumps(selection, indent=2))
    # Selection is persisted before computing any final held-out speaker metrics.
    live.ASSETS.mkdir(exist_ok=True)
    (live.ASSETS / "audio").mkdir(exist_ok=True)
    catalog = {
        "maps": [],
        "source": PROTOCOL["source"],
        "alignment_note": PROTOCOL["alignment_note"],
        "window_ms": 160,
        "hop_ms": 80,
    }
    report = {
        "protocol": PROTOCOL,
        "corpus": json.loads((BASE / "summary.json").read_text()),
        "selection": selection,
        "maps": {},
    }
    for ident, phones in PROTOCOL["maps"].items():
        if ident != "mandarin":
            continue
        model = selection[ident]["model"]
        x = by_model[model]
        allowed = np.isin(labels, phones)
        dev, test = allowed & (splits != "test"), allowed & (splits == "test")
        p = fit_projection(x[dev], labels[dev], selection[ident]["method"], selection[ident]["alpha"])
        means = centers(live.coordinates(x[dev], p), labels[dev], phones)
        p["center"] = means.mean(axis=0)
        target = target_coordinates(phones)
        p["rotation"] = orthogonal_procrustes(means - p["center"], target - target.mean(axis=0))[0]
        p["radius"] = float(np.sqrt(np.mean(np.square(means - p["center"]).sum(axis=1))))
        points = live.coordinates(x, p)
        means = centers(points[dev], labels[dev], phones)
        prediction = classify(points[test], means, phones)
        metrics = score(labels[test], prediction, speakers[test], phones)
        # Test the same exact 2D map on boundary shifts, without retraining it.
        shifted_indices = []
        for phone in phones:
            eligible = np.flatnonzero(test & (labels == phone))
            shifted_indices.extend(sorted(eligible, key=lambda i: stable(samples[i]["id"]))[:40])
        shifted_indices = np.array(shifted_indices)
        shift_scores = {}
        for shift in [-0.02, 0, 0.02]:
            if shift == 0:
                values = points[shifted_indices]
            else:
                values = live.coordinates(
                    np.stack([phonetics.embedding(crop(samples[i], shift), model) for i in shifted_indices]),
                    p,
                )
            guess = classify(values, means, phones)
            shift_scores[str(round(shift * 1000))] = float(
                balanced_accuracy_score(labels[shifted_indices], guess)
            )
        models.unload()
        metrics["boundary_shift_balanced_accuracy"] = shift_scores
        metrics["boundary_shift_tokens"] = len(shifted_indices)
        m = {
            "id": ident,
            "name": "Mandarin sounds",
            "model": model,
            "test": metrics,
            "categories": [],
        }
        for n, phone in enumerate(phones):
            indices = np.flatnonzero(dev & (labels == phone))
            cloud = points[indices]
            cov = np.cov(cloud.T)
            category = {
                "phone": phone,
                "label": LABELS[phone],
                "kind": "vowel" if phone in PROTOCOL["maps"]["vowels"] else "consonant",
                "color": COLORS[n],
                "center": means[n].tolist(),
                "covariance": cov.tolist(),
                "points": points[sorted(indices, key=lambda i: stable(samples[i]["id"]))[:50]].tolist(),
                "examples": [],
            }
            used_speakers, used_words = set(), set()
            for i in sorted(indices, key=lambda i: stable(samples[i]["id"])):
                row = samples[i]
                if row["speaker"] in used_speakers or row["word"] in used_words:
                    continue
                used_speakers.add(row["speaker"])
                used_words.add(row["word"])
                wave = utterance(row["utterance"])
                a = max(0, round((row["word_start"] - 0.025) * 16000))
                b = min(len(wave), round((row["word_end"] + 0.025) * 16000))
                file = f"{ident}-{n}-{len(category['examples'])}.flac"
                sf.write(live.ASSETS / "audio" / file, wave[a:b], 16000, subtype="PCM_16")
                category["examples"].append(
                    {
                        **row,
                        "file": file,
                        "point": points[i].tolist(),
                        "excerpt_start": a / 16000,
                        "excerpt_end": b / 16000,
                    }
                )
                if len(category["examples"]) == 3:
                    break
            m["categories"].append(category)
        extent = np.quantile(np.abs(points[dev]), 0.98, axis=0)
        # A square, fixed viewport preserves the distance geometry on both axes.
        m["extent"] = float(max(2.0, max(extent) * 1.2))
        p["model"] = model
        p["phones"] = phones
        (live.ASSETS / f"{ident}.json").write_text(json.dumps(serializable(p), allow_nan=False))
        catalog["maps"].append(m)
        report["maps"][ident] = {"model": model, **metrics}
        print(ident, "TEST", json.dumps(metrics, ensure_ascii=False), flush=True)
    (live.ASSETS / "catalog.json").write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2, allow_nan=False)
    )
    (RESULTS / "mandarin-maps.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    )


if __name__ == "__main__":
    main()
