"""Fit and evaluate an expanded fixed map using development speakers only."""

import json
import sys
import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from speechlab import live, phonetics

from .mandarin_expand_prepare import OUT, PROTOCOL, stable
from .mandarin_features import crop
from .mandarin_map import RESULTS, centers, classify, reference_examples, score

COLORS = {
    "vowels": "#498e7a",
    "fricatives": "#8063a6",
    "sonorants": "#b0783d",
    "stops": "#447f9c",
    "affricates": "#b26683",
}
NAMES = {
    "vowels": "Vowels",
    "fricatives": "F · H · S · SH · X",
    "sonorants": "M · N · NG · L · R",
    "stops": "B · P · D · T · G · K",
    "affricates": "Z · C · ZH · CH · J · Q",
}


def targets(labels):
    anchors = {}
    for row, phones in enumerate(PROTOCOL["families"].values()):
        for col, phone in enumerate(phones):
            anchors[phone] = [col - (len(phones) - 1) / 2, 2 - row]
    return np.array([anchors[label] for label in labels])


def fit(x, y, method, dimensions, pooling="four"):
    original = np.asarray(x, dtype=np.float64)
    x = original.reshape(len(original), 4, -1).mean(axis=1) if pooling == "mean4" else original
    scaler = StandardScaler().fit(x)
    pca = PCA(n_components=dimensions, svd_solver="randomized", random_state=PROTOCOL["seed"]).fit(
        scaler.transform(x)
    )
    reduced = pca.transform(scaler.transform(x))
    norm = StandardScaler().fit(reduced)
    reduced = norm.transform(reduced) if method != "classifier" else reduced
    weights = np.array([1 / np.mean(y == label) for label in y])
    weights /= weights.mean()
    model = (
        LogisticRegression(C=0.1, max_iter=2000)
        if method == "classifier"
        else Ridge(alpha=100)
        if method == "ridge"
        else MLPRegressor(
            hidden_layer_sizes=(128,),
            activation="tanh",
            alpha=1,
            learning_rate_init=0.001,
            max_iter=250,
            early_stopping=True,
            n_iter_no_change=15,
            random_state=PROTOCOL["seed"],
        )
    )
    model.fit(reduced, y if method == "classifier" else targets(y), sample_weight=weights)
    pre = pca.components_.T / norm.scale_
    offset = (pca.mean_ @ pca.components_.T + norm.mean_) / norm.scale_
    if method == "classifier":
        pre = pca.components_.T
        offset = pca.mean_ @ pre
    first = model.coef_.T if method != "mlp" else model.coefs_[0]
    intercept = model.intercept_ if method != "mlp" else model.intercepts_[0]
    p = dict(
        mean=scaler.mean_,
        scale=scaler.scale_,
        basis=pre @ first,
        shift=offset @ first - intercept,
        center=np.zeros(2),
        rotation=np.eye(2),
        radius=1.0,
        input_pooling=pooling,
        method=method,
        pca_dimensions=dimensions,
        model="qwen-asr-large",
    )
    if method == "mlp":
        p.update(
            hidden_activation="tanh",
            output_weight=model.coefs_[1],
            output_bias=model.intercepts_[1],
            training_iterations=int(model.n_iter_),
        )
    if method == "classifier":
        assert model.n_iter_.max() < model.max_iter, "Classifier did not converge"
        p.update(
            hidden_activation="softmax", anchors=targets(model.classes_), classes=model.classes_.tolist()
        )
        p["optimizer_iterations"] = int(model.n_iter_.max())
        expected = model.predict_proba(reduced) @ p["anchors"]
    else:
        expected = model.predict(reduced)
    np.testing.assert_allclose(live.coordinates(original, p), expected, atol=1e-8)
    return p


def main(jitter=False):
    rows = json.loads((OUT / "manifest.json").read_text())
    saved = np.load(OUT / "features.npz")
    np.testing.assert_array_equal(saved["ids"], [r["id"] for r in rows])
    x = saved["x"]
    y = np.array([r["phone"] for r in rows])
    splits = np.array([r["split"] for r in rows])
    speakers = np.array([r["speaker"] for r in rows])
    phones = list(PROTOCOL["phones"])
    tr = splits == "train"
    val = splits == "validation"
    validation = {}
    validation_details = {}
    configs = {}
    if jitter:
        items = json.loads((OUT / "jitter-manifest.json").read_text())
        augmented = np.load(OUT / "jitter.npz")
        np.testing.assert_array_equal(augmented["ids"], [r["id"] for r in items])
        aug_x = augmented["x"]
        aug_indices = np.array([r["index"] for r in items])
        aug_y = y[aug_indices]
        aug_train = splits[aug_indices] == "train"
        aug_val = splits[aug_indices] == "validation"
        assert not np.any(splits[aug_indices] == "test")
        candidates = [
            ("mlp", 128, pooling, use_aug) for pooling in ["four", "mean4"] for use_aug in [False, True]
        ] + [("classifier", 128, "four", use_aug) for use_aug in [False, True]]
    else:
        candidates = [(method, dim, "four", False) for dim in [128, 256] for method in ["ridge", "mlp"]]
    for method, dim, pooling, use_aug in candidates:
        started = time.monotonic()
        fit_x = np.concatenate([x[tr], aug_x[aug_train]]) if use_aug else x[tr]
        fit_y = np.concatenate([y[tr], aug_y[aug_train]]) if use_aug else y[tr]
        p = fit(fit_x, fit_y, method, dim, pooling)
        mu = centers(live.coordinates(x[tr], p), y[tr], phones)
        pred = classify(live.coordinates(x[val], p), mu, phones)
        center_score = float(balanced_accuracy_score(y[val], pred))
        shift_score = (
            float(
                balanced_accuracy_score(
                    aug_y[aug_val], classify(live.coordinates(aug_x[aug_val], p), mu, phones)
                )
            )
            if jitter
            else center_score
        )
        key = f"{method}/{dim}/{pooling}/{use_aug}"
        validation[key] = (center_score + shift_score) / 2
        validation_details[key] = dict(centered=center_score, shifted=shift_score)
        configs[key] = dict(method=method, dimensions=dim, pooling=pooling, augmented=use_aug)
        print(
            key,
            "validation",
            validation_details[key],
            "seconds",
            round(time.monotonic() - started, 1),
            flush=True,
        )
    winner = max(validation, key=validation.get)
    chosen = configs[winner]
    selection = dict(
        validation=validation, validation_details=validation_details, **chosen, protocol=PROTOCOL
    )
    (RESULTS / "mandarin-expanded-selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2)
    )
    dev = splits != "test"
    test = ~dev
    fit_x = np.concatenate([x[dev], aug_x]) if chosen["augmented"] else x[dev]
    fit_y = np.concatenate([y[dev], aug_y]) if chosen["augmented"] else y[dev]
    p = fit(fit_x, fit_y, chosen["method"], chosen["dimensions"], chosen["pooling"])
    points = live.coordinates(x, p)
    mu = centers(points[dev], y[dev], phones)
    guess = classify(points[test], mu, phones)
    metrics = score(y[test], guess, speakers[test], phones)
    if p.get("hidden_activation") == "softmax":
        logits = ((x[test] - p["mean"]) / p["scale"]) @ p["basis"] - p["shift"]
        raw_prediction = np.array(p["classes"])[logits.argmax(axis=1)]
        metrics["classifier_balanced_accuracy_before_2d"] = float(
            balanced_accuracy_score(y[test], raw_prediction)
        )
    metrics["families"] = {
        family: float(np.mean([metrics["recall"][phone] for phone in group]))
        for family, group in PROTOCOL["families"].items()
    }
    print("TEST", json.dumps(metrics, ensure_ascii=False), flush=True)
    indices = np.array(
        [
            i
            for phone in phones
            for i in sorted(np.flatnonzero(test & (y == phone)), key=lambda i: stable(rows[i]["id"]))[:20]
        ]
    )
    shifts = {}
    for shift in [-0.04, -0.02, 0, 0.02, 0.04]:
        if shift == 0:
            values = points[indices]
        else:
            cache = OUT / f"timing-{round(shift * 1000)}.npz"
            expected_ids = np.array([rows[i]["id"] for i in indices])
            if cache.exists():
                stored = np.load(cache)
                np.testing.assert_array_equal(stored["ids"], expected_ids)
                vectors = stored["x"]
            else:
                vectors = np.stack(
                    [phonetics.embedding(crop(rows[i], shift), "qwen-asr-large") for i in indices]
                )
                np.savez_compressed(cache, x=vectors, ids=expected_ids)
            values = live.coordinates(vectors, p)
        pred = classify(values, mu, phones)
        key = str(round(shift * 1000))
        shifts[key] = dict(
            balanced_accuracy=float(balanced_accuracy_score(y[indices], pred)),
            families={
                family: float(np.mean([np.mean(pred[y[indices] == phone] == phone) for phone in group]))
                for family, group in PROTOCOL["families"].items()
            },
        )
        print("Timing", key, shifts[key]["balanced_accuracy"], flush=True)
    metrics["timing_shifts"] = shifts
    metrics["timing_tokens"] = len(indices)
    catalog = live.catalog()
    m = dict(
        id="mandarin-all",
        name="All Mandarin sounds",
        description="21 initial consonants, final ng, and six vowels. Brief consonants leave a trace as you say a syllable.",
        model="qwen-asr-large",
        test=metrics,
        categories=[],
        families=[{"id": key, "label": NAMES[key]} for key in PROTOCOL["families"]],
        extent=float(max(3.5, np.quantile(np.abs(points[dev]), 0.995) * 1.08)),
        window_ms=160,
        hop_ms=40,
        projection_kind="category_blend" if chosen["method"] == "classifier" else "direct",
    )
    for n, phone in enumerate(phones):
        indices = np.flatnonzero(dev & (y == phone))
        family = next(key for key, group in PROTOCOL["families"].items() if phone in group)
        examples = reference_examples(m["id"], n, indices, rows, points)
        m["categories"].append(
            dict(
                phone=phone,
                label=PROTOCOL["phones"][phone],
                kind="vowel" if phone in PROTOCOL["vowels"] else "consonant",
                family=family,
                color=COLORS[family],
                center=mu[n].tolist(),
                covariance=np.cov(points[indices].T).tolist(),
                points=points[sorted(indices, key=lambda i: stable(rows[i]["id"]))[:25]].tolist(),
                examples=examples,
                recall=metrics["recall"][phone],
            )
        )
    # Compact float64 archives preserve exact math without megabytes of textual coefficients.
    arrays = {key: value for key, value in p.items() if isinstance(value, np.ndarray)}
    meta = {key: value for key, value in p.items() if key not in arrays}
    meta["arrays_file"] = "mandarin-all.npz"
    np.savez_compressed(live.ASSETS / "mandarin-all.npz", **arrays)
    (live.ASSETS / "mandarin-all.json").write_text(json.dumps(meta, indent=2))
    catalog["maps"] = [old for old in catalog["maps"] if old["id"] != m["id"]] + [m]
    catalog["default_map"] = "mandarin-all"
    (live.ASSETS / "catalog.json").write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2, allow_nan=False)
    )
    (RESULTS / "mandarin-expanded.json").write_text(
        json.dumps(
            dict(
                protocol=PROTOCOL,
                corpus=json.loads((OUT / "summary.json").read_text()),
                selection=selection,
                test=metrics,
            ),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main(jitter="--jitter" in sys.argv)
