"""A regularized, supervised direction with session/speaker-held-out evaluation."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from speechlab import models, store


def train(
    negative_ids,
    positive_ids,
    name,
    negative_label="Less",
    positive_label="More",
    model="wavlm-base",
    layer=None,
    group_by="session",
    progress=lambda _: None,
):
    if len(set(negative_ids)) < 3 or len(set(positive_ids)) < 3:
        raise ValueError("Use at least three distinct clips on each side of the direction.")
    if set(negative_ids) & set(positive_ids):
        raise ValueError("A clip cannot appear on both sides.")
    negative_ids, positive_ids = list(dict.fromkeys(negative_ids)), list(dict.fromkeys(positive_ids))
    ids = negative_ids + positive_ids
    x, groups, infos = [], [], []
    for ident in ids:
        clip = store.read("clips", ident)
        info = models.extract(
            ident, model, layer, clip.get("region_start", 0), clip.get("region_end"), progress
        )
        x.append(models.read_features(info)[0])
        groups.append(clip.get(group_by, "unknown"))
        infos.append(info)
    x, y, groups = np.stack(x), np.array([0] * len(negative_ids) + [1] * len(positive_ids)), np.array(groups)
    predictions, truth = [], []
    skipped = 0
    for tr, te in LeaveOneGroupOut().split(x, y, groups) if len(set(groups)) > 1 else []:
        if len(set(y[tr])) < 2:
            skipped += 1
            continue
        probe = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000))
        probe.fit(x[tr], y[tr])
        predictions.extend(probe.predict(x[te]).tolist())
        truth.extend(y[te].tolist())
    probe = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000))
    probe.fit(x, y)
    scaler, clf = probe.steps[0][1], probe.steps[1][1]
    score = float(balanced_accuracy_score(truth, predictions)) if len(set(truth)) == 2 else None
    return store.save(
        "axes",
        {
            "name": name,
            "model": model,
            "layer": infos[0]["layer"],
            "negative_label": negative_label,
            "positive_label": positive_label,
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "coef": clf.coef_[0].tolist(),
            "intercept": float(clf.intercept_[0]),
            "training_ids": ids,
            "group_by": group_by,
            "groups": len(set(groups)),
            "validation": {
                "balanced_accuracy": score,
                "evaluated_clips": len(truth),
                "skipped_folds": skipped,
            },
            "note": "A learned contrast, not a native-likeness score or a measured articulatory direction. Validate on new sessions and speakers.",
        },
    )


def predict(clip_id, axis_id, progress=lambda _: None):
    axis, clip = store.read("axes", axis_id), store.read("clips", clip_id)
    info = models.extract(
        clip_id, axis["model"], axis["layer"], clip.get("region_start", 0), clip.get("region_end"), progress
    )
    vector, _ = models.read_features(info)
    margin = float(
        np.dot((vector - axis["mean"]) / np.array(axis["scale"]), axis["coef"]) + axis["intercept"]
    )
    probability = float(1 / (1 + np.exp(-np.clip(margin, -40, 40))))
    return {
        "kind": "direction",
        "clip_id": clip_id,
        "axis_id": axis_id,
        "name": axis["name"],
        "margin": margin,
        "position": probability,
        "negative_label": axis["negative_label"],
        "positive_label": axis["positive_label"],
        "in_training": clip_id in axis["training_ids"],
        "validation": axis["validation"],
        "note": "Position along the fitted contrast; probability is uncalibrated and is not pronunciation accuracy.",
    }
