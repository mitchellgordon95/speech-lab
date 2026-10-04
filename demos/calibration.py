import numpy as np
from scipy.spatial.distance import cosine

from speechlab import models, store


def compare(clip_id, reference_ids, anchor_ids, model="wavlm-base", layer=None, progress=lambda _: None):
    if not reference_ids or not anchor_ids:
        raise ValueError("Choose at least one population reference and one personal anchor.")
    if clip_id in reference_ids + anchor_ids:
        raise ValueError("Use a held-out attempt, not one of the anchors.")

    def vector(ident):
        c = store.read("clips", ident)
        result = models.extract(ident, model, layer, c.get("region_start", 0), c.get("region_end"), progress)
        return models.read_features(result)[0]

    attempt = vector(clip_id)
    refs, anchors = np.stack([vector(i) for i in reference_ids]), np.stack([vector(i) for i in anchor_ids])
    reference_distance = float(cosine(attempt, refs.mean(0)))
    personal_distance = float(cosine(attempt, anchors.mean(0)))
    return {
        "kind": "calibration",
        "clip_id": clip_id,
        "model": model,
        "reference_ids": reference_ids,
        "anchor_ids": anchor_ids,
        "reference_distance": reference_distance,
        "personal_distance": personal_distance,
        "reference_distances": [float(cosine(attempt, v)) for v in refs],
        "personal_distances": [float(cosine(attempt, v)) for v in anchors],
        "note": "Smaller cosine distance means more similar. A personal anchor must be independently accepted; similarity does not prove correctness. These distances are not interchangeable quality scores.",
    }
