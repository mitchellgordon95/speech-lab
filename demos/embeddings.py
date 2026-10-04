import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

from speechlab import models, store


def compare(clip_ids, model="wavlm-base", layer=None, progress=lambda _: None):
    if len(clip_ids) < 2:
        raise ValueError("Select at least two clips to compare.")
    infos, vectors, frame_sets, labels = [], [], [], []
    for ident in clip_ids:
        clip = store.read("clips", ident)
        info = models.extract(
            ident, model, layer, clip.get("region_start", 0), clip.get("region_end"), progress
        )
        vector, frames = models.read_features(info)
        infos.append(info)
        vectors.append(vector)
        frame_sets.append(frames)
        labels.append(clip["label"])
    x = np.stack(vectors)
    xy = PCA(n_components=2).fit_transform(x)
    distances = cdist(x, x, metric="cosine")
    return {
        "kind": "comparison",
        "model": model,
        "layer": infos[0]["layer"],
        "clip_ids": clip_ids,
        "labels": labels,
        "points": xy.tolist(),
        "cosine_distances": np.clip(distances, 0, 2).tolist(),
        "features": infos,
        "note": "PCA is exploratory; its axes are not tongue position or pronunciation quality. Compare the same sound context.",
    }
