import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from research.mandarin_map import fit_projection
from speechlab import live, store
from speechlab.app import app


@pytest.mark.parametrize("method", ["lda", "ridge"])
def test_two_dimensional_export_and_orientation_preserve_distances(method):
    rng = np.random.default_rng(12)
    labels = np.repeat(["a", "i", "u"], 70)
    x = rng.normal(size=(210, 128))
    x[:, :3] += np.eye(3)[np.repeat(np.arange(3), 70)] * 3
    p = fit_projection(x, labels, method=method)  # Also checks the fitted sklearn transform.
    original = live.coordinates(x, p)
    p["center"] = np.array([0.8, -0.3])
    p["rotation"] = np.array([[0, -1], [1, 0]])
    p["radius"] = 2.7
    turned = live.coordinates(x, p)
    np.testing.assert_allclose(
        np.linalg.norm(np.diff(original, axis=0), axis=1),
        np.linalg.norm(np.diff(turned, axis=0), axis=1) * 2.7,
    )
    encoded = json.loads(
        json.dumps({k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in p.items()})
    )
    np.testing.assert_allclose(turned, live.coordinates(x, encoded))


def test_live_api_rejects_bad_frames_and_origin_before_inference(monkeypatch):
    def forbidden(*args):
        pytest.fail("Invalid input reached the encoder")

    monkeypatch.setattr(live.phonetics, "embedding", forbidden)
    with TestClient(app) as client:
        path = "/api/live/mandarin/frame"
        assert client.post(path, content=b"short").status_code == 400
        assert client.post(path, content=b"0" * 10241).status_code == 413
        assert client.post(path, content=np.full(2560, np.nan, dtype="<f4").tobytes()).status_code == 400
        assert client.post(path, content=np.full(2560, 2, dtype="<f4").tobytes()).status_code == 400
        assert client.post(path, content=np.zeros(2560, dtype="<f4").tobytes()).json() == {
            "active": False,
            "reason": "quiet",
        }
        assert client.post(path, content=np.ones(2560, dtype="<f4").tobytes()).json() == {
            "active": False,
            "reason": "clipping",
        }
        assert client.post(path, headers={"Origin": "https://unrelated.example"}).status_code == 403
        assert client.post("/api/live/unknown/frame", content=b"anything").status_code == 400


def test_live_frames_use_projection_and_never_persist(monkeypatch):
    monkeypatch.setattr(live.phonetics, "embedding", lambda *args: np.array([2.0, 3.0]))
    monkeypatch.setattr(
        live,
        "projection",
        lambda *args: dict(
            mean=np.zeros(2),
            scale=np.ones(2),
            basis=np.eye(2),
            shift=np.zeros(2),
            center=np.zeros(2),
            rotation=np.eye(2),
            radius=2.0,
        ),
    )
    before = {kind: len(store.records(kind)) for kind in ["clips", "results"]}
    wave = (np.sin(np.arange(2560) * 0.1) * 0.1).astype("<f4")
    with TestClient(app) as client:
        result = client.post("/api/live/mandarin/frame", content=wave.tobytes()).json()
        assert result["active"] is True and result["x"] == 1 and result["y"] == 1.5
        assert result["processing_ms"] >= 0
        json.dumps(result, allow_nan=False)
    assert before == {kind: len(store.records(kind)) for kind in before}


def test_reference_assets_have_development_sources_and_finite_regions():
    with TestClient(app) as client:
        assert client.get("/live.html").status_code == 200
        catalog = client.get("/api/live").json()
        json.dumps(catalog, allow_nan=False)
        for m in catalog["maps"]:
            assert m["test"]["speakers"] == 10
            assert live.projection(m["id"])["model"] == m["model"]
            for c in m["categories"]:
                assert np.linalg.eigvalsh(c["covariance"]).min() > 0
                assert len({e["speaker"] for e in c["examples"]}) == 3
                for index, example in enumerate(c["examples"]):
                    assert example["split"] != "test"
                    result = client.get(f"/api/live/{m['id']}/example/{c['phone']}/{index}")
                    assert result.content[:4] == b"fLaC"
