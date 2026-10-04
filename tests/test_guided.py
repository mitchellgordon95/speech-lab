import json
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from speechlab import audio, guided, phonetics, store
from speechlab.app import app


def test_exported_math_matches_fitted_classifier():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(120, 5))
    x[::8, 1] = np.nan
    y = (x[:, 0] + 0.3 * x[:, 2] > 0).astype(int)
    fit = make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(C=0.1)).fit(x, y)
    imputer, scaler, classifier = [s[1] for s in fit.steps]
    exported = dict(
        impute=imputer.statistics_.tolist(),
        mean=scaler.mean_.tolist(),
        scale=scaler.scale_.tolist(),
        coef=classifier.coef_[0].tolist(),
        intercept=float(classifier.intercept_[0]),
    )
    np.testing.assert_allclose(guided.margin(exported, x), fit.decision_function(x), atol=1e-10)


def test_windows_reject_silence_clipping_and_short_sounds():
    with pytest.raises(ValueError, match="clear sound"):
        guided.active_windows(np.zeros(16000))
    with pytest.raises(ValueError, match="clipping"):
        guided.active_windows(np.tile([-1.0, 1.0], 8000))
    wave = np.sin(np.arange(2560) * 2 * np.pi * 200 / 16000).astype(np.float32) * 0.1
    with pytest.raises(ValueError, match="Hold"):
        guided.active_windows(np.pad(wave, (4000, 4000)))


def test_windows_ignore_surrounding_silence_and_do_not_change_pitch():
    wave = np.sin(np.arange(16000) * 2 * np.pi * 200 / 16000).astype(np.float32) * 0.1
    padded = np.pad(wave, (8000, 12000))
    windows = guided.active_windows(padded)
    assert len(windows) == 7
    for start, y in windows:
        assert 0.5 <= start < 1.5
        assert len(y) == 2560
        assert phonetics.acoustic(y)[11] == pytest.approx(200, abs=1)


def test_phone_preprocessing_is_level_invariant():
    wave = np.random.default_rng(5).normal(0, 0.1, 2560).astype(np.float32)
    np.testing.assert_allclose(phonetics.padded_phone(wave), phonetics.padded_phone(wave * 0.25), atol=1e-6)
    assert phonetics.padded_phone(wave).shape == (8000,)


def test_guided_assets_are_real_and_all_promoted():
    catalog = guided.catalog()
    assert catalog["contrasts"]
    by_id = {r["contrast"]: r for r in catalog["results"]}
    for c in catalog["contrasts"]:
        assert by_id[c["id"]]["eligible"]
        assert c["test"]["speakers"] == 40
        assert guided.probe(c["id"])["contrast"] == c["id"]
        for side, examples in enumerate(c["examples"]):
            assert len({e["speaker"] for e in examples}) == len(examples)
            for i, example in enumerate(examples):
                assert example["source_split"] == "test_clean"
                assert example["phone_end"] > example["phone_start"]
                assert guided.reference(c["id"], side, i).read_bytes()[:4] == b"fLaC"
    json.dumps(catalog, allow_nan=False)


def test_guided_api_validates_before_queueing():
    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        assert client.get("/workbench.html").status_code == 200
        c = client.get("/api/guided").json()["contrasts"][0]["id"]
        assert client.post("/api/guided/analyze", json={"contrast": c}).status_code == 400
        assert client.post("/api/guided/analyze", json={"contrast": "../../.env"}).status_code == 400
        assert client.get(f"/api/guided/{c}/example/0/99/word").status_code == 400
        assert client.get(f"/api/guided/{c}/example/0/0/word").headers["content-type"] == "audio/flac"


@pytest.mark.parametrize("contrast", ["s_sh", "r_l", "iy_ih"])
@pytest.mark.parametrize("voiced", [False, True])
def test_guided_recording_result_saves_and_serializes(contrast, voiced, monkeypatch):
    # Keep the encoder deterministic and confident, so the real voicing check
    # decides the flag rather than an earlier low-confidence branch masking it.
    monkeypatch.setattr(guided, "represent", lambda wave, name: np.array([2.0]))
    monkeypatch.setattr(
        guided,
        "probe",
        lambda ident: {
            "representation": "test",
            "impute": [0.0],
            "mean": [0.0],
            "scale": [1.0],
            "coef": [1.0],
            "intercept": 0.0,
        },
    )
    wave = (
        np.sin(np.arange(16000) * 2 * np.pi * 200 / 16000) * 0.1
        if voiced
        else np.random.default_rng(9).normal(0, 0.05, 16000)
    )
    with TestClient(app) as client:
        clip = client.post("/api/clips", files={"file": ("sound.wav", audio.wav_bytes(wave, 16000))}).json()
        response = client.post("/api/guided/analyze", json={"contrast": contrast, "clip_id": clip["id"]})
        assert response.status_code == 200
        job = response.json()
        deadline = time.monotonic() + 5
        while job["status"] in ("queued", "running") and time.monotonic() < deadline:
            time.sleep(0.01)
            job = client.get(f"/api/jobs/{job['id']}").json()
        assert job["status"] == "done", job
        result = job["result"]
        wrong_kind = voiced if contrast == "s_sh" else not voiced
        assert result["uncertain"] is wrong_kind
        assert (result["position"] is None) is wrong_kind
        assert store.read("results", result["id"])["uncertain"] is wrong_kind
