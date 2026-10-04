import json
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from demos import acoustics, calibration, directions, fixtures, providers
from speechlab import audio, models, store
from speechlab.app import app


def test_audio_and_crops(clips):
    original, rate = audio.load(clips[0]["id"])
    cropped, reduced_rate = audio.load(clips[0]["id"], 0.2, 0.6, sr=16000)
    assert rate == 48000 and reduced_rate == 16000
    assert len(cropped) == 6400 and len(original) == 43200
    with pytest.raises(ValueError, match="region"):
        audio.load(clips[0]["id"], 0.8, 0.4)
    with pytest.raises(ValueError, match="silent"):
        audio.decode(audio.wav_bytes(np.zeros(16000), 16000))
    with pytest.raises(ValueError, match="30 seconds"):
        audio.decode(audio.wav_bytes(np.ones(16000 * 31) * 0.1, 16000))


def test_formants_and_spectrum_detect_synthetic_contrast(clips):
    a = acoustics.analyze(next(c["id"] for c in clips if c["contrast"] == "A"))
    b = acoustics.analyze(next(c["id"] for c in clips if c["contrast"] == "B"))
    assert a["formant_medians"][0] < b["formant_medians"][0]
    assert a["formant_medians"][1] > b["formant_medians"][1]
    assert a["features"][6] > 0.8
    json.dumps(a, allow_nan=False)


def test_unvoiced_signal_does_not_invent_formants():
    y = np.random.default_rng(1).normal(0, 0.1, 48000)
    c = audio.add(audio.wav_bytes(y, 48000), {"label": "noise"})
    result = acoustics.analyze(c["id"])
    assert result["formant_medians"] == [None, None, None]
    assert result["warnings"]


@pytest.fixture
def synthetic_features(clips, monkeypatch):
    def extract(ident, *args, **kwargs):
        return {"id": ident, "layer": 6}

    def read(info):
        c = store.read("clips", info["id"])
        # Deliberately separable labels plus an unrelated session coordinate.
        return np.array([1 if c["contrast"] == "A" else -1, 0.1, int(c["session"][-1])]), None

    monkeypatch.setattr(models, "extract", extract)
    monkeypatch.setattr(models, "read_features", read)
    return clips


def test_direction_group_validation_and_training_detection(synthetic_features):
    clips = synthetic_features
    sides = [[c["id"] for c in clips if c["contrast"] == side] for side in ("A", "B")]
    axis = directions.train(*sides, name="Synthetic A / B")
    assert axis["validation"]["balanced_accuracy"] == 1
    assert axis["validation"]["evaluated_clips"] == 12
    assert directions.predict(sides[0][0], axis["id"])["position"] < 0.5
    assert directions.predict(sides[1][0], axis["id"])["in_training"]
    with pytest.raises(ValueError, match="both sides"):
        directions.train(sides[0], sides[0], "Invalid")
    one_day = [
        [c["id"] for c in clips if c["contrast"] == side and c["session"].endswith("1")]
        for side in ("A", "B")
    ]
    axis = directions.train(*one_day, name="No independent day")
    assert axis["validation"]["balanced_accuracy"] is None
    assert axis["validation"]["evaluated_clips"] == 0


def test_personal_calibration_rejects_self_comparison(synthetic_features):
    ids = [c["id"] for c in synthetic_features]
    with pytest.raises(ValueError, match="held-out"):
        calibration.compare(ids[0], ids[1:3], [ids[0]])
    result = calibration.compare(ids[0], ids[3:6], ids[1:3])
    assert result["personal_distance"] < result["reference_distance"]


def test_feature_cache_keeps_regions_and_layers_separate(clips, monkeypatch):
    calls = []

    def frames(y, name, layer, progress):
        calls.append((len(y), layer))
        return np.ones((9, 5), dtype=np.float32), "test"

    monkeypatch.setattr(models, "extract_frames", frames)
    ident = clips[0]["id"]
    a = models.extract(ident)
    assert models.extract(ident)["cached"]
    b = models.extract(ident, start=0.2, end=0.6)
    c = models.extract(ident, layer=7)
    assert len({x["feature_id"] for x in (a, b, c)}) == 3
    assert calls == [(14400, 6), (6400, 6), (14400, 7)]


def test_api_upload_analysis_and_cloud_gate():
    with TestClient(app) as client:
        blob = audio.wav_bytes(fixtures.vowel(350, 2200), 48000)
        response = client.post(
            "/api/clips",
            files={"file": ("vowel.wav", blob, "audio/wav")},
            data={"metadata": json.dumps({"label": "test", "session": "day-1"})},
        )
        assert response.status_code == 200
        clip = response.json()
        assert client.get(f"/api/clips/{clip['id']}/audio").content == blob
        assert (
            client.post("/api/analyze", json={"demo": "providers", "clip_id": clip["id"]}).status_code == 400
        )
        assert client.post("/api/analyze", json={"demo": "acoustics"}).status_code == 400
        assert client.post("/api/fixtures", headers={"origin": "https://example.com"}).status_code == 403
        job = client.post("/api/analyze", json={"demo": "acoustics", "clip_id": clip["id"]}).json()
        deadline = time.monotonic() + 10
        while job["status"] in ("queued", "running") and time.monotonic() < deadline:
            time.sleep(0.03)
            job = client.get("/api/jobs/" + job["id"]).json()
        assert job["status"] == "done", job
        assert job["result"]["formant_medians"][0] > 0
        assert len(client.get("/api/export").json()["results"]) == 1
        assert client.patch("/api/clips/" + clip["id"], json={"region_start": 4}).status_code == 400


@pytest.mark.parametrize("provider", ["azure", "speechsuper", "qwen"])
def test_provider_requires_credentials(provider, monkeypatch, clips):
    for key in providers.ENV[provider]:
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(ValueError, match="credentials"):
        providers.assess(clips[0]["id"], provider, "你好")


def test_seed_is_idempotent_and_traversal_rejected(clips):
    assert len(fixtures.seed()) == 12
    with pytest.raises(ValueError, match="identifier"):
        store.read("clips", "../../.env")
