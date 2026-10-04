"""Validate actual outbound payloads without sending recordings to a service."""

import base64
import io
import json

import httpx
import pytest
import soundfile as sf

from demos import providers
from speechlab import store


@pytest.mark.parametrize("name", ["azure", "speechsuper", "qwen"])
def test_payload_and_response(name, clips, monkeypatch):
    clip = clips[0]
    clip.update(region_start=0.2, region_end=0.6)
    store.write_json(store.path_for("clips", clip["id"]), clip)
    for key in providers.ENV[name]:
        monkeypatch.setenv(key, "test-key")
    monkeypatch.setenv("AZURE_SPEECH_REGION", "eastus")
    monkeypatch.setenv("QWEN_API_URL", "https://example.com/chat/completions")
    monkeypatch.setenv("SPEECHSUPER_CORE_TYPE", "cn.word.score")
    requests = []

    def handle(request):
        requests.append(request)
        assert request.method == "POST"
        if name == "azure":
            config = json.loads(base64.b64decode(request.headers["Pronunciation-Assessment"]))
            assert config["ReferenceText"] == "你好"
            assert config["Granularity"] == "Phoneme"
            wav = request.content
        elif name == "speechsuper":
            assert request.url.path == "/cn.word.score"
            assert b'"coreType": "cn.word.score"' in request.content
            assert b'name="audio"; filename="clip.wav"' in request.content
            assert b"test-keytest-key" not in request.content
            wav = None
        else:
            body = json.loads(request.content)
            assert body["stream"] and body["modalities"] == ["text"]
            data = body["messages"][0]["content"][1]["input_audio"]["data"]
            wav = base64.b64decode(data.split(",", 1)[1])
        if wav:
            y, sr = sf.read(io.BytesIO(wav))
            assert sr == 16000 and len(y) == 6400  # Only the selected 0.4-second crop.
        if name == "qwen":
            chunks = [
                'data: {"choices":[{"delta":{"content":"Test "}}]}',
                'data: {"choices":[{"delta":{"content":"feedback"}}]}',
                "data: [DONE]",
            ]
            return httpx.Response(200, text="\n\n".join(chunks))
        return httpx.Response(200, json={"result": "mock response"})

    original_client = httpx.Client
    monkeypatch.setattr(
        providers.httpx,
        "Client",
        lambda **kwargs: original_client(transport=httpx.MockTransport(handle), **kwargs),
    )
    result = providers.assess(clip["id"], name, "你好")
    assert len(requests) == 1 and result["provider"] == name
    assert result["response"] == (
        {"text": "Test feedback"} if name == "qwen" else {"result": "mock response"}
    )
