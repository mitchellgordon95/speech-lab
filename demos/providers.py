"""Optional cloud comparisons. Invoked only by the explicit Send button."""

import base64
import hashlib
import json
import os
import re
import time

import httpx

from speechlab import store
from speechlab.audio import load, wav_bytes

ENV = {
    "azure": ["AZURE_SPEECH_KEY", "AZURE_SPEECH_REGION"],
    "speechsuper": ["SPEECHSUPER_APP_KEY", "SPEECHSUPER_SECRET_KEY"],
    "qwen": ["QWEN_API_KEY", "QWEN_API_URL"],
}


def status():
    return {
        name: {
            "configured": all(os.getenv(k) for k in keys),
            "missing": [k for k in keys if not os.getenv(k)],
        }
        for name, keys in ENV.items()
    }


def assess(clip_id, provider, reference, language="zh-CN", progress=lambda _: None):
    if provider not in ENV or not status()[provider]["configured"]:
        raise ValueError("Configure this provider's credentials in .env and restart the server.")
    if not reference.strip():
        raise ValueError("Provide the intended word or sentence.")
    clip = store.read("clips", clip_id)
    y, sr = load(clip_id, clip.get("region_start", 0), clip.get("region_end"), sr=16000)
    audio = wav_bytes(y, sr)
    progress(f"Sending selected audio to {provider}")
    with httpx.Client(timeout=120) as client:
        if provider == "azure":
            region = os.environ["AZURE_SPEECH_REGION"]
            if not re.fullmatch(r"[a-z0-9-]+", region):
                raise ValueError("Invalid Azure region")
            config = {
                "ReferenceText": reference,
                "GradingSystem": "HundredMark",
                "Granularity": "Phoneme",
                "Dimension": "Comprehensive",
                "EnableMiscue": True,
            }
            response = client.post(
                f"https://{region}.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1",
                params={"language": language, "format": "detailed"},
                content=audio,
                headers={
                    "Ocp-Apim-Subscription-Key": os.environ["AZURE_SPEECH_KEY"],
                    "Content-Type": "audio/wav; codecs=audio/pcm; samplerate=16000",
                    "Pronunciation-Assessment": base64.b64encode(json.dumps(config).encode()).decode(),
                },
            )
        elif provider == "speechsuper":
            key, secret = os.environ["SPEECHSUPER_APP_KEY"], os.environ["SPEECHSUPER_SECRET_KEY"]
            stamp, user = str(int(time.time())), "speech-lab"
            core = os.getenv("SPEECHSUPER_CORE_TYPE", "cn.word.score")
            if not re.fullmatch(r"[a-zA-Z0-9.]+", core):
                raise ValueError("Invalid SpeechSuper core type")

            def sign(value):
                return hashlib.sha1(value.encode()).hexdigest()

            params = {
                "connect": {
                    "cmd": "connect",
                    "param": {
                        "sdk": {"version": 16777472, "source": 9, "protocol": 2},
                        "app": {"applicationId": key, "sig": sign(key + stamp + secret), "timestamp": stamp},
                    },
                },
                "start": {
                    "cmd": "start",
                    "param": {
                        "app": {
                            "userId": user,
                            "applicationId": key,
                            "timestamp": stamp,
                            "sig": sign(key + stamp + user + secret),
                        },
                        "audio": {"audioType": "wav", "channel": 1, "sampleBytes": 2, "sampleRate": 16000},
                        "request": {"coreType": core, "refText": reference, "tokenId": store.uid()},
                    },
                },
            }
            response = client.post(
                "https://api.speechsuper.com/" + core,
                data={"text": json.dumps(params)},
                headers={"Request-Index": "0"},
                files={"audio": ("clip.wav", audio, "audio/wav")},
            )
        else:
            url = os.environ["QWEN_API_URL"]
            if not url.startswith("https://"):
                raise ValueError("QWEN_API_URL must use HTTPS")
            prompt = (
                f"Evaluate this learner recording of {reference!r} in {language}. Focus on consonants and vowels. "
                "Separate audible observations, uncertainty, and suggested practice. Do not infer exact tongue "
                "position from audio. Do not treat successful transcription as proof of accurate pronunciation. "
                "If the clip cannot support a conclusion, say so. Keep the answer under 200 words."
            )
            payload = {
                "model": os.getenv("QWEN_AUDIO_MODEL", "qwen3.8-omni-flash"),
                "stream": True,
                "modalities": ["text"],
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "input_audio",
                                "input_audio": {
                                    "data": "data:audio/wav;base64," + base64.b64encode(audio).decode(),
                                    "format": "wav",
                                },
                            },
                        ],
                    }
                ],
            }
            pieces = []
            with client.stream(
                "POST", url, json=payload, headers={"Authorization": "Bearer " + os.environ["QWEN_API_KEY"]}
            ) as stream:
                if stream.status_code >= 400:
                    raise ValueError(
                        f"Qwen returned HTTP {stream.status_code}. Check the endpoint, model, region, and key."
                    )
                for line in stream.iter_lines():
                    if line.startswith("data: ") and line[6:] != "[DONE]":
                        packet = json.loads(line[6:])
                        for choice in packet.get("choices", []):
                            pieces.append(choice.get("delta", {}).get("content") or "")
            return {
                "kind": "provider",
                "provider": provider,
                "clip_id": clip_id,
                "reference": reference,
                "response": {"text": "".join(pieces)},
                "note": "An unvalidated model opinion; compare against human labels.",
            }
        if response.status_code >= 400:
            raise ValueError(
                f"{provider} returned HTTP {response.status_code}. Check credentials, quota, and provider settings."
            )
        result = response.json()
    return {
        "kind": "provider",
        "provider": provider,
        "clip_id": clip_id,
        "reference": reference,
        "response": result,
        "note": "Provider-specific output. Scores are not directly comparable across services.",
    }
