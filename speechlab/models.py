"""Frozen encoders. Only one model is resident; all features are cached on disk."""

import gc
import hashlib
import importlib
import json
import os
import time
from pathlib import Path

import numpy as np

from .audio import load
from .config import DATA

MODELS = {
    "wavlm-base": {
        "repo": "microsoft/wavlm-base-plus",
        "revision": "4c66d4806a428f2e922ccfa1a962776e232d487b",
        "family": "wavlm",
        "layer": 6,
        "layers": 12,
        "size": "~380 MB",
    },
    "wavlm-large": {
        "repo": "microsoft/wavlm-large",
        "revision": "c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c",
        "family": "wavlm",
        "layer": 9,
        "layers": 24,
        "size": "~1.3 GB",
    },
    "xls-r": {
        "repo": "facebook/wav2vec2-xls-r-300m",
        "revision": "1a640f32ac3e39899438a2931f9924c02f080a54",
        "family": "wav2vec2",
        "layer": 12,
        "layers": 24,
        "size": "~1.3 GB",
    },
    "qwen-asr": {
        "repo": "Qwen/Qwen3-ASR-0.6B-hf",
        "revision": "7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c",
        "family": "qwen3_asr",
        "layer": -1,
        "layers": 18,
        "size": "~1.2 GB checkpoint",
    },
    "qwen-asr-large": {
        "repo": "Qwen/Qwen3-ASR-1.7B-hf",
        "revision": "bcd2b5b7f32b480ab5790554cfa8347f246a14f3",
        "family": "qwen3_asr",
        "layer": -1,
        "layers": 24,
        "size": "~3.5 GB checkpoint",
    },
    "qwen-omni": {
        "repo": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "revision": "26291f793822fb6be9555850f06dfe95f2d7e695",
        "family": "qwen3_omni_moe",
        "layer": -1,
        "layers": 32,
        "size": "Only encoder-containing shards; several GB",
    },
}
_resident = None


def device():
    import torch

    requested = os.environ.get("SPEECHLAB_DEVICE", "auto")
    if requested != "auto":
        return requested
    return "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


def unload():
    global _resident
    _resident = None
    gc.collect()
    import torch

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _qwen_encoder(spec, progress):
    """Select tensors by their audio_tower prefix without constructing the LLM."""
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from transformers import AutoConfig, AutoFeatureExtractor

    repo, family = spec["repo"], spec["family"]
    cfg = AutoConfig.from_pretrained(repo, revision=spec["revision"], trust_remote_code=False)
    cfg = getattr(cfg, "thinker_config", cfg)
    audio_cfg = cfg.audio_config
    module = importlib.import_module(f"transformers.models.{family}.modeling_{family}")
    cls_name = "Qwen3OmniMoeAudioEncoder" if family == "qwen3_omni_moe" else "Qwen3ASREncoder"
    cls = getattr(module, cls_name)
    audio_cfg._attn_implementation = "eager"
    if family == "qwen3_omni_moe":
        index = json.loads(
            Path(hf_hub_download(repo, "model.safetensors.index.json", revision=spec["revision"])).read_text()
        )
        relevant = {
            k: v
            for k, v in index["weight_map"].items()
            if ".audio_tower." in k or k.startswith("audio_tower.")
        }
        shards = sorted(set(relevant.values()))
    else:
        shards = ["model.safetensors"]
    if not shards:
        raise ValueError("No separate audio encoder tensors found in this checkpoint.")
    weights = {}
    for shard in shards:
        progress(f"Downloading/reading audio encoder weights: {shard}")
        path = hf_hub_download(repo, shard, revision=spec["revision"])
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if ".audio_tower." in key or key.startswith("audio_tower."):
                    weights[key.split("audio_tower.", 1)[1]] = f.get_tensor(key)
    if not weights:
        raise ValueError("Checkpoint does not expose the expected audio_tower weights.")
    # meta + assign avoids a second random allocation of the encoder weights.
    with torch.device("meta"):
        model = cls(audio_cfg)
    model.load_state_dict(weights, strict=True, assign=True)
    # Nonpersistent sinusoidal buffers aren't in the state dict.
    for mod in model.modules():
        if hasattr(mod, "compute_default_singular_positional_embedding"):
            mod.positional_embedding = mod.compute_default_singular_positional_embedding().cpu()
    model = model.float().eval()
    extractor = AutoFeatureExtractor.from_pretrained(repo, revision=spec["revision"])
    return model, extractor


def get_model(name, progress=lambda _: None):
    global _resident
    if name not in MODELS:
        raise ValueError("Unknown encoder")
    if _resident and _resident[0] == name:
        return _resident[1:]
    unload()
    import torch
    from transformers import AutoFeatureExtractor, Wav2Vec2Model, WavLMModel

    torch.set_num_threads(min(6, os.cpu_count() or 4))
    spec = MODELS[name]
    progress(f"Loading {spec['repo']} (first run downloads weights)")
    if spec["family"].startswith("qwen"):
        model, extractor = _qwen_encoder(spec, progress)
    else:
        cls = WavLMModel if spec["family"] == "wavlm" else Wav2Vec2Model
        model = cls.from_pretrained(spec["repo"], revision=spec["revision"]).eval()
        extractor = AutoFeatureExtractor.from_pretrained(spec["repo"], revision=spec["revision"])
    backend = device()
    model = model.to(backend)
    _resident = (name, model, extractor, backend)
    return _resident[1:]


def extract_frames(y, name, layer=None, progress=lambda _: None, normalize=True):
    import torch

    model, extractor, backend = get_model(name, progress)
    layer = MODELS[name]["layer"] if layer is None else int(layer)
    if not normalize and MODELS[name]["family"] == "wavlm":
        inputs = {"input_values": torch.as_tensor(y, dtype=torch.float32)[None]}
    else:
        inputs = extractor(y, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
    progress(f"Extracting {name} on {backend}")

    def run(where):
        if MODELS[name]["family"].startswith("qwen"):
            features = inputs["input_features"].to(where)
            mask = inputs.get(
                "input_features_mask", inputs.get("attention_mask", inputs.get("feature_attention_mask"))
            ).to(where)
            # Hooks expose unprojected intermediate representations without loading a text decoder.
            captured = []
            handle = None
            if layer >= 0:
                if layer >= len(model.layers):
                    raise ValueError(
                        f"Layer must be -1 or 0 through {len(model.layers) - 1} for this encoder"
                    )
                handle = model.layers[layer].register_forward_hook(
                    lambda _m, _i, out: captured.append(out[0] if isinstance(out, tuple) else out)
                )
            try:
                if MODELS[name]["family"] == "qwen3_asr":
                    # The extractor pads to 30 s. Discard wholly empty CNN chunks;
                    # retaining complete chunks preserves each valid frame's context.
                    chunk = model.n_window * 2
                    valid = int(inputs["attention_mask"].sum(-1).max())
                    padded = ((valid + chunk - 1) // chunk) * chunk
                    output = model(features[:, :, :padded], input_features_mask=mask[:, :padded])
                else:
                    lengths = mask.sum(-1)
                    features = features.permute(0, 2, 1)[mask.bool()].permute(1, 0)
                    output = model(features, feature_lens=lengths)
            finally:
                if handle:
                    handle.remove()
            states = captured[0] if captured else output.last_hidden_state
            return states.reshape(-1, states.shape[-1]).float().cpu().numpy()
        output = model(**{k: v.to(where) for k, v in inputs.items()}, output_hidden_states=True)
        if layer < -len(output.hidden_states) or layer >= len(output.hidden_states):
            raise ValueError(f"Layer index out of range; model has {len(output.hidden_states)} hidden states")
        return output.hidden_states[layer][0].float().cpu().numpy()

    with torch.inference_mode():
        try:
            frames = run(backend)
        except (NotImplementedError, RuntimeError) as e:
            if backend != "mps" or not any(s in str(e).lower() for s in ("mps", "metal", "not implemented")):
                raise
            progress("This operation is unavailable on MPS; retrying on CPU")
            model.to("cpu")
            try:
                frames = run("cpu")
                backend = "cpu (MPS fallback)"
            finally:
                model.to(device())
    if not np.isfinite(frames).all() or len(frames) < 2:
        raise ValueError("Encoder returned insufficient or nonfinite features")
    return frames, backend


def extract(clip_id, name="wavlm-base", layer=None, start=0, end=None, progress=lambda _: None):
    layer = MODELS[name]["layer"] if layer is None else int(layer)
    config = {
        "clip_id": clip_id,
        "model": name,
        "revision": MODELS[name]["revision"],
        "layer": layer,
        "start": start,
        "end": end,
        "version": 2,
    }
    key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:32]
    path = DATA / "features" / f"{key}.npz"
    info_path = path.with_suffix(".json")
    if path.exists() and info_path.exists():
        return {**json.loads(info_path.read_text()), "cached": True}
    started = time.monotonic()
    y, _ = load(clip_id, start, end, sr=16000)
    frames, backend = extract_frames(y, name, layer, progress)
    # Four ordered bins preserve coarse timing; keep full frame sequence for aligned comparisons.
    bins = np.array_split(frames, min(4, len(frames)))
    while len(bins) < 4:
        bins.append(bins[-1])
    vector = np.concatenate([b.mean(0) for b in bins]).astype(np.float32)
    np.savez_compressed(path, frames=frames, vector=vector)
    result = {
        "kind": "embedding",
        **config,
        "feature_id": key,
        "frames": len(frames),
        "dimensions": frames.shape[-1],
        "vector_dimensions": len(vector),
        "device": backend,
        "duration": len(y) / 16000,
        "seconds": round(time.monotonic() - started, 3),
        "cached": False,
        "created": __import__("speechlab.store", fromlist=["now"]).now(),
        "note": "Four ordered temporal means. Distance is representation similarity, not pronunciation accuracy.",
    }
    info_path.write_text(json.dumps(result))
    return result


def read_features(info):
    with np.load(DATA / "features" / f"{info['feature_id']}.npz", allow_pickle=False) as f:
        return f["vector"], f["frames"]
