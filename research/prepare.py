"""Download pinned aligned speech and select a speaker-separated phoneme benchmark."""

import hashlib
import io
import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import hf_hub_download

from speechlab.config import DATA, ROOT

BASE = DATA / "research"
PROTOCOL = json.loads((ROOT / "research/protocol.json").read_text())


def stable(value):
    return hashlib.sha256(f"{PROTOCOL['seed']}:{value}".encode()).hexdigest()


def download(split):
    path = hf_hub_download(
        PROTOCOL["dataset"],
        f"data/{split}-00000-of-00001.parquet",
        repo_type="dataset",
        revision=PROTOCOL["revision"],
    )
    print(f"Downloaded {split}", flush=True)
    return split, path


def prepare():
    BASE.mkdir(exist_ok=True)
    (BASE / "utterances").mkdir(exist_ok=True)
    if (BASE / "manifest.json").exists():
        print("Prepared manifest already exists; no data reselection.")
        return
    with ThreadPoolExecutor(max_workers=2) as pool:
        paths = dict(pool.map(download, ["dev_clean", "test_clean"]))
    allowed = {p for pair in PROTOCOL["contrasts"].values() for p in pair}
    vowels = {"IY", "IH", "AE", "EH", "UW", "UH"}
    samples, rejected, utterances = [], Counter(), {}
    for split, path in paths.items():
        table = pq.read_table(path)
        print(split, table.schema, flush=True)
        rows = table.to_pylist()
        candidates = defaultdict(list)
        for row in rows:
            ident = row["id"]
            speaker = ident.split("-")[0]
            for i, phone in enumerate(row["phonemes"]):
                name = re.sub(r"\d", "", phone["phoneme"]).upper()
                if name not in allowed:
                    continue
                start, end = float(phone["start"]), float(phone["end"])
                if not PROTOCOL["minimum_phone_seconds"] <= end - start <= PROTOCOL["maximum_phone_seconds"]:
                    rejected["duration"] += 1
                    continue
                if name in vowels and not phone["phoneme"].endswith("1"):
                    rejected["unstressed_vowel"] += 1
                    continue
                words = [w for w in row["words"] if w["start"] - 0.005 <= start and w["end"] + 0.005 >= end]
                if len(words) != 1 or words[0]["word"].startswith("<"):
                    rejected["word_boundary_or_unknown"] += 1
                    continue
                w = words[0]
                token = {
                    "id": f"{ident}-{i}",
                    "utterance": ident,
                    "speaker": speaker,
                    "sex": row["sex"],
                    "source_split": split,
                    "phone": name,
                    "start": start,
                    "end": end,
                    "word": w["word"],
                    "word_start": w["start"],
                    "word_end": w["end"],
                    "left_phone": row["phonemes"][i - 1]["phoneme"] if i else "sil",
                    "right_phone": row["phonemes"][i + 1]["phoneme"]
                    if i + 1 < len(row["phonemes"])
                    else "sil",
                }
                candidates[(speaker, name)].append(token)
        selected = []
        for tokens in candidates.values():
            seen_words = set()
            for token in sorted(tokens, key=lambda t: stable(t["id"])):
                if token["word"] in seen_words:
                    continue
                selected.append(token)
                seen_words.add(token["word"])
                if len(seen_words) >= PROTOCOL["max_tokens_per_phone_per_speaker"]:
                    break
        selected_ids = {t["utterance"] for t in selected}
        for row in rows:
            if row["id"] not in selected_ids:
                continue
            y, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
            if y.ndim != 1 or sr != 16000 or not np.isfinite(y).all():
                raise ValueError(f"Unexpected audio in {row['id']}")
            sf.write(BASE / "utterances" / f"{row['id']}.flac", y, sr)
            utterances[row["id"]] = {
                "transcript": row["transcript"],
                "duration": len(y) / sr,
                "words": row["words"],
                "phonemes": row["phonemes"],
            }
        samples.extend(selected)
        print(split, len(selected), "selected phone tokens", flush=True)
    dev_speakers = sorted({s["speaker"] for s in samples if s["source_split"] == "dev_clean"}, key=stable)
    test_speakers = {s["speaker"] for s in samples if s["source_split"] == "test_clean"}
    assert not test_speakers & set(dev_speakers)
    val_speakers = set(
        dev_speakers[: max(1, round(len(dev_speakers) * PROTOCOL["validation_speaker_fraction"]))]
    )
    for token in samples:
        token["split"] = (
            "test"
            if token["source_split"] == "test_clean"
            else "validation"
            if token["speaker"] in val_speakers
            else "train"
        )
        assert 0 <= token["start"] < token["end"] <= utterances[token["utterance"]]["duration"] + 0.01
    samples.sort(key=lambda s: s["id"])
    (BASE / "manifest.json").write_text(json.dumps(samples, indent=2))
    (BASE / "utterances.json").write_text(json.dumps(utterances))
    summary = {
        "protocol": PROTOCOL,
        "tokens": len(samples),
        "utterances": len(utterances),
        "speakers": {
            split: len({s["speaker"] for s in samples if s["split"] == split})
            for split in ("train", "validation", "test")
        },
        "counts": dict(Counter(s["split"] + "/" + s["phone"] for s in samples)),
        "rejected": dict(rejected),
    }
    (BASE / "data-summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    prepare()
