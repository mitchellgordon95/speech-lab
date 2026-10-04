"""Prepare Mandarin references with fixed-duration inputs and disjoint speakers."""

import hashlib
import io
import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import soundfile as sf
from huggingface_hub import hf_hub_download

from speechlab.config import DATA

BASE = DATA / "mandarin"
PROTOCOL = json.loads(Path(__file__).with_name("mandarin_protocol.json").read_text())


def stable(value):
    return hashlib.sha256(f"{PROTOCOL['seed']}:{value}".encode()).hexdigest()


def base_phone(value):
    # Preserve syllabic and nonsyllabic IPA diacritics; they denote different phones.
    return re.sub("[˥˦˧˨˩ːˑ˘]", "", value)


def prepare():
    import pyarrow.parquet as pq

    BASE.mkdir(exist_ok=True)
    if not (BASE / "source.json").exists():

        def download(index):
            return hf_hub_download(
                PROTOCOL["dataset"],
                f"data/train-{index:05d}-of-00005.parquet",
                revision=PROTOCOL["revision"],
                repo_type="dataset",
            )

        with ThreadPoolExecutor(max_workers=3) as pool:
            files = list(pool.map(download, range(5)))
        (BASE / "source.json").write_text(
            json.dumps({"dataset": PROTOCOL["dataset"], "revision": PROTOCOL["revision"], "files": files})
        )
    source = json.loads((BASE / "source.json").read_text())
    assert source["revision"] == PROTOCOL["revision"]
    alignment = hf_hub_download(
        "anyspeech/THCHS-30-alignments",
        "data/train-00000-of-00001-c7710d0536782c3f.parquet",
        revision=PROTOCOL["alignment_revision"],
        repo_type="dataset",
    )
    words = {r["id"]: r for r in pq.read_table(alignment).to_pylist()}
    allowed = {phone for phones in PROTOCOL["maps"].values() for phone in phones}
    candidates = defaultdict(list)
    rows = {}
    for file in source["files"]:
        for row in pq.read_table(
            file, columns=["utt_id", "text", "phones", "phone_starts", "phone_ends", "speaker_id", "duration"]
        ).to_pylist():
            ident = row["utt_id"]
            assert len(row["phones"]) == len(row["phone_starts"]) == len(row["phone_ends"])
            assert row["speaker_id"] == ident.split("_")[0]
            rows[ident] = {"text": row["text"], "duration": row["duration"]}
            w = words[ident]
            for index, (label, start, end) in enumerate(
                zip(row["phones"], row["phone_starts"], row["phone_ends"])
            ):
                phone = base_phone(label)
                if phone not in allowed or not 0.15999 <= end - start <= 0.40001:
                    continue
                matching = [
                    (a, b, text)
                    for a, b, text in zip(w["word_start"], w["word_end"], w["words"])
                    if a - 0.005 <= start and end <= b + 0.005 and text != "[SIL]"
                ]
                if len(matching) != 1 or not 0 <= start < end <= row["duration"] + 0.001:
                    continue
                a, b, text = matching[0]
                candidates[(row["speaker_id"], phone)].append(
                    {
                        "id": f"{ident}-{index}",
                        "utterance": ident,
                        "speaker": row["speaker_id"],
                        "phone": phone,
                        "original_label": label,
                        "start": start,
                        "end": end,
                        "word": text,
                        "word_start": a,
                        "word_end": b,
                    }
                )
    selected = []
    for values in candidates.values():
        used = set()
        for row in sorted(values, key=lambda r: stable(r["id"])):
            if row["word"] in used:
                continue
            selected.append(row)
            used.add(row["word"])
            if len(used) == 20:
                break
    speakers = sorted({s["speaker"] for s in selected}, key=stable)
    ntrain, nval = round(len(speakers) * 0.6), round(len(speakers) * 0.2)
    split = {
        s: "train" if i < ntrain else "validation" if i < ntrain + nval else "test"
        for i, s in enumerate(speakers)
    }
    for s in selected:
        s["split"] = split[s["speaker"]]
    selected.sort(key=lambda s: s["id"])
    needed = {s["utterance"] for s in selected}
    out = BASE / "utterances"
    out.mkdir(exist_ok=True)
    for file in source["files"]:
        # Bound decoded table memory while retaining the lossless original audio.
        for batch in pq.ParquetFile(file).iter_batches(batch_size=64, columns=["utt_id", "audio"]):
            for row in batch.to_pylist():
                if row["utt_id"] not in needed:
                    continue
                y, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
                assert sr == 16000 and y.ndim == 1
                sf.write(out / f"{row['utt_id']}.flac", y, sr, subtype="PCM_16")
    (BASE / "manifest.json").write_text(json.dumps(selected, ensure_ascii=False, indent=2))
    (BASE / "utterances.json").write_text(json.dumps({k: rows[k] for k in needed}, ensure_ascii=False))
    summary = {
        "tokens": len(selected),
        "speakers": {p: sum(v == p for v in split.values()) for p in ("train", "validation", "test")},
        "counts": dict(Counter(s["split"] + "/" + s["phone"] for s in selected)),
        "protocol": PROTOCOL,
    }
    (BASE / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    prepare()
