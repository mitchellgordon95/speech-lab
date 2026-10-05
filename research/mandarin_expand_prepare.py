"""Prepare every Mandarin initial, final ng, and the existing six vowels."""

import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
from huggingface_hub import hf_hub_download

from speechlab.config import DATA

from .mandarin_features import crop, utterance
from .mandarin_prepare import BASE, base_phone
from .mandarin_prepare import PROTOCOL as SOURCE

PROTOCOL = json.loads(Path(__file__).with_name("mandarin_expanded_protocol.json").read_text())
OUT = DATA / "mandarin-expanded"


def stable(value):
    return hashlib.sha256(f"{PROTOCOL['seed']}:{value}".encode()).hexdigest()


def prepare():
    import pyarrow.parquet as pq

    OUT.mkdir(exist_ok=True)
    source = json.loads((BASE / "source.json").read_text())
    old = json.loads((BASE / "manifest.json").read_text())
    splits = {r["speaker"]: r["split"] for r in old}
    path = hf_hub_download(
        "anyspeech/THCHS-30-alignments",
        "data/train-00000-of-00001-c7710d0536782c3f.parquet",
        revision=SOURCE["alignment_revision"],
        repo_type="dataset",
    )
    words = {r["id"]: r for r in pq.read_table(path).to_pylist()}
    candidates = defaultdict(list)
    for file in source["files"]:
        for row in pq.read_table(
            file, columns=["utt_id", "speaker_id", "phones", "phone_starts", "phone_ends", "duration"]
        ).to_pylist():
            w = words[row["utt_id"]]
            for index, (label, start, end) in enumerate(
                zip(row["phones"], row["phone_starts"], row["phone_ends"])
            ):
                phone = base_phone(label)
                lower = 0.15999 if phone in PROTOCOL["vowels"] else 0.03999
                mid = (start + end) / 2
                if (
                    phone not in PROTOCOL["phones"]
                    or not lower <= end - start <= 0.40001
                    or not 0.12 <= mid <= row["duration"] - 0.12
                ):
                    continue
                match = [
                    (a, b, word)
                    for a, b, word in zip(w["word_start"], w["word_end"], w["words"])
                    if a - 0.005 <= start and end <= b + 0.005 and word != "[SIL]"
                ]
                if len(match) != 1:
                    continue
                a, b, word = match[0]
                candidates[(row["speaker_id"], phone)].append(
                    dict(
                        id=f"{row['utt_id']}-{index}",
                        utterance=row["utt_id"],
                        speaker=row["speaker_id"],
                        split=splits[row["speaker_id"]],
                        phone=phone,
                        original_label=label,
                        start=start,
                        end=end,
                        word=word,
                        word_start=a,
                        word_end=b,
                        previous_phone=base_phone(row["phones"][index - 1]) if index else None,
                        next_phone=base_phone(row["phones"][index + 1])
                        if index + 1 < len(row["phones"])
                        else None,
                        phone_fraction=min(1.0, (end - start) / 0.16),
                    )
                )
    shortlist = {}
    for key, rows in candidates.items():
        seen = set()
        keep = []
        for row in sorted(rows, key=lambda r: stable(r["id"])):
            if row["word"] in seen:
                continue
            seen.add(row["word"])
            keep.append(row)
            if len(keep) == 30:
                break
        shortlist[key] = keep
    needed = {r["utterance"] for rows in shortlist.values() for r in rows}
    for file in source["files"]:
        for batch in pq.ParquetFile(file).iter_batches(batch_size=64, columns=["utt_id", "audio"]):
            for row in batch.to_pylist():
                out = BASE / "utterances" / f"{row['utt_id']}.flac"
                if row["utt_id"] not in needed or out.exists():
                    continue
                y, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
                assert sr == 16000 and y.ndim == 1
                sf.write(out, y, sr, subtype="PCM_16")
        print("Audio shard ready", Path(file).name, flush=True)
    selected = []
    rejected = Counter()
    for rows in shortlist.values():
        count = 0
        for row in rows:
            y = crop(row)
            wave = utterance(row["utterance"])
            a = max(0, round((row["word_start"] - 0.025) * 16000))
            b = min(len(wave), round((row["word_end"] + 0.025) * 16000))
            word = wave[a:b]
            if np.mean(np.abs(y) > 0.99) > 0.01 or np.mean(np.abs(word) > 0.99) > 0.01:
                rejected["clipping"] += 1
                continue
            if np.sqrt(np.mean((y - y.mean()) ** 2)) < 0.002:
                rejected["quiet"] += 1
                continue
            selected.append(row)
            count += 1
            if count == 10:
                break
    selected.sort(key=lambda r: r["id"])
    (OUT / "manifest.json").write_text(json.dumps(selected, ensure_ascii=False, indent=2))
    summary = dict(
        tokens=len(selected),
        speakers={
            split: len({r["speaker"] for r in selected if r["split"] == split})
            for split in ["train", "validation", "test"]
        },
        counts=dict(Counter(r["split"] + "/" + r["phone"] for r in selected)),
        quality_rejections=dict(rejected),
        protocol=PROTOCOL,
    )
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    prepare()
