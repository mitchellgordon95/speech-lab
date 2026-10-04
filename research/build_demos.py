"""Package only contrasts that passed the frozen test and robustness gates."""

import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from speechlab import guided, models

from .evaluate import OUT
from .features import crop, utterance
from .prepare import BASE, PROTOCOL, stable

COPY = {
    "s_sh": {
        "name": "S & SH",
        "labels": ["S", "SH"],
        "words": ["see", "she"],
        "description": "Two kinds of hiss. See where your sound lands.",
        "instruction": "Hold ssss as in ‘see’, or shhhh as in ‘she’, for about a second. Leave off the vowel.",
        "category": "CONSONANTS",
    },
    "r_l": {
        "name": "R & L",
        "labels": ["R", "L"],
        "words": ["red", "led"],
        "description": "A voiced consonant contrast, across different voices.",
        "instruction": "Hold the English R at the start of ‘red’, or the L at the start of ‘led’, for about a second. Leave off the rest of the word.",
        "category": "CONSONANTS",
    },
    "iy_ih": {
        "name": "EE & IH",
        "labels": ["EE", "IH"],
        "words": ["sheep", "ship"],
        "description": "The vowel in ‘sheep’ and the vowel in ‘ship’.",
        "instruction": "Hold only the vowel from ‘sheep’ or ‘ship’ for about a second. Keep it steady.",
        "category": "VOWELS",
    },
    "uw_uh": {
        "name": "OO & UH",
        "labels": ["OO", "UH"],
        "words": ["food", "foot"],
        "description": "The vowel in ‘food’ and the vowel in ‘foot’.",
        "instruction": "Hold only the vowel from ‘food’ or ‘foot’ for about a second. Keep it steady.",
        "category": "VOWELS",
    },
    "ae_eh": {
        "name": "A & EH",
        "labels": ["A", "EH"],
        "words": ["bad", "bed"],
        "description": "The vowel in ‘bad’ and the vowel in ‘bed’.",
        "instruction": "Hold only the vowel from ‘bad’ or ‘bed’ for about a second. Keep it steady.",
        "category": "VOWELS",
    },
}
WORDS = {
    "S": "see said say so some sun soon saw seemed seven",
    "SH": "she shall ship shop show shore shoes should",
    "R": "red right road room rain run read really river round rose",
    "L": "let long look light little love left last led like live",
    "IY": "see feel keep sheep he me she feet seem sea deep",
    "IH": "sit bit ship little live give him his did lips with",
    "AE": "cat back had man bad hand that black",
    "EH": "bed red said head let dead men get",
    "UW": "food soon move who blue too two new true",
    "UH": "good could should look book stood took wood",
}


def build():
    report = json.loads((OUT / "benchmark.json").read_text())
    robustness = json.loads((OUT / "robustness.json").read_text())
    samples = json.loads((BASE / "manifest.json").read_text())
    acoustic = np.load(BASE / "features/acoustic.npz")["x"]
    assets = guided.ASSETS
    (assets / "audio").mkdir(parents=True, exist_ok=True)
    contrasts, attribution = [], []
    for ident, copy in COPY.items():
        if not robustness[ident]["demo_eligible"] or len(contrasts) >= 3:
            continue
        r = report["contrasts"][ident]
        name = r["selected_representation"]
        if name not in models.MODELS and name not in ("formants", "spectrum", "acoustic", "sparc"):
            continue
        fitted = json.loads((BASE / "probes" / f"{ident}.json").read_text())
        fitted["encoder"] = models.MODELS.get(name)
        (assets / f"{ident}.json").write_text(json.dumps(fitted, allow_nan=False))
        sides = []
        for phone in r["phones"]:
            candidates = [
                s
                for s in samples
                if s["phone"] == phone
                and s["split"] == "test"
                and s["word"] in WORDS[phone].split()
                and (phone not in ("S", "SH", "R", "L") or s["start"] - s["word_start"] < 0.015)
            ]
            # Word/speaker diversity and deterministic order, never classifier correctness.
            candidates.sort(key=lambda s: stable(s["id"]))
            chosen, used_speakers, used_words = [], set(), set()
            for s in candidates:
                if s["speaker"] in used_speakers or s["word"] in used_words:
                    continue
                chosen.append(s)
                used_speakers.add(s["speaker"])
                used_words.add(s["word"])
                if len(chosen) == 3:
                    break
            if len(chosen) < 2:
                raise ValueError(f"Too few reference speakers for {phone}")
            examples = []
            for s in chosen:
                prefix = f"{ident}-{s['id']}"
                wave = utterance(s["utterance"])
                start = max(0, s["word_start"] - 0.025)
                end = min(len(wave) / 16000, s["word_end"] + 0.025)
                word_wave = wave[round(start * 16000) : round(end * 16000)]
                phone_wave = crop(s)
                # Lossless PCM16 crop; no synthesis, time stretching, or pitch changes.
                for kind, y in (("word", word_wave), ("phone", phone_wave)):
                    sf.write(assets / "audio" / f"{prefix}-{kind}.flac", y, 16000, subtype="PCM_16")
                item = {
                    "word": s["word"],
                    "phone": phone,
                    "speaker": s["speaker"],
                    "utterance": s["utterance"],
                    "token": s["id"],
                    "phone_start": s["start"],
                    "phone_end": s["end"],
                    "word_crop_start": start,
                    "word_crop_end": end,
                    "phone_in_word": [s["start"] - start, s["end"] - start],
                    "word_file": f"{prefix}-word.flac",
                    "phone_file": f"{prefix}-phone.flac",
                    "source_split": s["source_split"],
                }
                examples.append(item)
                attribution.append(item)
            sides.append(examples)
        c = {
            "id": ident,
            **copy,
            "phones": r["phones"],
            "model": name,
            "test": r["representations"][name],
            "examples": sides,
            "robustness": robustness[ident]["conditions"],
        }
        if ident == "s_sh":
            c["spectral_medians_khz"] = [
                round(
                    float(
                        np.median(
                            acoustic[
                                [
                                    i
                                    for i, s in enumerate(samples)
                                    if s["phone"] == p and s["split"] != "test"
                                ],
                                3,
                            ]
                        )
                    )
                    / 1000,
                    2,
                )
                for p in r["phones"]
            ]
        contrasts.append(c)
    results = []
    for ident, r in report["contrasts"].items():
        name = r["selected_representation"]
        results.append(
            {
                "contrast": ident,
                "phones": r["phones"],
                "model": name,
                "test": r["representations"][name],
                "representations": r["representations"],
                "eligible": robustness[ident]["demo_eligible"],
                "conditions": robustness[ident]["conditions"],
            }
        )
    payload = {
        "version": 1,
        "contrasts": contrasts,
        "results": results,
        "adaptation_control": json.loads((OUT / "adaptation-control.json").read_text()),
        "tokens": report["data"]["tokens"],
        "test_speakers": report["data"]["speakers"]["test"],
        "source": "https://www.openslr.org/12",
        "license": "https://creativecommons.org/licenses/by/4.0/",
        "alignment_source": "https://zenodo.org/records/2619474",
        "protocol_sha256": hashlib.sha256(Path(__file__).with_name("protocol.json").read_bytes()).hexdigest(),
    }
    (assets / "catalog.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    (assets / "attribution.json").write_text(
        json.dumps(
            {
                "corpus": "LibriSpeech, Vassil Panayotov, Guoguo Chen, Daniel Povey, Sanjeev Khudanpur; LibriVox readers",
                "source": payload["source"],
                "license": payload["license"],
                "alignments": "Loren Lugosch / Montreal Forced Aligner; packaging by Kim Gilkey",
                "alignment_source": payload["alignment_source"],
                "dataset_revision": PROTOCOL["revision"],
                "dataset": "https://huggingface.co/datasets/gilkeyio/librispeech-alignments",
                "changes": "Word and phone crops, encoded as lossless 16 kHz PCM16 FLAC. No synthetic speech.",
                "clips": attribution,
            },
            indent=2,
        )
        + "\n"
    )
    print("Packaged:", [c["id"] for c in contrasts])


if __name__ == "__main__":
    build()
