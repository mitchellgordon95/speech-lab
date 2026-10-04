"""Render the completed measurements; keep interpretation separate from model selection."""

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import transformers

from speechlab import models

from .evaluate import OUT
from .prepare import BASE


def pct(value):
    return f"{value * 100:.1f}%"


def write():
    benchmark = json.loads((OUT / "benchmark.json").read_text())
    selection = json.loads((OUT / "selection.json").read_text())
    robustness = json.loads((OUT / "robustness.json").read_text())
    rows = benchmark["contrasts"]
    lines = [
        "# Completed English speech study",
        "",
        "Executed locally on the M5 Pro Mac. These are measured results, not proposed experiments.",
        "",
        "10,676 phone segments from 4,019 utterances and 80 speakers: 30 training, 10 validation, 40 final-test speakers. The final test speakers were excluded from fitting and model selection. Frozen encoder pretraining may have included this corpus; this holdout claim applies only to the linear probes.",
        "",
        "The corpus contains correct read English, not learner errors or intelligibility judgments. We test sound-category discrimination. A useful category direction is not yet a validated pronunciation coach.",
        "",
        "## Selected models",
        "",
        "Representation selection used validation speakers only. The selected models were then refit on all 40 development speakers and evaluated once on the final test. No failed winner was replaced using test results.",
        "",
        "| Contrast | Selected representation | Validation BA | Test BA | 95% speaker-bootstrap interval | Test clips | All demo gates |",
        "|---|---|---:|---:|---|---:|---|",
    ]
    for contrast, r in rows.items():
        name = r["selected_representation"]
        s = r["representations"][name]
        ci = "–".join(pct(v) for v in s["speaker_bootstrap_95_ci"])
        va = selection["selected"][contrast]["validation"]["balanced_accuracy"]
        passed = "Pass" if robustness[contrast]["demo_eligible"] else "Withheld"
        lines.append(
            f"| {' / '.join(r['phones'])} | {name} | {pct(va)} | {pct(s['balanced_accuracy'])} | {ci} | {s['n']} | {passed} |"
        )
    lines += [
        "",
        "BA = balanced accuracy: average recall of the two categories. Chance is 50%. Intervals use 500 bootstrap resamples of complete speakers. These are individual descriptive intervals, not multiple-comparison-adjusted significance tests.",
        "",
        "The frozen gate requires validation and test BA ≥85%, final-test interval lower bound ≥80%, and BA ≥80% for 20 dB noise, both ±20 ms shifts, and central crops up to 160 ms. The guided homepage exposes at most three sustained-sound contrasts, in the preset order S/SH, R/L, IY/IH, UW/UH, AE/EH. B/P requires different elicitation and was not a candidate for sustained-sound practice.",
        "",
        "## Every representation",
        "",
        "All rows use the same fixed C=0.1, class-balanced linear classifier. These final-test scores are descriptive; only validation scores determined selection. We did not search model layers or tune hyperparameters on test data.",
        "",
        "| Representation | " + " | ".join("/".join(r["phones"]) for r in rows.values()) + " |",
        "|---|" + "---:|" * len(rows),
    ]
    for name in next(iter(rows.values()))["representations"]:
        lines.append(
            "| "
            + name
            + " | "
            + " | ".join(pct(r["representations"][name]["balanced_accuracy"]) for r in rows.values())
            + " |"
        )
    lines += [
        "",
        "Formants = median voiced F1–F3. Spectrum = eight spectral/zero-crossing features. Acoustics = those eleven plus pitch, voicing fraction, duration and log RMS. `timing_pitch_level` is a diagnostic nuisance baseline, not a candidate for promotion. SPARC is a pretrained estimate of articulation, not observed tongue motion. Its discrimination accuracy does not validate anatomical accuracy.",
        "",
        "All encoders receive the exact phone crop, demeaned, RMS-normalized, faded 5 ms at the edges, centered in 0.5 s of silence, then pooled into four ordered temporal means. This removes neighboring speech outside the crop, but not coarticulation inside it. Layers and pinned model revisions are in `speechlab/models.py`.",
        "",
        "## Distance and speaker adaptation",
        "",
        "The following use each contrast's validation-selected representation. Neural/SPARC centroid comparisons use raw cosine distance; acoustic comparisons use training-imputed, training-standardized features. Triplets compare a same-phone/different-speaker sample against a different-phone/same-speaker sample, 1,000 random anchor draws per contrast. This is a diagnostic, not 1,000 independent speakers.",
        "",
        "| Contrast | Learned direction BA | Population centroid BA | Triplet success | Personal centroid BA* | Other-speaker centroid BA* | Adaptation speakers | Adaptation queries |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows.values():
        s = r["representations"][r["selected_representation"]]
        p = s["personalization"]
        lines.append(
            f"| {'/'.join(r['phones'])} | {pct(s['balanced_accuracy'])} | {pct(s['centroid']['balanced_accuracy'])} | {pct(s['triplets']['same_phone_other_speaker_closer'])} | {pct(p['personal_mean_balanced_accuracy'])} | {pct(p['other_speakers_mean_balanced_accuracy'])} | {p['speakers']} | {p['queries']} |"
        )
    lines += [
        "",
        "*Separate exploratory adaptation evaluation: reserve two known-correct examples per category from each eligible test speaker; query different utterances from that speaker. Compare personal centroids with centroids from the other test speakers. Values are means of per-speaker balanced accuracies, with different queries from the main evaluation. They are not directly comparable to the main BA. This experiment assumes the learner can supply correct examples; it does not establish unsupervised voice normalization or adaptation for an unproducible sound.",
        "",
        "## Robustness of the frozen selected models",
        "",
        "80 deterministic test clips per sound, 160 per contrast. Boundary shifts translate the whole crop by ±20 ms, retaining duration except at an utterance edge. Noise is additive Gaussian noise at 20 dB signal-to-noise ratio. Central cropping retains at most 160 ms, leaving naturally shorter phones unchanged.",
        "",
        "| Contrast | Original subset | Quarter volume | 20 dB noise | −20 ms | +20 ms | Central ≤160 ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for contrast, r in rows.items():
        s = robustness[contrast]["conditions"]
        lines.append(
            "| "
            + "/".join(r["phones"])
            + " | "
            + " | ".join(
                pct(s[k]["balanced_accuracy"])
                for k in (
                    "baseline_subset",
                    "gain_quarter",
                    "noise_20db",
                    "boundary_minus_20ms",
                    "boundary_plus_20ms",
                    "central_160ms",
                )
            )
            + " |"
        )
    lines += [
        "",
        "## Limits and unrun experiments",
        "",
        "- Word/phone timings are automatic Montreal Forced Aligner output, not manually verified timestamp ground truth. Boundary perturbation tests robustness, not timing correctness. Audio labels assume correct read English.",
        "- Tokens are 80–400 ms; vowels have primary stress. Results do not cover all natural instances of each phone. Word types are unique within phone/speaker but can recur across speakers. Unseen-word and dataset-provided sex-group breakdowns are included in `results/benchmark.json`.",
        "- No Mandarin transfer, intelligibility ratings, learner improvement, hearing-loss rehabilitation, or tongue-position accuracy was measured. Binary models can place an unrelated third sound on either side. Marker positions are sigmoid-transformed margins, not calibrated correctness probabilities.",
        "- Microphone mode uses energy-selected windows of a held sound; natural word phones and sustained learner productions are different domains. The UI's silence/clipping/voicing/consistency checks are heuristics, not a validated open-set error detector. End-to-end browser checks verify software and audio capture, not pedagogy.",
        "- Manual VOT landmarks are not a completed automatic VOT estimator; this corpus does not provide hand-labeled release/voicing landmarks. B/P discrimination is reported without claiming a VOT measurement.",
        "- Paid Azure, SpeechSuper and Qwen assessor APIs were not run because no credentials were supplied. No cloud outputs were used to fit or evaluate another engine. The Qwen experiments here are frozen local audio encoders, not verbal judgments from their language decoders.",
        "",
        "## Data, credits, reproduction",
        "",
        "[LibriSpeech / OpenSLR](https://www.openslr.org/12), by Vassil Panayotov, Guoguo Chen, Daniel Povey and Sanjeev Khudanpur, derives from LibriVox audiobook readings and is distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). [Loren Lugosch's alignments](https://zenodo.org/records/2619474) use Montreal Forced Aligner. [Kim Gilkey's packaging](https://huggingface.co/datasets/gilkeyio/librispeech-alignments) is pinned to revision `0daa1eb43dda38ee6ce752e785555380e5628f5c`.",
        "",
        "See [the executable protocol and reproduction steps](README.md), `results/selection.json` (frozen before final test), `results/benchmark.json`, `results/robustness.json`, and `results/run.json`. Source audio, feature caches and personal recordings are excluded from Git. Guided examples have per-clip attribution in `speechlab/guided_assets/attribution.json`.",
    ]
    control = json.loads((OUT / "adaptation-control.json").read_text())
    lines += [
        "",
        "## Exploratory equal-size adaptation control",
        "",
        "Added after inspecting the main results, without changing any selected model, gate, or score. The original two-personal-vs-many-other comparison mixes reference count with speaker matching. Here both methods get two known-correct anchors per category, with 30 random draws per eligible speaker and paired query utterances. Both methods use the same raw cosine-centroid rule. Different query sets mean these absolute scores should not be compared directly with the main classifier BA.",
        "",
        "| Contrast | Personal references | Equal-size other-speaker references | Paired advantage | 95% paired speaker interval | Speakers |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for contrast, c in control["results"].items():
        ci = " to ".join(f"{v * 100:+.1f}" for v in c["paired_speaker_bootstrap_95_ci"])
        lines.append(
            f"| {'/'.join(rows[contrast]['phones'])} | {pct(c['personal_mean_balanced_accuracy'])} | {pct(c['equal_size_other_speakers_mean_balanced_accuracy'])} | {c['paired_personal_minus_other'] * 100:+.1f} pp | {ci} pp | {c['speakers']} |"
        )
    lines += [
        "",
        "Matching the speaker helps several contrasts when reference counts are equal, but this sparse-centroid approach still performs poorly. This supports investigating adaptation; it does not justify replacing the already fitted population classifier or asking the learner for correct Mandarin anchors. See `results/adaptation-control.json` for the complete exploratory protocol.",
    ]
    Path(__file__).with_name("RESULTS.md").write_text("\n".join(lines) + "\n")
    dimensions = {}
    for file in (BASE / "features").glob("*.npz"):
        if "partial" not in file.name:
            with np.load(file) as data:
                dimensions[file.stem] = list(data["x"].shape)
    (OUT / "run.json").write_text(
        json.dumps(
            {
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "platform": platform.platform(),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "device": models.device(),
                "manifest_sha256": hashlib.sha256((BASE / "manifest.json").read_bytes()).hexdigest(),
                "feature_shapes": dimensions,
                "protocol_sha256": hashlib.sha256(
                    Path(__file__).with_name("protocol.json").read_bytes()
                ).hexdigest(),
                "model_specs": models.MODELS,
            },
            indent=2,
        )
        + "\n"
    )
    print("Wrote research/RESULTS.md and results/run.json")


if __name__ == "__main__":
    write()
