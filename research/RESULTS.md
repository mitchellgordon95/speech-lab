# Completed English speech study

Executed locally on the M5 Pro Mac. These are measured results, not proposed experiments.

10,676 phone segments from 4,019 utterances and 80 speakers: 30 training, 10 validation, 40 final-test speakers. The final test speakers were excluded from fitting and model selection. Frozen encoder pretraining may have included this corpus; this holdout claim applies only to the linear probes.

The corpus contains correct read English, not learner errors or intelligibility judgments. We test sound-category discrimination. A useful category direction is not yet a validated pronunciation coach.

## Selected models

Representation selection used validation speakers only. The selected models were then refit on all 40 development speakers and evaluated once on the final test. No failed winner was replaced using test results.

| Contrast | Selected representation | Validation BA | Test BA | 95% speaker-bootstrap interval | Test clips | All demo gates |
|---|---|---:|---:|---|---:|---|
| S / SH | qwen-asr-large | 99.0% | 97.6% | 96.2%–98.8% | 799 | Pass |
| R / L | qwen-omni | 100.0% | 99.1% | 98.2%–99.8% | 800 | Pass |
| F / TH | qwen-asr-large | 83.4% | 78.9% | 76.3%–81.4% | 788 | Withheld |
| B / P | wavlm-large | 88.4% | 86.0% | 82.5%–89.1% | 787 | Pass |
| IY / IH | qwen-omni | 94.0% | 92.2% | 90.0%–94.1% | 800 | Pass |
| AE / EH | qwen-asr | 86.5% | 83.2% | 79.9%–86.2% | 800 | Withheld |
| UW / UH | qwen-asr-large | 95.4% | 90.7% | 88.0%–93.4% | 558 | Pass |

BA = balanced accuracy: average recall of the two categories. Chance is 50%. Intervals use 500 bootstrap resamples of complete speakers. These are individual descriptive intervals, not multiple-comparison-adjusted significance tests.

The frozen gate requires validation and test BA ≥85%, final-test interval lower bound ≥80%, and BA ≥80% for 20 dB noise, both ±20 ms shifts, and central crops up to 160 ms. The guided homepage exposes at most three sustained-sound contrasts, in the preset order S/SH, R/L, IY/IH, UW/UH, AE/EH. B/P requires different elicitation and was not a candidate for sustained-sound practice.

## Every representation

All rows use the same fixed C=0.1, class-balanced linear classifier. These final-test scores are descriptive; only validation scores determined selection. We did not search model layers or tune hyperparameters on test data.

| Representation | S/SH | R/L | F/TH | B/P | IY/IH | AE/EH | UW/UH |
|---|---:|---:|---:|---:|---:|---:|---:|
| formants | 52.4% | 87.8% | 45.1% | 59.4% | 82.5% | 68.9% | 76.4% |
| spectrum | 87.7% | 89.4% | 67.6% | 67.8% | 76.5% | 58.0% | 72.8% |
| acoustic | 87.5% | 92.8% | 67.2% | 76.2% | 84.8% | 68.2% | 78.4% |
| timing_pitch_level | 63.3% | 52.2% | 59.5% | 74.7% | 62.3% | 59.2% | 59.4% |
| wavlm-base | 91.5% | 94.2% | 69.9% | 83.6% | 89.1% | 75.4% | 86.5% |
| wavlm-large | 95.9% | 97.4% | 77.3% | 86.0% | 90.9% | 79.0% | 92.2% |
| xls-r | 96.5% | 97.5% | 77.4% | 86.2% | 92.9% | 80.9% | 92.0% |
| qwen-asr | 98.1% | 98.1% | 79.4% | 84.0% | 93.4% | 83.2% | 92.8% |
| qwen-asr-large | 97.6% | 98.8% | 78.9% | 86.6% | 92.8% | 84.6% | 90.7% |
| qwen-omni | 97.0% | 99.1% | 82.3% | 87.4% | 92.2% | 83.6% | 92.0% |
| sparc | 87.5% | 90.2% | 69.6% | 78.6% | 85.9% | 71.4% | 84.5% |

Formants = median voiced F1–F3. Spectrum = eight spectral/zero-crossing features. Acoustics = those eleven plus pitch, voicing fraction, duration and log RMS. `timing_pitch_level` is a diagnostic nuisance baseline, not a candidate for promotion. SPARC is a pretrained estimate of articulation, not observed tongue motion. Its discrimination accuracy does not validate anatomical accuracy.

All encoders receive the exact phone crop, demeaned, RMS-normalized, faded 5 ms at the edges, centered in 0.5 s of silence, then pooled into four ordered temporal means. This removes neighboring speech outside the crop, but not coarticulation inside it. Layers and pinned model revisions are in `speechlab/models.py`.

## Distance and speaker adaptation

The following use each contrast's validation-selected representation. Neural/SPARC centroid comparisons use raw cosine distance; acoustic comparisons use training-imputed, training-standardized features. Triplets compare a same-phone/different-speaker sample against a different-phone/same-speaker sample, 1,000 random anchor draws per contrast. This is a diagnostic, not 1,000 independent speakers.

| Contrast | Learned direction BA | Population centroid BA | Triplet success | Personal centroid BA* | Other-speaker centroid BA* | Adaptation speakers | Adaptation queries |
|---|---:|---:|---:|---:|---:|---:|---:|
| S/SH | 97.6% | 94.0% | 46.4% | 68.3% | 93.7% | 40 | 622 |
| R/L | 99.1% | 94.4% | 29.6% | 70.3% | 95.3% | 40 | 611 |
| F/TH | 78.9% | 61.4% | 42.6% | 59.7% | 72.4% | 40 | 610 |
| B/P | 86.0% | 78.2% | 52.0% | 64.6% | 78.5% | 40 | 608 |
| IY/IH | 92.2% | 88.2% | 26.9% | 65.2% | 87.1% | 40 | 623 |
| AE/EH | 83.2% | 74.4% | 37.7% | 52.9% | 74.6% | 40 | 624 |
| UW/UH | 90.7% | 85.3% | 34.9% | 59.4% | 83.2% | 15 | 179 |

*Separate exploratory adaptation evaluation: reserve two known-correct examples per category from each eligible test speaker; query different utterances from that speaker. Compare personal centroids with centroids from the other test speakers. Values are means of per-speaker balanced accuracies, with different queries from the main evaluation. They are not directly comparable to the main BA. This experiment assumes the learner can supply correct examples; it does not establish unsupervised voice normalization or adaptation for an unproducible sound.

## Robustness of the frozen selected models

80 deterministic test clips per sound, 160 per contrast. Boundary shifts translate the whole crop by ±20 ms, retaining duration except at an utterance edge. Noise is additive Gaussian noise at 20 dB signal-to-noise ratio. Central cropping retains at most 160 ms, leaving naturally shorter phones unchanged.

| Contrast | Original subset | Quarter volume | 20 dB noise | −20 ms | +20 ms | Central ≤160 ms |
|---|---:|---:|---:|---:|---:|---:|
| S/SH | 98.1% | 98.1% | 98.8% | 98.8% | 98.1% | 98.1% |
| R/L | 99.4% | 99.4% | 97.5% | 97.5% | 98.8% | 99.4% |
| F/TH | 78.1% | 78.1% | 75.0% | 76.2% | 80.6% | 78.8% |
| B/P | 92.5% | 92.5% | 85.0% | 81.2% | 88.8% | 91.9% |
| IY/IH | 94.4% | 94.4% | 93.8% | 93.1% | 93.8% | 95.0% |
| AE/EH | 83.1% | 83.1% | 87.5% | 78.8% | 80.6% | 82.5% |
| UW/UH | 91.9% | 91.9% | 85.0% | 93.1% | 93.1% | 91.9% |

## Limits and unrun experiments

- Word/phone timings are automatic Montreal Forced Aligner output, not manually verified timestamp ground truth. Boundary perturbation tests robustness, not timing correctness. Audio labels assume correct read English.
- Tokens are 80–400 ms; vowels have primary stress. Results do not cover all natural instances of each phone. Word types are unique within phone/speaker but can recur across speakers. Unseen-word and dataset-provided sex-group breakdowns are included in `results/benchmark.json`.
- No Mandarin transfer, intelligibility ratings, learner improvement, hearing-loss rehabilitation, or tongue-position accuracy was measured. Binary models can place an unrelated third sound on either side. Marker positions are sigmoid-transformed margins, not calibrated correctness probabilities.
- Microphone mode uses energy-selected windows of a held sound; natural word phones and sustained learner productions are different domains. The UI's silence/clipping/voicing/consistency checks are heuristics, not a validated open-set error detector. End-to-end browser checks verify software and audio capture, not pedagogy.
- Manual VOT landmarks are not a completed automatic VOT estimator; this corpus does not provide hand-labeled release/voicing landmarks. B/P discrimination is reported without claiming a VOT measurement.
- Paid Azure, SpeechSuper and Qwen assessor APIs were not run because no credentials were supplied. No cloud outputs were used to fit or evaluate another engine. The Qwen experiments here are frozen local audio encoders, not verbal judgments from their language decoders.

## Data, credits, reproduction

[LibriSpeech / OpenSLR](https://www.openslr.org/12), by Vassil Panayotov, Guoguo Chen, Daniel Povey and Sanjeev Khudanpur, derives from LibriVox audiobook readings and is distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). [Loren Lugosch's alignments](https://zenodo.org/records/2619474) use Montreal Forced Aligner. [Kim Gilkey's packaging](https://huggingface.co/datasets/gilkeyio/librispeech-alignments) is pinned to revision `0daa1eb43dda38ee6ce752e785555380e5628f5c`.

See [the executable protocol and reproduction steps](README.md), `results/selection.json` (frozen before final test), `results/benchmark.json`, `results/robustness.json`, and `results/run.json`. Source audio, feature caches and personal recordings are excluded from Git. Guided examples have per-clip attribution in `speechlab/guided_assets/attribution.json`.

## Exploratory equal-size adaptation control

Added after inspecting the main results, without changing any selected model, gate, or score. The original two-personal-vs-many-other comparison mixes reference count with speaker matching. Here both methods get two known-correct anchors per category, with 30 random draws per eligible speaker and paired query utterances. Both methods use the same raw cosine-centroid rule. Different query sets mean these absolute scores should not be compared directly with the main classifier BA.

| Contrast | Personal references | Equal-size other-speaker references | Paired advantage | 95% paired speaker interval | Speakers |
|---|---:|---:|---:|---|---:|
| S/SH | 67.1% | 61.4% | +5.7 pp | +3.5 to +7.8 pp | 40 |
| R/L | 70.8% | 63.7% | +7.1 pp | +5.0 to +9.2 pp | 40 |
| F/TH | 56.3% | 53.3% | +3.0 pp | +0.5 to +5.7 pp | 40 |
| B/P | 64.8% | 61.2% | +3.6 pp | +1.8 to +5.4 pp | 40 |
| IY/IH | 65.2% | 60.1% | +5.1 pp | +2.9 to +7.3 pp | 40 |
| AE/EH | 54.0% | 53.7% | +0.3 pp | -1.4 to +1.9 pp | 40 |
| UW/UH | 61.8% | 58.4% | +3.3 pp | -1.3 to +8.3 pp | 15 |

Matching the speaker helps several contrasts when reference counts are equal, but this sparse-centroid approach still performs poorly. This supports investigating adaptation; it does not justify replacing the already fitted population classifier or asking the learner for correct Mandarin anchors. See `results/adaptation-control.json` for the complete exploratory protocol.
