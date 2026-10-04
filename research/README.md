# Corpus study

This replaces manual user-led experiments with an executed benchmark and fitted demo models. The plan is frozen in `protocol.json` before final-test evaluation. Corpora, feature caches and intermediate models stay in ignored `data/research/`.

Read [the completed results](RESULTS.md). The user-facing application loads committed fitted probes and example crops; none of the steps below are needed to use it.

## Reproduce

```sh
.bootstrap/bin/uv sync --group dev --group research
.venv/bin/python -m research.prepare
HF_HUB_OFFLINE=1 .venv/bin/python -m research.features
.venv/bin/python -m research.evaluate
HF_HUB_OFFLINE=1 .venv/bin/python -m research.robustness
.venv/bin/python -m research.adaptation
.venv/bin/python -m research.build_demos
.venv/bin/python -m research.write_report
```

Download each encoder once using the workbench benchmark before offline feature extraction. `prepare` downloads only LibriSpeech dev-clean and test-clean from a pinned revision of [Kim Gilkey's packaging](https://huggingface.co/datasets/gilkeyio/librispeech-alignments). [LibriSpeech](https://www.openslr.org/12) is read English derived from LibriVox audiobooks, licensed CC BY 4.0. [Loren Lugosch's alignments](https://zenodo.org/records/2619474) were generated with Montreal Forced Aligner. Audio/caption pairs are real human speech; phoneme boundaries are automatic, not manually verified.

## Design

- 30 development speakers train probes; 10 separate speakers select the representation. All 40 test-clean speakers are held out for final evaluation of the fitted probes. Encoder pretraining exposure is unknown; this is not a claim of zero prior corpus exposure.
- Seven contrasts: S/SH, R/L, F/TH, B/P, stressed IY/IH, AE/EH, UW/UH. Select at most ten tokens per phone per speaker, with unique word types within each phone/speaker. Remove unknown words, invalid durations and word-boundary crossings. Selection uses deterministic hashes, not model scores.
- Each input is just the timestamped phone, with no neighboring speech. Encoder input retains duration, normalizes level, tapers crop edges, and centers the phone in 0.5 seconds of zero padding. Four temporal-bin means preserve coarse ordering. This removes lexical context outside the phone but does not remove coarticulation inside it.
- Compare formants, spectral measurements, combined acoustics, all six frozen encoders, SPARC estimated articulation, and a diagnostic timing/pitch/level baseline. Every probe uses the same fixed regularization; preprocessing is learned on training data only.
- Save the validation-only selection before looking at final-test scores. Refit chosen probes on development speakers, then report final balanced accuracy, AUC, confusion matrices, speaker-bootstrap intervals, unseen-word subsets, and dataset-provided sex groups.
- Compare supervised probes with cosine nearest-centroid decisions. A triplet diagnostic asks whether the same phone in another speaker is closer than the other phone in the same speaker. Personal-reference comparisons reserve two accepted examples per class from a test speaker and query distinct utterances. This tests reference adaptation, not learning a new sound without correct personal examples.
- Perturb frozen models' inputs with quarter gain, 20 dB noise, +/-20 ms boundary shifts, and central crops of at most 160 ms. Fragile winners are withheld instead of being replaced using test performance.

The corpus tests sound discrimination and repeatability. It cannot establish learner improvement, human intelligibility judgments, Mandarin transfer, or true articulator positions. Paid cloud assessors are not run because no credentials were supplied. The new demos expose only supported English contrasts with pre-fitted models, native recordings, and a record/play interaction.

`adaptation.py` is an explicitly exploratory control added after the main test: both personal and other-speaker methods get two accepted anchors per category, with repeated paired queries. It addresses the unequal reference counts in the original adaptation comparison. It has no effect on model selection or promotion.

## Mandarin live maps

The separate [Mandarin study](MANDARIN.md) uses a frozen [map protocol](mandarin_protocol.json), real THCHS-30 recordings and its own 30/10/10 speaker split. The frontend receives pre-fitted, fixed two-dimensional maps. Reproduction is optional:

```sh
.venv/bin/python -m research.mandarin_prepare
HF_HUB_OFFLINE=1 .venv/bin/python -m research.mandarin_features
HF_HUB_OFFLINE=1 .venv/bin/python -m research.mandarin_map
```

Preparation downloads five pinned training shards (about 2.9 GB), checks phone and word timing consistency, and extracts selected utterances to ignored `data/mandarin/`. Encoder extraction is checkpointed every 1,000 excerpts. Map fitting saves validation selection before final testing and exports both coefficients and attributed reference audio. Test speakers never supply map regions, orientation, reference examples or encoder selection. The available timings are approximate and are not documented as hand-verified.
