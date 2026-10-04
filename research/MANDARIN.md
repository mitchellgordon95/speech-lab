# A live, shared Mandarin sound map

The demo puts **a/e/i/o/u/ü and s/sh/x on one fixed grid**. A rolling 160 ms microphone window becomes an encoder embedding, then two learned coordinates. A short trail makes changes visible. No correct personal reference recordings are required. Start it at [the local live map](http://127.0.0.1:8767/live.html).

The selected map uses **Qwen3-ASR 1.7B's frozen audio encoder**, training-fitted standardization, PCA96, and a class-balanced linear regression directly predicting two display coordinates. The regression's target layout arranges the nine sound categories around a circle. Actual native reference regions are measured from development recordings; microphone points are continuous predictions, never snapped to class centers. The circle and axis directions are display choices, not tongue positions or a claim that all inter-sound distances have equal perceptual meaning.

## What was measured

THCHS-30 supplied **6,819 excerpts from 50 speakers**. A deterministic speaker split reserves 30 for fitting, 10 for validation, and 10 for the final test. At most 20 distinct source-word tokens per phone/speaker were selected without looking at model predictions. Each input is the exact central 160 ms of a labeled 160–400 ms phone. Speaker, word and timing metadata are retained locally. Tone and length marks are removed from category labels; apical/syllabic vowels are not folded into /i/.

The user clarified that they wanted one shared map after initial separate-map validation, before final test scores were computed. Validation was therefore used to develop the projection method as well as choose its encoder. The [protocol and amendment](mandarin_protocol.json) record that change. [Validation selection](results/mandarin-selection.json) was saved before final test evaluation. No test speaker influenced coordinates, model selection, regions, example choice, or the final display rotation.

| Encoder | Shared map: first two LDA coordinates | Shared map: direct coordinate regression |
|---|---:|---:|
| WavLM Base+ | 54.2% | 57.3% |
| Qwen3-ASR 1.7B | 51.4% | **88.3%** |
| Qwen3-Omni | 63.9% | 86.7% |

These are **validation** balanced accuracies, using nearest class center in the actual 2D grid. Regression compared regularization strengths 10, 100 and 1,000; 10 was selected. Keeping only two discriminant coordinates discarded substantial information when combining vowels and consonants. Learning the two display coordinates directly retained much more category separation. This result is about this supervised display, not a general judgment of those encoders.

## Final test

On **1,256 excerpts from 10 held-out speakers**, the selected shared map scored **87.3% balanced accuracy**; the 95% speaker-bootstrap interval was **84.7–90.0%**. This is mean per-category recall, not a pronunciation-quality score.

| Display label | IPA label | Test excerpts | Recall |
|---|---|---:|---:|
| a | a | 166 | 91.0% |
| e | ɤ | 167 | 88.0% |
| i | i | 165 | 95.8% |
| o | o | 39 | 89.7% |
| u | u | 161 | 93.2% |
| ü | y | 162 | 90.7% |
| s | s | 126 | 91.3% |
| sh | ʂ | 149 | 76.5% |
| x | ɕ | 121 | 69.4% |

The main limitation is consonant overlap: x often falls nearer sh, and sh sometimes falls nearer s. The shared layout preserves less of that distinction than a dedicated consonant map. The UI leaves overlapping reference regions visible. The /o/ estimate has fewer examples than the other vowels.

A deterministic test subset of 359 excerpts scored 89.7% at the original crop center, 86.4% shifted 20 ms earlier, and 85.2% shifted 20 ms later, with the projection frozen. This is a modest drop, not proof that every supplied boundary is accurate. Full counts, confusion matrix, bootstrap and timing-shift results are in [the machine-readable report](results/mandarin-maps.json).

## Live behavior and speed

The browser captures 16 kHz mono PCM through an AudioWorklet. It sends the newest 160 ms window every 80 ms when no request is in flight; busy windows are discarded. There is no growing audio queue. Results older than 600 ms are discarded, including a slow initial model load. The displayed dot interpolates fresh measurements and keeps a short trail. Quiet and clipped input hide the dot; both voiced vowels and unvoiced consonants use the same projection. Background sounds and unmodeled phones are not reliably rejected.

Warm encoder extraction on this M5 Pro Mac (48 GB, MPS), 40 calls per encoder:

| Encoder | Median | 95th percentile |
|---|---:|---:|
| WavLM Base+ | 7.7 ms | 9.7 ms |
| Qwen3-ASR 1.7B | 32.3 ms | 35.0 ms |
| Qwen3-Omni | 46.8 ms | 47.7 ms |

[Timing details](results/live-encoder-timing.json). These numbers exclude capture buffering, HTTP, display interpolation, and cold loading; they are not end-to-end feedback latency. A 160 ms observation window plus inference and display smoothing implies roughly a few tenths of a second of perceptual response, not a measured 32 ms response.

The completed browser check returned **161 active frames in 16 seconds** (about 10.1/s, including silence gaps and startup). Median server processing was **50.8 ms**, with a **67.2 ms** 95th percentile. See [the integration result](results/live-browser-check.json).

The browser check sends looped public vowel and consonant excerpts through Chrome's fake microphone, the real AudioWorklet, local HTTP, encoder, and projection. It checks changing coordinates, valid JSON, microphone release, reference playback and mobile layout. Looping is a software fixture, not new pronunciation evidence. Live windows are not persisted as clips or analysis jobs.

## References and limits

Original data: **[THCHS-30](https://www.openslr.org/18/)**, Tsinghua University's Center for Speech and Language Technologies, published by Dong Wang and Xuewei Zhang, Apache 2.0. Audio/phone packaging: **[changelinglab/thchs30-segment](https://huggingface.co/datasets/changelinglab/thchs30-segment)**. Word/phone timings: **[anyspeech/THCHS-30-alignments](https://huggingface.co/datasets/anyspeech/THCHS-30-alignments)**. Both revisions are pinned in the protocol. The supplied metadata does not document alignment construction or human verification; treat its boundaries as approximate, not timestamp-perfect ground truth.

The 27 playable words are real speech crops from development speakers, selected by a deterministic hash with three different voices per category. Playback selection skips quiet or clearly clipped audio; this development-example quality check does not remove any training or test tokens. They include surrounding sounds; the hollow reference dot marks the measured central phone, not the whole word. [Attribution and modifications](../speechlab/live_assets/README.md), [per-example source timestamps](../speechlab/live_assets/catalog.json).

These results establish some separation of labeled native speech across speakers. They do not validate intermediate learner pronunciations, sustained-sound performance, personal voice adaptation, speech rehabilitation, tones, or tongue-motion directions. Encoder pretraining exposure to THCHS-30 is unknown. The grid is a useful candidate for visual exploration, with known overlap, rather than a calibrated correctness scale.
