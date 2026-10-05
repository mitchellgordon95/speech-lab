# All Mandarin consonants on the live map

The expanded map includes **all 21 Mandarin initials, final ng, and six vowels**: 28 categories, 84 playable word excerpts, and one continuously updating dot. Open [the local map](http://127.0.0.1:8767/live.html), choose a family with **Show**, and use **Freeze trace** to inspect a brief sound. The **Original · 9 sounds** option preserves the previous map.

## How the two maps differ

The original nine-sound map takes a 160 ms waveform, runs Qwen3-ASR's frozen audio encoder, and predicts two display coordinates with a small linear regression. Its supervised target layout is a circle. The plotted reference regions come from actual native recordings projected through that fitted model. It does not discover a natural two-dimensional atlas of articulation.

The expanded map uses the same encoder, standardization and PCA128, followed by a class-balanced multinomial logistic classifier. Its 28 category scores are normalized with softmax and used to blend fixed screen positions arranged in sound-family rows. This keeps distinctions that were lost by directly regressing two coordinates. The fitted transform is cached as a compact NumPy archive; encoder weights remain separate.

**The expanded dot is a category-resemblance display.** Its distances are neither formant differences nor tongue movements. It always blends positions inside the known category layout; an unfamiliar sound or a mixture can land near an unrelated category. No displayed position is a calibrated pronunciation grade. The original direct projection remains available for comparison.

## Data and timing

The set contains **13,231 original windows from 50 speakers**, with the previous 30/10/10 fitting/validation/test assignment. Vowels retain the 160–400 ms phone-duration criterion. Consonants accept 40–400 ms phones, using a central 160 ms slice of continuous audio. For short phones, that slice includes neighboring sounds. The median consonant occupies about 56% of its input window; this is not context-free isolated-phone assessment.

Selection uses up to ten unique source words per phone/speaker, stable hashing, and quiet/clipping checks. A further **3,317 shifted windows from development speakers only** improve timing tolerance. Their offsets are selected deterministically from −40, −20, +20, and +40 ms. No test speaker supplies fitted examples, reference regions, playback words, or training augmentation.

The same test speakers were reused after the first timing check exposed a weakness. The method was then improved and candidates selected on centered and shifted validation windows. These are transparent **exploratory engineering results**, not an untouched confirmatory test. The [protocol](mandarin_expanded_protocol.json) records both amendments; earlier results are retained.

## What worked

Direct two-coordinate regression struggled with this larger inventory. The initial linear versions scored 41.9–48.6% on centered validation windows. A small nonlinear regression improved that to 64.1%, but its centered-only test map fell to 28.9–32.5% under ±40 ms shifts. Adding timing variation helped modestly; averaging the four encoder time bins did not help.

Learning category scores, then blending their positions, worked substantially better. The final selection compared both centered and shifted validation windows:

| Candidate | Centered validation | Shifted validation |
|---|---:|---:|
| mlp/128/four/False | 64.1% | 41.2% |
| mlp/128/four/True | 60.1% | 48.4% |
| mlp/128/mean4/False | 55.1% | 39.9% |
| mlp/128/mean4/True | 52.0% | 40.2% |
| classifier/128/four/False | 87.9% | 75.1% |
| classifier/128/four/True | 87.3% | 80.5% |

The selected classifier uses PCA128, regularization C=0.1, four temporal bins and timing augmentation. On **2,546 windows from ten test speakers**, the displayed 2D positions scored **86.8% balanced accuracy**, with a 95% speaker-bootstrap interval of **82.7%–89.8%**. The full classifier before 2D projection scored 89.2%. The reported map metric uses the nearest development-speaker class center, not the classifier label.

| Sound family | Mean recall in the 28-category map |
|---|---:|
| vowels | 92.0% |
| fricatives | 88.6% |
| sonorants | 83.7% |
| stops | 92.0% |
| affricates | 77.3% |

| Pinyin | IPA | Test recall |
|---|---|---:|
| a | a | 91.9% |
| e | ɤ | 93.1% |
| i | i | 92.9% |
| o | o | 92.3% |
| u | u | 95.1% |
| ü | y | 86.6% |
| b | p | 88.2% |
| p | pʰ | 92.2% |
| d | t | 93.9% |
| t | tʰ | 94.8% |
| g | k | 87.5% |
| k | kʰ | 95.3% |
| z | ts | 81.5% |
| c | tsʰ | 74.5% |
| zh | ʈʂ | 59.2% |
| ch | ʈʂʰ | 70.0% |
| j | tɕ | 91.0% |
| q | tɕʰ | 87.9% |
| f | f | 94.7% |
| h | x | 97.9% |
| s | s | 80.5% |
| sh | ʂ | 79.0% |
| x | ɕ | 90.8% |
| m | m | 91.4% |
| n | n | 72.0% |
| ng | ŋ | 78.0% |
| l | l | 91.4% |
| r | ɻ | 85.6% |

Timing shifts on the same 560 test windows, with the trained model frozen:

| Window-center offset | Balanced accuracy |
|---|---:|
| -40 ms | 71.4% |
| -20 ms | 83.2% |
| 0 ms | 87.7% |
| 20 ms | 85.7% |
| 40 ms | 77.3% |

Small timing errors are much less damaging now, but ±40 ms still causes a meaningful drop. This matters for brief consonants and is why the UI preserves a four-second trail and can freeze it. The browser offers fresh 160 ms windows every 40 ms; it drops windows while inference is busy. It does not detect phoneme boundaries, and a running syllable can move the dot through multiple regions. Family filters hide reference regions without changing coordinates, the model, or the live microphone.

## Browser verification

The real AudioWorklet → local API → encoder → projection path returned 104 active frames during a 15-second fake-microphone replay containing real word clips and silence. Median server processing was 53.5 ms, with a 94.0 ms 95th percentile. This excludes capture buffering and display interpolation. The test verifies live movement, filtering while the microphone remains active, a frozen trace that stays fixed, microphone release, original-map availability, and mobile layout. It is a software check, not a learner or continuous consonant-recognition evaluation. [Recorded checks](results/mandarin-expanded-browser.json).

## Sources and reproduction

Recordings are [THCHS-30](https://www.openslr.org/18/), Tsinghua University, Apache 2.0. The pinned audio package and timestamp sources are the same as in [the original study](MANDARIN.md). Phone timings are approximate and their construction/human verification is undocumented. The initial inventory follows [Hong Kong Polytechnic University’s Mandarin initials reference](https://www.polyu.edu.hk/bepth/introduction-to-phonetics/initials/introduction-to-initials/?sc_lang=en); the [ASHA chart](https://www.asha.org/siteassets/uploadedFiles/Mandarin-Phoneme-charts.pdf) includes final /ŋ/. Glides and additional vowel allophones are outside this version.

Each playable excerpt retains the natural word, extended when needed to contain the entire measured 160 ms window. No synthesis, pitch change, or time stretching is used. Source identifiers, speaker IDs and crop times are in [the asset catalog](../speechlab/live_assets/catalog.json). All examples come from development speakers and pass quiet/clipping checks; they are selected independently of model correctness.

```sh
.venv/bin/python -m research.mandarin_expand_prepare
HF_HUB_OFFLINE=1 .venv/bin/python -m research.mandarin_expand_features
HF_HUB_OFFLINE=1 .venv/bin/python -m research.mandarin_expand_jitter
HF_HUB_OFFLINE=1 .venv/bin/python -m research.mandarin_expand_map --jitter
```

The original study must be prepared first so its cached corpus and speaker assignment are available. No research steps are needed to use the app. [Final results](results/mandarin-expanded.json), [validation selection](results/mandarin-expanded-selection.json), [centered-only baseline](results/mandarin-expanded-centered-baseline.json), and [timing-augmented regression](results/mandarin-expanded-jitter-regression.json) are committed; full recordings, embeddings and intermediate windows stay local.

This tests labeled read speech across speakers. Learner errors, useful articulatory directions, personal voice adaptation, rehabilitation outcomes, and open-set sound rejection remain unvalidated. Encoder pretraining exposure to this corpus is unknown.
