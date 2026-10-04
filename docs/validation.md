# Validation record

Tested on an Apple M5 Pro MacBook Pro with 48 GB unified memory, Python 3.12, PyTorch 2.14.1, and Transformers 5.18.0. Model revisions are pinned in the source and Python dependencies in `uv.lock`.

## Completed corpus study and guided demos

All six encoders and SPARC were run on **10,676 real LibriSpeech phone segments**. The speaker-separated study, frozen selection, final test, perturbation checks, and exploratory equal-size adaptation control are complete. [Measured results](../research/RESULTS.md), [run metadata and pinned models](../research/results/run.json).

The default homepage now exposes three fitted contrasts: S/SH, R/L, and IY/IH (the vowels in sheep/ship). Each passed the preset test and robustness gates. It includes 18 public speech examples (three speakers per sound), with both whole-word and isolated-phone playback. No synthetic examples appear in the guided flow.

- **26 automated tests pass**, including exact exported-classifier math, energy-window selection, pitch preservation, amplitude normalization, silence/clipping/short-input rejection, packaged corpus attribution and promotion checks, and API validation. Six recording-path regression cases cover voiced/unvoiced inputs in every guided contrast, including JSON responses and saved results. These caught and now protect against a NumPy boolean escaping the voicing check; built-in examples bypassed that check and had missed the bug.
- **Guided Chrome checks pass** with real local model inference for both endpoints of all three demos, reference playback, MediaRecorder/ffmpeg capture through a fake microphone looping a public corpus sound, result display, a 390-pixel layout, and the completed evidence page. All six first reference examples were classified on their expected side. No JavaScript page errors were observed.
- **All eight original workbench routes pass again**, including microphone capture, local models, fitted directions, hidden feedback, and cloud-consent rejection. Browser checks use a separate `test-results/guided-data` recording store so personal recordings are untouched.
- Python lint/format and JavaScript syntax checks pass. The final desktop and mobile layouts were visually inspected. The homepage screenshot is [guided.png](guided.png).
- All **18 packaged examples** were checked against their cached held-out predictions through the serving API. Maximum position difference was `5.96e-08`; all 18 landed on their expected side. Reference selection used deterministic word/speaker diversity, not model scores.

The local app uses port **8767** because an unrelated interval-trainer server occupies 8765. That server was left running.

Browser capture here is an engineering check. Held learner sounds, real microphones, learning outcomes, Mandarin transfer, and clinical efficacy have not been evaluated. Corpus accuracy is English category discrimination, not a pronunciation grade. Automatic alignment timings have not been manually certified.

## Original setup: inference timing

All six encoders and SPARC ran successfully on **MPS**, with `HF_HUB_OFFLINE=1` after downloading weights. Input: an original 0.9-second synthetic vowel, not human or Mandarin speech.

| Encoder | Frames × dimensions | Load + first inference | Warm inference |
|---|---:|---:|---:|
| WavLM Base+ | 44 × 768 | 2.922 s | 0.013 s |
| WavLM Large | 44 × 1024 | 0.700 s | 0.015 s |
| XLS-R 300M | 44 × 1024 | 1.059 s | 0.018 s |
| Qwen3-ASR 0.6B encoder | 12 × 896 | 0.460 s | 0.062 s |
| Qwen3-ASR 1.7B encoder | 12 × 1024 | 0.584 s | 0.069 s |
| Qwen3-Omni audio tower | 12 × 2048 | 1.271 s | 0.048 s |

SPARC produced 45 frames of 12 coordinates in 0.576 s including model load. [Raw report](benchmark-m5-pro.json).

These are single-run engineering checks, not a controlled performance study. Warm timing includes preprocessing and copying outputs to CPU, which waits for GPU work. First inference includes initialization and first-use library overhead; downloads are excluded. First-load timings should not be used as a speed ranking. Longer inputs and memory pressure change timing.

## Original setup: software checks

- **14 automated tests pass:** cropping/resampling, silent/long audio rejection, synthetic formant contrast, unvoiced formant handling, session-held-out validation, calibration self-comparison rejection, feature-cache separation, upload/analysis/export, cross-origin writes, cloud consent, missing credentials, provider payloads and cropped audio, streamed response parsing, and fixture idempotence.
- **All eight browser routes pass** in Chrome with a simulated microphone: MediaRecorder/ffmpeg capture, local analyses, fitting/using directions, hidden/revealed feedback, cloud-consent rejection, and a 390-pixel mobile layout. No JavaScript page errors were observed.
- Python lint and JavaScript syntax checks pass. Overview, acoustic, articulation, and mobile screenshots were inspected.

Browser tests use real local inference. Provider payload tests use mocked HTTP responses without contacting services.

## Limits

No live Azure, SpeechSuper, or Qwen API calls were made because credentials were not supplied. The original synthetic setup checks established only software execution; the subsequent corpus study above evaluates English sound discrimination. No real learner recording, instructional effectiveness, clinical utility, or accessibility outcome was evaluated.

Upstream warnings concern WavLM mask types, unused XLS-R pretraining-head weights, Qwen text-side config keys, the older sklearn version of the SPARC checkpoint, and Starlette's HTTP test-client transition. Outputs passed finite-value/shape checks. SPARC uses numerical coefficients directly instead of the old estimator's prediction method.
