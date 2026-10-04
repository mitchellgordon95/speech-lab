# Validation record

Tested on an Apple M5 Pro MacBook Pro with 48 GB unified memory, Python 3.12, PyTorch 2.14.1, and Transformers 5.18.0. Model revisions are pinned in the source and Python dependencies in `uv.lock`.

## Real inference

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

## Software checks

- **14 automated tests pass:** cropping/resampling, silent/long audio rejection, synthetic formant contrast, unvoiced formant handling, session-held-out validation, calibration self-comparison rejection, feature-cache separation, upload/analysis/export, cross-origin writes, cloud consent, missing credentials, provider payloads and cropped audio, streamed response parsing, and fixture idempotence.
- **All eight browser routes pass** in Chrome with a simulated microphone: MediaRecorder/ffmpeg capture, local analyses, fitting/using directions, hidden/revealed feedback, cloud-consent rejection, and a 390-pixel mobile layout. No JavaScript page errors were observed.
- Python lint and JavaScript syntax checks pass. Overview, acoustic, articulation, and mobile screenshots were inspected.

Browser tests use real local inference. Provider payload tests use mocked HTTP responses without contacting services.

## Limits

No live Azure, SpeechSuper, or Qwen API calls were made because credentials were not supplied. No real learner recording, native-speaker discrimination, instructional effectiveness, clinical utility, or accessibility outcome was evaluated. Inference success verifies the software path, not useful pronunciation feedback.

Upstream warnings concern WavLM mask types, unused XLS-R pretraining-head weights, Qwen text-side config keys, the older sklearn version of the SPARC checkpoint, and Starlette's HTTP test-client transition. Outputs passed finite-value/shape checks. SPARC uses numerical coefficients directly instead of the old estimator's prediction method.
