# Experiment guide

The central question is whether feedback tells you which change to make, then whether that change helps without the display. Separating two recorded words is an earlier, easier test.

## Collect one Mandarin contrast

Keep vowel context and tone comparable. Starting points:

- **Aspiration:** pinyin `b/p`, `d/t`, or `g/k`; mark release and voicing onset manually. These are not simply English-style voiced/voiceless contrasts.
- **Frication:** `s/sh` with comparable following vowels; inspect frication spectra and embeddings. Vowel transitions and microphone response can confound the difference.
- **Affricates:** `z/zh`, or aspiration within `z/c`, `zh/ch`, and `j/q`. Preserve release, frication, and transition timing.
- **Vowels:** sustained `i`, `u`, and `ü`, then short syllables. Inspect stable middle portions first and trajectories second.

Use a proficient speaker or teacher to label acceptable productions. Endpoint labels should describe what was actually heard or measured, rather than what the learner intended. A first collection could contain 10–20 attempts per endpoint across three sessions, with both endpoints randomized within each day. Three clips per side only makes a demo fit possible; it is not sufficient validation. Do not duplicate recordings or treat overlapping crops as independent examples.

## Acoustic microscope

Acoustic analysis preserves the input sample rate. Praat/Parselmouth estimates pitch and Burg formants; median formants use voiced frames. Try different ceilings when tracks look implausible for your voice. Frication moments summarize power above 1 kHz, up to 12 kHz or the input's Nyquist limit. A 16 kHz recording cannot recover energy above 8 kHz.

The waveform markers measure a manually selected interval. They do not automatically identify release or voicing onset. Use expert-marked examples to evaluate timing accuracy. Browser processing is requested off, but hardware and browser behavior can still affect measurements.

## Frozen speech and Omni representations

Selected audio is resampled to 16 kHz. Full frame features are saved alongside four consecutive temporal-bin means. This preserves coarse order without claiming phoneme alignment. Silence, word identity, duration, speaker, and microphone can all affect distance.

Compare the same targets. Evaluate layers and encoders with the same held-out sessions. PCA coordinates are recomputed for each batch and cannot be compared across plots. Cosine distance has no universal pronunciation threshold.

Try repeated accepted productions, target contrasts across days, and nuisance changes such as volume or room. Compare a whole syllable against a manually cropped consonant-plus-transition region. Forced alignment and dynamic time warping are follow-up experiments, not implemented features.

## Learned directions

A standardized, regularized logistic regression fits two labeled sets. Its signed margin and sigmoid position define a contrast between those labels. Standardization is fitted inside each training fold; validation leaves entire sessions or speakers out. Folds whose training set contains only one label are skipped. One session has no independent validation.

Balanced accuracy describes the discrimination task, not coaching effectiveness. Repeatedly choosing layers on the same validation set also overfits; reserve a final unseen day. A direction becomes interpretable only when its training examples represent a meaningful, independently verified contrast. It does not make an arbitrary feature dimension equivalent to “move your tongue 2 mm forward.”

## Estimated articulation

SPARC maps WavLM features to reference-space coordinates for tongue dorsum, tongue blade, tongue tip, lower incisor, upper lip, and lower lip. The demo follows the published normalization, padding, filtering, and linear projection, without loading speech synthesis.

These are model estimates, not anatomy measurements or a personal mouth calibration. Try whether relative trajectories repeat for the same person and sound before interpreting gestures. Mandarin transfer, atypical speech, and clinical use have not been validated here.

## Personal calibration

Use independently accepted recordings of the same intended target from your own voice, plus population references. Compare a held-out attempt to each centroid. A smaller personal distance might reflect reduced speaker mismatch, or merely similarity to an established error; anchor selection matters.

This version adapts the reference set without fine-tuning an encoder or applying anatomical normalization. For a supervised personal probe, use the direction-fitting demo. Test whether either distance agrees with independent human ratings on new sessions.

## Optional assessors

Separate transcription success from phonetic accuracy and an LLM suggestion from a measurement. Services use different definitions and scales; do not average raw scores. Provider outputs never become automatic training labels. Observe the service-specific restrictions in the README.

## Feedback practice

Fit an axis and collect new recordings. Follow a short visual-feedback block with hidden-feedback attempts and a later session. Hidden trials save results without showing the gauge until requested. Training recordings are flagged when reused.

Compare independently judged production quality, retention, and user effort. The app records condition and timestamps but does not randomize a study, automatically score learning, or establish causality. Visual practice does not require listening to audio.
