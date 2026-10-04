# Models and methods

The registry pins the exact tested checkpoint revisions. Revisions are also included in feature-cache keys, so changing a checkpoint cannot silently reuse another version's cached vectors.

| UI name | Upstream checkpoint | Default feature |
|---|---|---|
| `wavlm-base` | [microsoft/wavlm-base-plus](https://huggingface.co/microsoft/wavlm-base-plus) | `hidden_states[6]` |
| `wavlm-large` | [microsoft/wavlm-large](https://huggingface.co/microsoft/wavlm-large) | `hidden_states[9]` |
| `xls-r` | [facebook/wav2vec2-xls-r-300m](https://huggingface.co/facebook/wav2vec2-xls-r-300m) | `hidden_states[12]` |
| `qwen-asr` | [Qwen/Qwen3-ASR-0.6B-hf](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf) | Final encoder output |
| `qwen-asr-large` | [Qwen/Qwen3-ASR-1.7B-hf](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf) | Final encoder output |
| `qwen-omni` | [Qwen/Qwen3-Omni-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) | Final audio tower output |

For WavLM/XLS-R, `hidden_states[0]` precedes transformer blocks and `-1` selects the last hidden state. Qwen's nonnegative indices hook outputs of zero-indexed transformer blocks; `-1` selects complete encoder output. Thus “layer 9” has different conventions between the families. Omni's projected final output and intermediate features also have different dimensions.

The Qwen loader uses native Transformers classes and selects `audio_tower` tensors from safetensors. ASR's entire single-file checkpoint is downloaded, but only audio weights are loaded into a model. Omni downloads only shards containing audio weights. No text decoder or speech generator is instantiated. Float32 provides a simple MPS/CPU path.

Features are cached by recording ID, crop, encoder, layer, and extraction version. Audio is immutable; label edits do not invalidate features, while crop edits do. Four ordered temporal means form comparison vectors, with full frames retained in `.npz` for future research. Axes are numerical JSON, not executable pickles. One worker queues model jobs; unsupported MPS operations may retry on CPU and report that fallback.

## SPARC

[Speech Articulatory Coding](https://github.com/Berkeley-Speech-Group/Speech-Articulatory-Coding) supplies the [linear inversion checkpoint](https://huggingface.co/cheoljun95/Speech-Articulatory-Coding). This demo uses a 16 kHz waveform, z-score before 160-sample padding on each side, WavLM Large `hidden_states[9]`, a fifth-order 10 Hz low-pass filter at the 50 Hz feature rate, and the published coefficients/intercept.

The upstream sklearn pickle is read with a restricted type allowlist. Uploaded files are never unpickled. No vocoder is loaded. Consult [Coding Speech through Vocal Tract Kinematics](https://arxiv.org/abs/2406.12998) for the research context; these estimates are not direct articulography or personalized anatomical measurements.

## Acoustic references

- [Praat Burg formants](https://www.fon.hum.uva.nl/praat/manual/Sound__To_Formant__burg____.html)
- [Parselmouth](https://parselmouth.readthedocs.io/en/stable/)
- [SciPy Welch spectra](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html)

## Cloud interfaces

- [Azure short-audio REST](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/rest-speech-to-text-short) and [pronunciation assessment](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/how-to-pronunciation-assessment).
- [SpeechSuper official samples](https://github.com/speechsuper/SpeechSuper-API-Samples). Confirm language, core type, and reference-text convention with your account. [Usage terms](https://speechsuper.com/terms.html) restrict reuse of outputs for other speech engines.
- [Qwen Omni API](https://www.alibabacloud.com/help/en/model-studio/qwen-omni): audio input and streamed text output. Set your workspace/region endpoint.

Availability and provider interfaces can change. No live paid calls were made during setup. Missing-credential errors show variable names, never secret values.
