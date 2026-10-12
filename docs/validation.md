# Completed validation

Validated on October 12, 2026 with Python 3.8.20, PyTorch 1.13.1+cu117, NumPy 1.23.5, Transformers 4.44.2, Accelerate 0.33.0 and pytest 8.3.5. Computation used CPU; CUDA was unavailable. Dependencies were installed into an isolated environment.

| Check | Result |
| --- | --- |
| Editable installation and CLI entry point | Passed |
| Python wheel build | Passed |
| Automated CPU tests | 20 passed |
| Rebuild vocabulary from plain GPT-2 vocabulary/merges | Exact token-ID fingerprint match |
| Real human and robot VQ state dictionaries | Strict loading passed; encoder input dimensions 263 and 3; codebooks 512×512 and 128×512 |
| Real-data manifest preparation | Passed; source families isolated across splits |
| Small real-data corpus | All six tasks present in train, validation and test |
| Recovered GPT checkpoint inference | All six tasks produced outputs and their correct task end token |

The local Python environment exposed unrelated ROS pytest plugins; tests were run with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` to isolate the package's test suite. One upstream tokenizer future-warning was emitted. Tests do not need network access.

## Local inventory

The recovered robot directory contains 6,540 trajectory tensors: 1,090 source segments and mirrored/scaled variants across 18 recordings. These motion files have multiple text templates; tensor count is not the paper's count of augmented text–motion pairs. A 23,842-line QUES-CAP question file and a shorter 2,300-line question file were located, along with their ordered human-motion ID mapping.

The prepared manifest contains:

| Embodiment | Train clips | Validation clips | Test clips |
| --- | ---: | ---: | ---: |
| Human | 24,508 | 1,528 | 4,646 |
| Robot | 5,004 | 840 | 696 |

Human counts include caption-timed subclips and mirrored motion files. Robot variants are grouped by the original recording. These are counts from this local preparation policy; they do not establish the original paper's train/test split sizes or exact dataset version.

The recovered GPT has 51,694 input embeddings and a 1,024-token context. The tokenizer fingerprint is `150c7bea17bedbd2c47edda02a98772e46b456d40b7eb97b2ebba5adfc4204c7`. New GPT training records this fingerprint in its config and refuses incompatible corpus/tokenizer mappings.

## Real inference smoke checks

| Task | Output | End token |
| --- | --- | --- |
| Text → human | 48 frames × 263 features | `<eog_tthm>` |
| Question → human | 80 frames × 263 features | `<eog_tthm>` |
| Text → robot | 192 frames × 3 features; trotting gait | `<eog_ttrm>` |
| Goal → robot | 192 frames × 3 features | `<eog_rmg>` |
| Human → text | Caption string | `<eog_hmtt>` |
| Robot → text | Caption string | `<eog_rmtt>` |

These are functionality checks on individual prompts. Motion semantic accuracy and goal success were not measured. Full-model retraining, distributed training, hardware execution and benchmark reproduction remain untested. Original checkpoints/data are linked only in the ignored `local/` directory, and are absent from the source release archive.

## Local demo and checkpoint release

The installable demo was additionally validated with Python 3.11.5, PyTorch 2.4.1+cpu, NumPy 1.26.4, Gradio 6.0.2, and the pinned research Transformers version. The complete suite passed **27 tests**, including checkpoint integrity, human feature recovery, upload validation, MP4 decoding and robot goal visualization.

All six tasks were exercised with the recovered checkpoint through the demo engine. Text-to-human produced 48 frames and question-to-human produced 184 frames; both returned playable videos. Robot generation and goal reaching each produced 192-frame trajectories, and both captioning tasks returned text. All examples emitted their corresponding task end token. Sampling explains the question-output length difference from the earlier greedy smoke check.

HTTP requests through the running Gradio interface verified human video generation and downloads, captioning of a generated feature file, and robot plot generation. A fresh anonymous download of the public checkpoint bundle passed all file hashes. The released GPT weights have SHA-256 `5ca7c583339ebb073b1223248c15d912ac00733bf3be95a9dbb46c05d96f37a6`, exactly matching the original GUI checkpoint. The GUI's tokenizer fingerprint and normalization arrays also match the released artifacts.

The checkpoint bundle is published separately on Hugging Face and packaged for GitHub Releases. Source archives exclude its model binaries, training data and optimizer state.
