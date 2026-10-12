# Recovery and reproduction notes

The reference is [MotionGlot, arXiv:2410.16623](https://arxiv.org/abs/2410.16623), including the May 2025 revision. The local October 19 snapshot supplied the combined tokenizer, VQ architecture, task preparation, GPT training and inference/evaluation scripts. Earlier local snapshots supplied the robot augmentation script and the T2M-GPT human VQ training reference. `source_manifest.json` records source-relative locations, SHA-256 hashes and release adaptations.

## Components recovered

| Stage | Research source | Supported release implementation |
| --- | --- | --- |
| Human VQ-VAE | `research/human_vq/train_vq.py`, tokenizer models | `motionglot train-vq`, `configs/human_vq.json` |
| Robot VQ-VAE | `research/tokenizer/tokenize_robot_trajectories.py` | `motionglot train-vq`, `configs/robot_vq.json` |
| Robot parameterization / augmentation | `post_process.py`, `process_robot_motion/data_process_utils.py`, `sort_robot_dataset.py` | Documented tensor inputs and `prepare-data` caption/augmentation handling |
| Shared vocabulary | `research/tokenizer/tokenize_motion2.py: define_vocabulary` | `motionglot build-vocab`, `assets/tokenizer/` |
| Six task instruction sequences | `research/tokenizer/tokenize_motion2.py` | `motionglot prepare-data`, `motionglot build-corpus` |
| GPT training | `research/train_motionglot/train.py` | `motionglot train-gpt` |
| Motion generation / captioning | Recovered generation and evaluation scripts | `motionglot generate` for all six tasks |

The original code's 32-code and 600-goal variants remain visible in archived scripts/configuration. The selected trained VQ checkpoints and saved `all_tasks_gait` tokenizer use the paper's **512 human codes, 128 robot codes, 784 goals**. GPT-2 has 50,257 base tokens; the complete saved vocabulary has 51,694 tokens. The real human and robot state dictionaries load strictly into the release models with 263 and 3 input dimensions respectively, code dimension 512, and temporal downsampling by four.

## Deliberate release changes

- VQ architecture and parameter names remain checkpoint-compatible. Package imports, initial codebook device allocation and batched decoding were corrected. EMA/reset quantization retains the original MSE commitment term.
- Absolute personal/cluster paths in the archive were replaced with `REPLACE_WITH_LOCAL_PATH`. The supported commands take explicit paths and avoid import-time execution of the legacy scripts.
- Instruction data are saved as JSONL, replacing large GPU tensor pickles. The archive's dataset saving was commented out; the release saves its output and provenance explicitly.
- Data splits are explicit. All captions, questions, mirrored samples, scaled samples and goal labels for one source stay together. Test data are not reused as validation. This differs from the archived last-30-example robot split and its inclusion of captioning examples across the holdout boundary.
- Questions train human generation and are not used as captioning targets. Descriptive captions are retained for motion-to-text.
- GPT padding labels use `attention_mask == 0`. A genuine EOS token is not accidentally masked just because EOS and padding share an ID.
- Checkpoint resumption is optional. The original trainer's fixed resume directory and fixed output directory were removed. Tracking services are disabled by default.
- GPT initialization is explicit: pretrained GPT-2 by default, random initialization via `--from-scratch`. The archived script loads pretrained weights and then calls `init_weights()`, whose effect depends on the installed Transformers version; that line is not treated as evidence of random initialization.
- The goal prompt preserves the actual saved checkpoint template (`<goal_id_N><eop><tog> \n `). It omits the paper's prose prefix `reach goal: `, which the archived corpus builder did not emit.
- Output generation is constrained to the correct embodiment's motion codes or base language tokens, plus the task's end token and optional initial gait. This is a release behavior, not a claim about the original decoding protocol.
- Human VQ training selects no FID checkpoint automatically; it saves the latest state. Robot VQ steps are explicit optimizer steps rather than the archived epoch loop. Full-run iteration counts must be selected for the dataset and hardware.

## Validation scope

Tests verify atomic token IDs and rebuilding the exact shared vocabulary, six task templates, all 784 goal cells, padding masks, split leakage rejection, robot axis/cropping behavior, VQ training gradients and checkpoint reloads, corpus preparation, short GPT training, and inference on all six tasks with a tiny GPT.

Local real-asset checks additionally run the recovered human VQ, robot VQ, and GPT checkpoint on CPU. A small real-data corpus is generated for every task in all three splits. One example of each task is generated with the recovered GPT; each terminates at its correct end marker. These are functionality checks, not benchmark evaluations or an independent replication of the paper's reported numbers.

Full VQ/GPT retraining, eight-GPU training, paper benchmark metrics and hardware-controller experiments have not been run for this release. GPU access was unavailable in the validation environment. Do not report the paper's metrics as newly reproduced results. `docs/validation.md` records the completed checks.

## Archived evaluation

The original evaluation scripts, feature extractor modules and utilities are retained under `research/tokenizer/` for traceability. They require dataset files, GloVe/evaluator artifacts, optional dependencies and path configuration from the original experiment environment. The `research` extra supplies common analysis libraries; it is not a guarantee that every historical optional experiment is independently runnable. The standalone supported package does not import these scripts.

The original raw robot-recording conversion consumes pickled robot pose/joint-state objects plus recording metadata. Reusing that path also requires the original SDK object definitions and metadata; the supported release starts with the recovered numeric SE(2) tensors. Robot execution additionally requires the separate gait/controller stack. Body-model fitting, video pose estimation, RT-2/OpenVLA baselines and unrelated experimental checkpoints are outside this source release.
