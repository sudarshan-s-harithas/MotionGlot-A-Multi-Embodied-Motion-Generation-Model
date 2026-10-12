# Recovered research sources

These files preserve the training, tokenization, task generation, motion processing and evaluation implementations recovered from the local research snapshots. Original source hashes are recorded in `../docs/source_manifest.json`. Personal/cluster absolute paths were replaced with `REPLACE_WITH_LOCAL_PATH`; otherwise their historical behavior is preserved.

Use `motionglot` commands from the repository root for the supported workflow. The archived scripts may parse arguments or execute work at import time, assume a specific working directory, refer to unavailable optional artifacts, or contain earlier codebook-size configurations. They are research references, not the supported command interface.

`human_vq/train_vq.py` comes from the local T2M-GPT checkout. `tokenizer/process_robot_motion/sort_robot_dataset.py` comes from the earlier `motioinglot/tokenizer` snapshot and contains the augmentation/caption association code missing from the October 19 snapshot. Remaining tokenizer/GPT scripts are from the October 19 MotionGlot snapshot. Required attribution and third-party licenses are in the repository root and `../licenses/`.
