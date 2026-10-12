MotionGlot now includes an installable interactive demo and the recovered step-29,200 checkpoint, with the exact shared tokenizer, both motion VQ-VAEs and matching human normalization.

Install the source with Python 3.10+, then run:

```bash
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[demo]'
motionglot-demo
```

Open http://127.0.0.1:7860. The first launch automatically downloads and verifies the checkpoint bundle. Human generation returns a playable motion video and feature arrays; robot generation returns trajectories and gait metadata; captioning accepts feature uploads. The UI supports all six tasks.

Public checkpoints are also available at https://huggingface.co/sudarshanharithas/MotionGlot. This release includes the source ZIP and a checkpoint archive for offline installation; the latter has a companion SHA-256 file.

The checkpoint was recovered from the original local live demo. Published research examples and single-example functionality checks are retained in the documentation. Full retraining and independent reproduction of benchmark results have not been run for this release. Training datasets and optimizer state are excluded.
