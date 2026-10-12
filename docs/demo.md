# Run the interactive demo locally

The local demo is adapted from the original live generator in `LAM/visulization/GUI/generate_human_motion.py`. Its GPT checkpoint is `checkpoint-29200`; the release preserves the exact GPT weights and vocabulary, with the matching human and robot VQ models and human normalization. The nearby historical `gradio_ui.py` merely played a prerecorded video and was not used as the inference implementation.

## Install and launch

Use Python 3.10–3.12 and an isolated environment for the commands below. The CPU example works without a GPU:

```bash
git clone https://github.com/sudarshan-s-harithas/MotionGlot-A-Multi-Embodied-Motion-Generation-Model.git
cd MotionGlot-A-Multi-Embodied-Motion-Generation-Model
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[demo]'
motionglot-demo
```

On Windows, replace the activation line with `.venv\Scripts\Activate.ps1` in PowerShell. On systems without the `venv` module, install the appropriate Python venv package or use a Conda environment with Python 3.10+.

Visit **http://127.0.0.1:7860**. The first start downloads the approximately 660 MB checkpoint bundle from [Hugging Face](https://huggingface.co/sudarshanharithas/MotionGlot), at the `v0.1.0` revision, into `checkpoints/release-v0.1.0/`. Downloaded files are checked against the bundle's SHA-256 manifest. Subsequent starts use that directory.

Gradio 6.0.2 is pinned to remain compatible with the research stack's Transformers 4.44.2 / Hugging Face Hub dependency range. Human previews use H.264 MP4 encoding and do not need SMPL assets or a hardware robot. Model instances are cached and requests are serialized.

## Use the three tabs

- **Human motion:** choose a description or situational question, enter a prompt and generate. The result is a motion video plus downloadable denormalized `[frames,263]` features and metadata.
- **Robot motion:** choose a text description or goal reaching. A trajectory plot shows motion in the model's planar axis convention. Download `[frames,3]` increments and metadata; generated gait appears in the result text when present.
- **Motion captioning:** upload a numeric `.npy` feature array and choose human or robot. Uploads accept 4–256 frames, up to 1 MB, with 263 or 3 features respectively. Generated feature downloads can be uploaded directly. Select axis swapping only for original robot dataset files that need the research conversion.

The tab choices cover all six tasks. Change the random seed to explore sampled alternatives. Requests create separate output files rather than overwriting a global video.

## Existing checkpoints and CUDA

```bash
python examples/download_checkpoints.py --output checkpoints/release-v0.1.0
motionglot-demo --checkpoint-dir checkpoints/release-v0.1.0
```

For CUDA, install a matching CUDA PyTorch build first, then run:

```bash
motionglot-demo --device cuda
```

`python app.py` and `python -m motionglot.web_demo` expose the same options. `--port 7861` changes the local port; `--threads 2` controls PyTorch CPU threads. Generated videos and files are placed in a demo-specific directory under the system temporary folder and copied into Gradio's managed output cache.

## Optional sharing

The default server listens on localhost. For a temporary public link from your machine:

```bash
motionglot-demo --share
```

Gradio reports the public URL and expiry in its terminal output. The machine and process must remain running. To listen on a server's network interface, use `--host 0.0.0.0` and configure that server's network access as needed. No hosted Space is required for local execution.

## Release artifacts

The [Hugging Face checkpoint repository](https://huggingface.co/sudarshanharithas/MotionGlot) contains inference artifacts only: GPT weights/tokenizer, human VQ, robot VQ, normalization and file manifest. Optimizer state, training data and body models are excluded. The [GitHub release](https://github.com/sudarshan-s-harithas/MotionGlot-A-Multi-Embodied-Motion-Generation-Model/releases/tag/v0.1.0) packages the source and checkpoint bundle for offline setup.

To use a checkpoint archive, extract its contents and pass the resulting directory to `--checkpoint-dir`. The directory must contain `gpt/`, `human_vq.pth`, `robot_vq.pth`, `normalization.npz`, and `manifest.json`.
