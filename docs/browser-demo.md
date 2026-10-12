# Browser demo on GitHub Pages

**[Open the online demo](https://sudarshan-s-harithas.github.io/motionglot-demo/)**

The three human-motion tasks run inside the visitor’s browser: text-to-human, question-to-human and human-to-text. GitHub Pages serves the static
interface, and the public Hugging Face model repository serves the ONNX files.
There is no Python inference server, account requirement or paid Space.
Prompts and uploaded motions are processed on the visitor’s device.

Human checkpoint examples play immediately. Click **Load model & generate** to
run a new prompt. The first generation downloads approximately 253 MB of GPT
weights and a 37–40 MB VQ decoder; captioning loads the appropriate 37–39 MB
encoder as well. Downloaded models are cached with the browser Cache API when
storage permits. A desktop browser with WebGL, WebAssembly SIMD and several GB
of available memory works best. Single-threaded WASM inference runs in a Web
Worker, so the viewer stays interactive and GitHub Pages needs no special
cross-origin isolation headers. Mobile playback works, but live inference may
exceed a phone’s available memory. The Python demo remains available.

## Visualization

[mjswan 0.11.6](https://github.com/ttktjmt/mjswan) provides MuJoCo WASM and its
Three.js viewer. `web/viewer.js` pauses the runtime and disables physics steps,
including the brief startup loop during scene loading. Playback writes mocap
positions and orientations, calls `mj_forward`, then refreshes render state.
The physics clock remains zero. There is no policy or physics integration.

Human features recover a 22-joint HumanML3D skeleton. The online demo includes
only human motion; its controls, examples, inference worker and viewer have no
robot task or playback paths.

The public mjswan engine API does not yet expose arbitrary frame updates. A
small adapter in `web/viewer.js` accesses the pinned version’s runtime fields,
including mocap render updates. Test that adapter before changing the mjswan
version. The rest of the app uses the public engine API. mjswan is credited in
the demo footer and its Apache-2.0 license is retained under `licenses/`.

## Run or host the static demo

```bash
python -m http.server 8000 --directory web
```

Open http://localhost:8000. No Node build or Python model installation is
required to serve the interface. The browser imports pinned mjswan, fflate
and ONNX Runtime Web bundles from jsDelivr. These and the model downloads
require internet access.

The published demo lives under the owner’s existing GitHub Pages site at
`/motionglot-demo/`. `web/` is the authoritative source; its deployable files
are copied to that site’s source and `gh-pages` branch with `python scripts/export_browser_site.py --output /path/to/site/motionglot-demo`. Relative paths allow
hosting the same directory at another URL. Keep `web/settings.json` pinned to
an immutable Hugging Face revision when publishing.

## Model export

Browser artifacts are published at
[sudarshanharithas/MotionGlot, browser-v0.1.0](https://huggingface.co/sudarshanharithas/MotionGlot/tree/browser-v0.1.0/browser).
The original PyTorch release at `v0.1.0` remains available.

GPT exports use an explicit key/value cache and last-token logits. We store
weights as float16, cast to float32 for computation and keep the cache in
float32. This reduces the download without using int8 quantization. Numerical
rounding can change sampled or near-tied greedy choices. The export validation
found a maximum logit difference of 0.00589 and mean difference of 0.00091 on
the recorded test prompt. Human decoder error was below 0.00014 on its test
input; human encoder code sequences matched their Python reference inputs exactly. These are numerical checks,
not a re-evaluation of the paper’s metrics.

Each browser model has a SHA-256 and size in `browser/manifest.json`. Export
from the public PyTorch bundle with Python 3.11:

```bash
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e . onnx==1.17.0 onnxruntime==1.20.1
python examples/download_checkpoints.py --output pretrained/release-v0.1.0
python scripts/export_browser_models.py \
  --checkpoint-dir pretrained/release-v0.1.0 \
  --output pretrained/browser-v0.1.0
```

Models stay outside Git. Point `web/settings.json` at your public ONNX folder
if you train a new model with the same configuration.

## Checks

```bash
node --test web/tests.mjs
python -m pip install playwright
python scripts/test_browser_demo.py --url http://localhost:8000/
```

The Node tests compare byte BPE with Python tokenization for task markers,
whitespace and Unicode, compare skeleton recovery with the Python reference,
and check input validation. They download only
the tokenizer/config JSON files. The Chrome test runs all three human tasks, checks
end markers and captions, verifies the physics clock on localhost, and checks
mobile overflow. It requires Chrome at `/usr/bin/google-chrome` or `--chrome`.
The production UI labels examples and newly generated results separately.

Motion downloads are JSON objects with `features`: human arrays have shape
[frames, 263] in unnormalized HumanML3D representation. Captioning accepts these JSON
objects or C-order float32/float64 `.npy` arrays. Frame counts must be divisible
by four. Use 20 fps for playback.
