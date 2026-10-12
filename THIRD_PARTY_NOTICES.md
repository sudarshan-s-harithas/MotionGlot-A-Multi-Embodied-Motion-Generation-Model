# Third-party components

The distributed VQ-VAE components originate from the T2M-GPT architecture and the local MotionGlot adaptations. T2M-GPT's Apache-2.0 license, including its copyright notice, is retained in `licenses/T2M-GPT-Apache-2.0.txt`. The human tokenizer training reference in `research/human_vq/train_vq.py` comes from the local T2M-GPT checkout. Package model files carry change notices where modified.

The included tokenizer contains the GPT-2 vocabulary and merge rules, augmented with MotionGlot tokens. GPT-2 and its tokenizer originate from OpenAI's [GPT-2 repository](https://github.com/openai/gpt-2). Its Modified MIT license is retained in `licenses/GPT-2-MIT.txt`. Hugging Face Transformers, Accelerate, PyTorch and NumPy are installed dependencies, not vendored implementations.

HumanML3D supplies human motion representations and captions. The archived research utility/evaluation files also use HumanML3D and text-to-motion conventions. The MIT license from the local HumanML3D checkout is retained in `licenses/HumanML3D-MIT.txt`; its code license does not grant access to all underlying motion datasets. Follow upstream instructions and the applicable AMASS/body-model terms when obtaining data or model assets.

The companion local [text-to-motion](https://github.com/EricGuo5513/text-to-motion) project's MIT license is retained in `licenses/Text-to-Motion-MIT.txt` for the archived evaluator/feature-utility lineage.

No AMASS motion files, SMPL/SMPL-H body models, robot hardware SDK, learned evaluator artifacts, QUAD-LOCO motion data or trained model weights are part of this source archive. QUES-CAP text and local asset links remain in the ignored `local/` folder for validation. Release those assets separately under their applicable terms if desired.

The recovered inference checkpoints are now published separately at [sudarshanharithas/MotionGlot](https://huggingface.co/sudarshanharithas/MotionGlot), with tokenizer and VQ attribution and license notices. The local demo downloads that bundle; source archives continue to exclude model binaries and training data.

`research/` records the recovered experiments. It includes evaluator code whose required artifacts and dependencies must be obtained separately. Consult `docs/reproduction.md` before using it to claim paper metrics.
