# Data contracts

## Motion manifest

Each JSONL record represents one motion clip. Paths may be absolute locally or relative to the manifest's directory.

```json
{"id":"000123","group":"000123","embodiment":"human","motion":"HumanML3D/new_joint_vecs/000123.npy","split":"train","captions":["A person walks."],"questions":["How does a person cross a room?"]}
{"id":"W_8_9","group":"W_8","embodiment":"robot","motion":"QUAD-LOCO/robot_trajectories/W_8_9.pt","split":"train","swap_xy":true,"captions":["A robot walks forward."],"questions":[]}
```

Required fields are `id`, `group`, `embodiment`, `motion` and `split`. Embodiment is `human` or `robot`; split is `train`, `valid` or `test`. Optional caption/question fields are lists of nonempty strings. IDs are unique within each embodiment. One source group must belong to exactly one split. Set the same group for mirrored, rescaled and cropped variants of a source recording. This is enforced before training or corpus generation.

Human data must have shape `[T,263]`, use the HumanML3D representation, and be finite. The recovered preparation policy accepts 40–199-frame full clips and eligible captioned subclips, interpreting HumanML3D caption timing tags at 20 FPS. Optional `start_frame` and `end_frame` fields slice a clip before encoding. Length is rounded down to a multiple of the VQ downsampling factor. Human normalization comes from a directory containing `mean.npy`/`std.npy` (case variants accepted), or an `.npz` containing `mean` and `std`. It must match the VQ checkpoint used.

Robot data must have shape `[T,3]`; `.pt` files must contain tensors and `.npy` files must contain ordinary numeric arrays. The research robot files encode per-frame planar displacements and angular increments despite some scripts calling them velocities. They are summed directly to obtain positions and headings. Do not insert an extra frame-time multiplier into goal labeling for these files. Files are cropped to 192 frames for the recovered model. `swap_xy: true` reproduces the axis swap in both the original robot VQ trainer and combined tokenizer. Generated robot arrays are already in the VQ model's axis convention.

HumanML3D features are not directly joint positions or SMPL parameters. Human rendering requires feature recovery and an appropriate visualization/body-model pipeline. Robot gait is a separate `<trott>` or `<gallop>` token; it does not add a dimension to the three-dimensional trajectory.

## Local recording annotations

For the original robot data layout, `J_1_0.pt` is segment 0 of recording `J_1`; `W_8_9.pt` is segment 9 of recording `W_8`. `M_` marks a mirrored sample; `S_5_`, `S_10_`, `S_15_`, and `S_20_` mark scaled samples. The recovered augmentation reverses the two planar channels for mirrored data and leaves the angular channel unchanged. Scaled variants multiply all three channels by `1 + scale/100`.

`prepare-data` reads base captions from `J_1.txt`/`W_8.txt` and uses `motionglot/resources/mirror_map.json` and `text_templates.json` to reconstruct the appropriate captions for each variant. The checked-in tables match `research/tokenizer/process_robot_motion/sort_robot_dataset.py`. The original tensors already contain their augmentations, so preparation does not apply the transformations a second time. Unknown mirrored captions fail explicitly.

Recordings are deterministically ordered by a seeded hash and assigned approximately 80%/10%/10% train/validation/test, with at least one recording in each split. At least three distinct recordings are required. All variants of each recording follow its assigned split. Human official train/validation/test split files are preserved, and mirrored variants are checked for agreement.

## QUES-CAP

Question line N is paired with motion ID line N of the original ordered `file_names.txt`, accepting the original `N. 000123.txt` format. Do not replace this mapping with a fresh filesystem listing. Multiple question files add questions to the same motions. A shorter question file maps to the first N entries, as in the research implementation; a longer question file is rejected.

Questions are used only for `question-to-human`. Motion captioning uses descriptive captions, so a question does not become the target caption. Human split membership determines question split membership. The recovered 23,842-line question file is not assumed to be identical to every version of the dataset used in the paper.

## Unified instructions

Each output corpus row contains `id`, `group`, `task`, and `text`. Training reads these files lazily and pads within each batch. Task start/end token spellings and gait tokens match the recovered tokenizer. All end tokens must survive tokenization; long samples raise an error instead of being silently truncated.

Goal cells cover `[-7,7) × [-7,7)` at 0.5-metre resolution (28 × 28 = 784 cells). Cell index is `x_index * 28 + y_index`, with x increasing and y centres descending. Goals use the reconstructed robot endpoint, matching the research preparation code. Out-of-grid examples are skipped for this task and counted. The release uses half-open cell boundaries; exact boundary ties can differ from the original nearest-centre calculation.

## Distribution

Training data, checkpoints, evaluator weights, body models and local manifests are ignored by Git. They are excluded from the source ZIP. Obtain third-party datasets through their upstream process. The source package's Apache license applies to the code, not to those assets.
