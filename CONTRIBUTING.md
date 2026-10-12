# Contributing

Install the package with `python -m pip install -e '.[test]'` and run `python -m pytest -q`. The tests use the included tokenizer and tiny CPU models; they do not download datasets or GPT weights.

Keep token spellings and token IDs compatible with released checkpoints. Add behavior tests when changing padding masks, task templates, data splits, VQ checkpoint loading or generation. Dataset and checkpoint files belong outside Git; keep experiments under ignored directories.

Explain any difference from the paper in `docs/reproduction.md`. Contributions are provided under the repository's Apache-2.0 source license.
