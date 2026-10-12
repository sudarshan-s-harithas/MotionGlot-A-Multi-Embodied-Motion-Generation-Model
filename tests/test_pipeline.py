"""Behavior checks for vocabulary, data leakage, checkpoints and training."""
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pytest
import torch
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel
from motionglot.cli import main
from motionglot.data import CausalCollator, load_motion, validate_manifest, write_jsonl, read_jsonl, prepare_manifest, recording_splits, robot_recording
from motionglot.tasks import TASKS, goal_center, goal_id, instruction, motion_string
from motionglot.vocabulary import validate_vocabulary, fingerprint
from motionglot.vq import VQConfig, load_vq, save_vq

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session", autouse=True)
def cpu_threads():
    torch.set_num_threads(2)


@pytest.fixture(scope="session")
def tokenizer():
    return AutoTokenizer.from_pretrained(ROOT / "assets/tokenizer", local_files_only=True)


def test_shared_vocabulary(tokenizer):
    meta = validate_vocabulary(tokenizer)
    assert meta["vocab_size"] == 51694
    assert meta["fingerprint"] == "150c7bea17bedbd2c47edda02a98772e46b456d40b7eb97b2ebba5adfc4204c7"
    assert tokenizer.encode("<human_motion_id_511><robot_motion_id_127><goal_id_783>", add_special_tokens=False) == [50768, 50896, 51680]


def test_rebuilt_vocab_preserves_ids(tmp_path, tokenizer):
    from motionglot.vocabulary import build_vocabulary
    base = tmp_path / "base"
    base.mkdir()
    # Rebuild from the plain GPT-2 vocabulary/merges, excluding added tokens.
    from transformers import GPT2Tokenizer
    GPT2Tokenizer(vocab_file=str(ROOT / "assets/tokenizer/vocab.json"),
                  merges_file=str(ROOT / "assets/tokenizer/merges.txt")).save_pretrained(base)
    result = build_vocabulary(str(base), tmp_path / "merged")
    assert result["fingerprint"] == fingerprint(tokenizer)


@pytest.mark.parametrize("task", TASKS)
def test_all_task_templates(tokenizer, task):
    target = "a caption" if TASKS[task].output == "text" else motion_string([0, 1], TASKS[task].output)
    text = instruction(task, "<goal_id_0>" if task == "goal-to-robot" else "input", target)
    ids = tokenizer.encode(text, add_special_tokens=False)
    assert ids[-1] == tokenizer.convert_tokens_to_ids(TASKS[task].end)
    assert text.count("<eop>") == 1


def test_every_goal_cell_round_trips():
    for index in range(784):
        assert goal_id(*goal_center(index)) == index
    assert goal_id(-7, -7) == 27
    for xy in [(7, 0), (0, 7), (-7.01, 0), (float("nan"), 0)]:
        with pytest.raises(ValueError):
            goal_id(*xy)


def test_padding_masks_only_padding(tokenizer):
    token = tokenizer.eos_token_id
    batch = CausalCollator(tokenizer)([
        {"input_ids": [5, token], "attention_mask": [1, 1]},
        {"input_ids": [6], "attention_mask": [1]}])
    assert batch["labels"].tolist() == [[5, token], [6, -100]]


def test_source_split_leakage_is_rejected():
    row = {"id": "one", "group": "recording", "embodiment": "robot", "motion": "x.pt", "split": "train"}
    with pytest.raises(ValueError, match="multiple splits"):
        validate_manifest([row, dict(row, id="two", split="valid")])


def test_recording_families_and_small_splits():
    assert robot_recording("M_W_8_9") == ("W_8", 9, True)
    assert robot_recording("S_10_W_8_9") == ("W_8", 9, False)
    groups = ["W_1", "W_2", "J_1"]
    assert set(recording_splits(groups).values()) == {"train", "valid", "test"}
    assert recording_splits(groups) == recording_splits(list(reversed(groups)))
    with pytest.raises(ValueError):
        recording_splits(["W_1"])


def test_prepare_maps_questions_segments_and_mirrors(tmp_path):
    human = tmp_path / "human"
    (human / "new_joint_vecs").mkdir(parents=True)
    (human / "texts").mkdir()
    for split, name in [("train", "000001"), ("val", "000002"), ("test", "000003")]:
        (human / (split + ".txt")).write_text(name + "\n")
        np.save(human / "new_joint_vecs" / (name + ".npy"), np.zeros((100, 263), dtype=np.float32))
        (human / "texts" / (name + ".txt")).write_text("A person walks.#tokens#0#0\nA person starts walking.#tokens#1#3\n")
    files = tmp_path / "file_names.txt"
    files.write_text("1. 000001.txt\n2. 000002.txt\n3. 000003.txt\n")
    questions = tmp_path / "questions.txt"
    questions.write_text("How does a person walk?\nHow does a person move?\n")
    robot, captions = tmp_path / "robot", tmp_path / "captions"
    robot.mkdir()
    captions.mkdir()
    for recording in ["W_1", "W_2", "J_1"]:
        (captions / (recording + ".txt")).write_text("walks forward.\n")
        for prefix in ["", "M_", "S_10_"]:
            torch.save(torch.zeros(198, 3), robot / (prefix + recording + "_0.pt"))
    output = tmp_path / "manifest.jsonl"
    prepare_manifest(human, robot, captions, [questions], files, output)
    rows = list(read_jsonl(output))
    full = next(r for r in rows if r["id"] == "000001")
    assert full["questions"] == ["How does a person walk?"]
    segment = next(r for r in rows if r["id"] == "000001:20:60")
    assert segment["questions"] == []
    mirrored = next(r for r in rows if r["id"] == "M_W_1_0")
    original = next(r for r in rows if r["id"] == "W_1_0")
    assert mirrored["captions"][0] == "A robot walks backward."
    assert mirrored["group"] == original["group"]
    assert mirrored["split"] == original["split"]


def test_robot_axis_swap_and_length(tmp_path):
    motion = torch.zeros(198, 3)
    motion[:, 0], motion[:, 1] = 1, 2
    path = tmp_path / "robot.pt"
    torch.save(motion, path)
    loaded = load_motion({"motion": str(path), "embodiment": "robot", "swap_xy": True})
    assert loaded.shape == (192, 3)
    assert loaded[0].tolist() == [2, 1, 0]


@pytest.mark.parametrize("embodiment,dimension,codes", [("human", 263, 512), ("robot", 3, 128)])
def test_vq_training_and_checkpoint_round_trip(tmp_path, embodiment, dimension, codes):
    config = VQConfig(embodiment=embodiment, nb_code=codes, code_dim=16, output_emb_width=16, width=16, depth=1)
    model = config.build().train()
    batch = torch.randn(2, 64, dimension)
    prediction, loss, _ = model(batch)
    (torch.nn.functional.l1_loss(prediction, batch) + loss).backward()
    assert any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in model.parameters())
    model.eval()
    expected = model.encode(batch)
    assert expected.shape == (2, 16)
    path = tmp_path / "model.pth"
    save_vq(path, model, config, 1)
    restored, restored_config = load_vq(path, embodiment)
    assert restored_config == config
    assert torch.equal(restored.encode(batch), expected)
    assert restored.forward_decoder(expected).shape == (2, 64, dimension)


@pytest.fixture
def corpus(tmp_path):
    from motionglot.pipeline import build_corpus
    rows = []
    for split in ("train", "valid", "test"):
        for embodiment, dim in (("human", 263), ("robot", 3)):
            path = tmp_path / (split + embodiment + ".npy")
            np.save(path, np.zeros((64, dim), dtype=np.float32))
            rows.append({"id": split + embodiment, "group": split + embodiment, "split": split,
                         "motion": str(path), "embodiment": embodiment, "captions": ["A robot walks." if embodiment == "robot" else "A person walks."],
                         "questions": ["How does a person walk?"] if embodiment == "human" else []})
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl(manifest, rows)
    norm = tmp_path / "normalization.npz"
    np.savez(norm, mean=np.zeros(263), std=np.ones(263))
    checkpoints = {}
    for embodiment, codes in (("human", 512), ("robot", 128)):
        config = VQConfig(embodiment=embodiment, nb_code=codes, code_dim=16, output_emb_width=16, width=16, depth=1)
        model = config.build()
        # Known zero endpoint ensures the synthetic goal sample falls within the grid.
        with torch.no_grad():
            model.vqvae.decoder.model[-1].weight.zero_()
            model.vqvae.decoder.model[-1].bias.zero_()
        path = tmp_path / (embodiment + ".pth")
        save_vq(path, model, config, 0)
        checkpoints[embodiment] = path
    args = SimpleNamespace(manifest=manifest, tokenizer=str(ROOT / "assets/tokenizer"),
        human_vq=checkpoints["human"], robot_vq=checkpoints["robot"], normalization=norm,
        output=tmp_path / "corpus", device="cpu", max_length=256, seed=123, sentiment_augmentation=True)
    metadata = build_corpus(args)
    return args, metadata


def test_corpus_has_six_tasks_in_each_split(corpus):
    args, meta = corpus
    for split in ("train", "valid", "test"):
        rows = list(read_jsonl(args.output / (split + ".jsonl")))
        assert set(r["task"] for r in rows) == set(TASKS)
        assert all(r["group"].startswith(split) for r in rows)
    assert meta["out_of_grid_goals_skipped"] == 0
    assert meta["counts"]["train:text-to-robot"] == 2


@pytest.mark.parametrize("embodiment,nb_code", [("human", 512), ("robot", 128)])
def test_train_vq_command_saves_a_loadable_checkpoint(corpus, tmp_path, embodiment, nb_code):
    args, _ = corpus
    config = tmp_path / "vq_config.json"
    config.write_text(json.dumps({"embodiment": embodiment, "nb_code": nb_code, "code_dim": 16,
                                 "output_emb_width": 16, "width": 16, "depth": 1}))
    output = tmp_path / (embodiment + "_trained")
    main(["train-vq", "--manifest", str(args.manifest), "--config", str(config),
          "--normalization", str(args.normalization), "--device", "cpu", "--steps", "2",
          "--window", "64", "--batch-size", "1", "--output", str(output)])
    _, restored = load_vq(output / "latest.pth", embodiment)
    assert restored.nb_code == nb_code
    if embodiment == "human":
        assert (output / "normalization.npz").is_file()


def test_train_gpt_and_generate_all_six_tasks(corpus, tmp_path):
    args, meta = corpus
    base = tmp_path / "tiny_gpt"
    GPT2Config(n_embd=16, n_layer=1, n_head=2, n_positions=256, vocab_size=51694).save_pretrained(base)
    model_dir = tmp_path / "trained_gpt"
    main(["train-gpt", "--corpus", str(args.output), "--tokenizer", args.tokenizer,
          "--base-model", str(base), "--from-scratch", "--output", str(model_dir),
          "--device", "cpu", "--steps", "2", "--batch-size", "2", "--gradient-accumulation", "1",
          "--eval-every", "1", "--save-every", "2", "--warmup-steps", "0"])
    restored = GPT2LMHeadModel.from_pretrained(model_dir)
    assert restored.config.motionglot_vocab_fingerprint == meta["vocabulary"]["fingerprint"]
    for task in TASKS:
        out = tmp_path / (task + ".json")
        command = ["generate", "--task", task, "--model", str(model_dir), "--human-vq", str(args.human_vq),
                   "--robot-vq", str(args.robot_vq), "--normalization", str(args.normalization),
                   "--device", "cpu", "--max-new-tokens", "8", "--output", str(out)]
        if task == "goal-to-robot":
            command += ["--goal", "1", "1"]
        elif task.endswith("-to-text"):
            embodiment = "human" if task.startswith("human") else "robot"
            command += ["--motion", str(tmp_path / ("train" + embodiment + ".npy"))]
        else:
            command += ["--text", "A person walks."]
        main(command)
        result = json.loads(out.read_text())
        assert result["task"] == task
        if TASKS[task].output == "text":
            assert "caption" in result
        else:
            assert np.load(out.with_suffix(".npy")).shape[1] == (263 if TASKS[task].output == "human" else 3)
