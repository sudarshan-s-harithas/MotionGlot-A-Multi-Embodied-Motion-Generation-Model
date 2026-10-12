"""Explicit motion manifests, source-level splits and disk-backed GPT data."""
import hashlib
import json
import re
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except ValueError as error:
                raise ValueError("Invalid JSON in {}:{}".format(path, number)) from error


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def group_split(group, seed=123):
    value = int(hashlib.sha256((str(seed) + ":" + group).encode()).hexdigest()[:8], 16) % 100
    return "test" if value < 10 else "valid" if value < 20 else "train"


def robot_recording(name):
    recording, segment = name.rsplit("_", 1)
    mirrored = recording.startswith("M_")
    recording = re.sub(r"^(M_|S_\d+_)", "", recording)
    return recording, int(segment), mirrored


def recording_splits(groups, seed=123):
    """Keep augmentation families together and guarantee all three split sets."""
    groups = sorted(set(groups), key=lambda g: hashlib.sha256((str(seed) + ":" + g).encode()).hexdigest())
    if len(groups) < 3:
        raise ValueError("At least three robot recordings are needed for train/valid/test splits")
    count = max(1, round(len(groups) * 0.1))
    return {group: "test" if i < count else "valid" if i < count * 2 else "train"
            for i, group in enumerate(groups)}


def load_motion(row, root=None, normalization=None, unit_length=4):
    path = Path(row["motion"])
    if not path.is_absolute():
        path = Path(root or ".") / path
    if path.suffix == ".npy":
        motion = np.load(path, allow_pickle=False)
    elif path.suffix == ".pt":
        motion = torch.load(path, map_location="cpu", weights_only=True)
        if not torch.is_tensor(motion):
            raise ValueError("Robot .pt files must contain a tensor")
        motion = motion.numpy()
    else:
        raise ValueError("Motion must be a .npy array or .pt tensor")
    motion = np.asarray(motion, dtype=np.float32)
    dim = 263 if row["embodiment"] == "human" else 3
    if motion.ndim != 2 or motion.shape[1] != dim or not np.isfinite(motion).all():
        raise ValueError("{} requires finite [frames,{}] data".format(path, dim))
    motion = motion[int(row.get("start_frame", 0)):row.get("end_frame")].copy()
    if row["embodiment"] == "robot":
        motion = motion[:192]
        if row.get("swap_xy", False):
            motion = motion[:, [1, 0, 2]]
    elif normalization is not None:
        mean, std = normalization
        motion = (motion - mean) / std
    usable = len(motion) // unit_length * unit_length
    if usable < unit_length:
        raise ValueError("Motion {} is shorter than one VQ token".format(path))
    return torch.from_numpy(motion[:usable].copy())


def validate_manifest(rows):
    seen = set()
    groups = {}
    counts = {}
    for row in rows:
        for key in ("id", "group", "embodiment", "motion", "split"):
            if key not in row:
                raise ValueError("Missing manifest field " + key)
        if row["embodiment"] not in ("human", "robot") or row["split"] not in ("train", "valid", "test"):
            raise ValueError("Unknown embodiment or split")
        key = (row["embodiment"], row["id"])
        if key in seen:
            raise ValueError("Duplicate motion ID " + str(key))
        seen.add(key)
        group = (row["embodiment"], row["group"])
        if group in groups and groups[group] != row["split"]:
            raise ValueError("Source group appears in multiple splits: " + str(group))
        groups[group] = row["split"]
        for kind in ("captions", "questions"):
            if not isinstance(row.get(kind, []), list) or any(not isinstance(x, str) or not x.strip() for x in row.get(kind, [])):
                raise ValueError(kind + " must contain nonempty strings")
        name = row["embodiment"] + ":" + row["split"]
        counts[name] = counts.get(name, 0) + 1
    if not seen:
        raise ValueError("Manifest is empty")
    return counts


def prepare_manifest(human_root, robot_root, robot_captions, questions, file_names,
                     output, seed=123, robot_caption_mode="recording"):
    rows = []
    human_root, robot_root = Path(human_root), Path(robot_root)
    question_map = {}
    if questions:
        if not file_names:
            raise ValueError("Question files require the original ordered file_names.txt")
        names = [line.strip().split(". ", 1)[-1].removesuffix(".txt") if hasattr(str, "removesuffix")
                 else line.strip().split(". ", 1)[-1].replace(".txt", "")
                 for line in Path(file_names).read_text().splitlines()]
        if len(names) != len(set(names)):
            raise ValueError("Question file mapping has duplicate IDs")
        for path in questions:
            lines = Path(path).read_text().splitlines()
            if len(lines) > len(names):
                raise ValueError("Question file has more lines than its ID mapping")
            for name, question in zip(names, lines):
                if question.strip():
                    question_map.setdefault(name, []).append(question.strip())
    # Preserve HumanML3D official splits. val.txt is validation, test.txt remains held out.
    split_ids = {}
    for split, filename in (("train", "train.txt"), ("valid", "val.txt"), ("test", "test.txt")):
        for name in (human_root / filename).read_text().splitlines():
            name = name.strip()
            if not name:
                continue
            if name in split_ids:
                raise ValueError("HumanML3D ID occurs in multiple splits: " + name)
            split_ids[name] = split
    # Mirrored M* motions share the source split with the original.
    for name, split in split_ids.items():
        original = name[1:] if name.startswith("M") else name
        if original in split_ids and split_ids[original] != split:
            raise ValueError("Mirrored HumanML3D motion crosses splits: " + name)
        motion = human_root / "new_joint_vecs" / (name + ".npy")
        text = human_root / "texts" / (name + ".txt")
        if not motion.exists() or not text.exists():
            continue
        frames = len(np.load(motion, mmap_mode="r", allow_pickle=False))
        full = []
        segments = {}
        for line in text.read_text().splitlines():
            fields = line.split("#")
            if not fields[0].strip():
                continue
            start, end = (float(fields[2]), float(fields[3])) if len(fields) >= 4 else (0, 0)
            if not np.isfinite(start) or not np.isfinite(end):
                start, end = 0, 0
            if start == 0 and end == 0:
                full.append(fields[0].strip())
            else:
                a, b = max(0, int(start * 20)), min(frames, int(end * 20))
                if b - a >= 40:
                    segments.setdefault((a, b), []).append(fields[0].strip())
        base = {"group": original, "embodiment": "human", "motion": str(motion.resolve()), "split": split}
        if 40 <= frames < 200 and full:
            rows.append(dict(base, id=name, captions=full, questions=question_map.get(name, [])))
        for (a, b), captions in segments.items():
            if b - a < 200:
                rows.append(dict(base, id=name + ":{}:{}".format(a, b), captions=captions,
                                 questions=[], start_frame=a, end_frame=b))
    cache = {}
    resources = Path(__file__).parent / "resources"
    mirror_map = json.loads((resources / "mirror_map.json").read_text())
    templates = json.loads((resources / "text_templates.json").read_text())
    robot_files = sorted(robot_root.glob("*.pt"))
    robot_splits = recording_splits([robot_recording(p.stem)[0] for p in robot_files], seed)
    for motion in robot_files:
        name = motion.stem
        recording, segment, mirrored = robot_recording(name)
        if robot_caption_mode == "recording":
            if recording not in cache:
                cache[recording] = (Path(robot_captions) / (recording + ".txt")).read_text().splitlines()
            captions = cache[recording]
            if segment < 0 or segment >= len(captions):
                raise ValueError("Robot segment has no caption: " + name)
            caption = captions[segment].strip().rstrip(".")
            if mirrored:
                if caption not in mirror_map or mirror_map[caption] == "—":
                    raise ValueError("No verified mirrored caption for " + name)
                caption = mirror_map[caption]
            captions = ["A robot " + caption + "."]
            captions.extend("A robot " + s.strip() for s in templates.get(caption, []) if s.strip())
        else:
            captions = [s.strip() for s in (Path(robot_captions) / (name + ".txt")).read_text().splitlines() if s.strip()]
        rows.append({"id": name, "group": recording, "embodiment": "robot", "motion": str(motion.resolve()),
                     "split": robot_splits[recording], "swap_xy": True, "captions": captions, "questions": []})
    counts = validate_manifest(rows)
    write_jsonl(output, rows)
    return counts


class MotionWindows(Dataset):
    def __init__(self, rows, root, normalization, window, unit_length):
        self.rows, self.root, self.normalization = rows, root, normalization
        self.window, self.unit_length = window, unit_length
        if window <= 0 or window % unit_length:
            raise ValueError("Window must be a positive multiple of VQ unit_length")
        self.rows = [r for r in rows if len(load_motion(r, root, normalization, unit_length)) >= window]
        if not self.rows:
            raise ValueError("No training motions are at least one window long")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        motion = load_motion(self.rows[index], self.root, self.normalization, self.unit_length)
        start = int(torch.randint(len(motion) - self.window + 1, (1,)))
        return motion[start:start + self.window]


class InstructionDataset(Dataset):
    """Index offsets only; avoid materializing multi-gigabyte training tensors."""
    def __init__(self, path, tokenizer, max_length):
        self.path, self.tokenizer, self.max_length = Path(path), tokenizer, max_length
        self.offsets = []
        with self.path.open("rb") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                if line.strip():
                    self.offsets.append(offset)

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index):
        with self.path.open("rb") as stream:
            stream.seek(self.offsets[index])
            row = json.loads(stream.readline())
        ids = self.tokenizer.encode(row["text"], add_special_tokens=False)
        if len(ids) > self.max_length:
            raise ValueError("Instruction exceeds context window; increase max_length, do not cut motion/end tokens")
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


class CausalCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, examples):
        batch = self.tokenizer.pad(examples, padding=True, return_tensors="pt")
        batch["labels"] = batch["input_ids"].clone()
        # GPT-2 EOS is also padding. Mask by attention, preserving genuine EOS targets.
        batch["labels"][batch["attention_mask"] == 0] = -100
        return batch
