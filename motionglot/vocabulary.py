"""Build, validate and fingerprint the shared GPT-2/motion vocabulary."""
import hashlib
import json
from pathlib import Path
from .tasks import SPECIAL_TOKENS


def fingerprint(tokenizer):
    raw = json.dumps(tokenizer.get_vocab(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def validate_vocabulary(tokenizer, human_codes=512, robot_codes=128):
    groups = {"human": ["<human_motion_id_{}>".format(i) for i in range(human_codes)],
              "robot": ["<robot_motion_id_{}>".format(i) for i in range(robot_codes)],
              "goal": ["<goal_id_{}>".format(i) for i in range(784)],
              "special": SPECIAL_TOKENS}
    vocab = tokenizer.get_vocab()
    all_ids = []
    for name, tokens in groups.items():
        for token in tokens:
            if token not in vocab or tokenizer.encode(token, add_special_tokens=False) != [vocab[token]]:
                raise ValueError("Missing or non-atomic token: " + token)
        ids = [vocab[t] for t in tokens]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate IDs in " + name)
        all_ids.extend(ids)
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Shared vocabulary groups overlap")
    return {"vocab_size": len(tokenizer), "human_codes": human_codes, "robot_codes": robot_codes,
            "goal_codes": 784, "fingerprint": fingerprint(tokenizer)}


def build_vocabulary(base, output, human_codes=512, robot_codes=128):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(base)
    for kind, count in (("human_motion", human_codes), ("robot_motion", robot_codes), ("goal", 784)):
        tokenizer.add_tokens(["<{}_id_{}>".format(kind, i) for i in range(count)])
    for token in SPECIAL_TOKENS:
        tokenizer.add_tokens(token, special_tokens=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    metadata = validate_vocabulary(tokenizer, human_codes, robot_codes)
    Path(output).mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output)
    (Path(output) / "motionglot_vocab.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata
