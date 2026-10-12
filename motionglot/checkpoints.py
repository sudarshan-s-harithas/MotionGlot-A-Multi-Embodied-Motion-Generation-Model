"""Download and verify the complete public research checkpoint bundle."""
import hashlib
import json
from pathlib import Path

MODEL_REPO = "sudarshanharithas/MotionGlot"


def validate_bundle(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    for item in manifest["files"]:
        path = directory / item["path"]
        try:
            path.resolve().relative_to(directory.resolve())
        except ValueError as error:
            raise ValueError("Invalid checkpoint manifest path") from error
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(block)
        if path.stat().st_size != item["bytes"] or digest.hexdigest() != item["sha256"]:
            raise ValueError("Checkpoint file does not match its manifest: " + item["path"])
    return manifest


def download_bundle(output, repo_id=MODEL_REPO, revision="v0.1.0"):
    from huggingface_hub import snapshot_download
    directory = snapshot_download(repo_id=repo_id, revision=revision, local_dir=output,
                                  allow_patterns=["gpt/*", "human_vq.pth", "robot_vq.pth", "normalization.npz", "manifest.json", "LICENSE", "NOTICE", "README.md", "licenses/*"])
    validate_bundle(directory)
    return Path(directory)
