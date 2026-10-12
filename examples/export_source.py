"""Export the reviewable source tree, respecting the repository's ignore rules.

Run: python examples/export_source.py
No commit or push is performed. Original data and model assets are excluded.
"""
from pathlib import Path
import hashlib
import json
import subprocess
import zipfile


def main():
    root = Path(__file__).resolve().parents[1]
    entries = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=root)
    names = sorted(set(entries.decode().split("\0")) - {""})
    manifest = []
    output = root / "dist" / "MotionGlot-source.zip"
    output.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            path = root / name
            if path.is_symlink():
                raise ValueError("Source export contains a symlink: " + name)
            if not path.is_file():
                continue
            if any(part in ("local", "data", "checkpoints", ".git", "wandb") for part in Path(name).parts):
                raise ValueError("Local asset unexpectedly included: " + name)
            data = path.read_bytes()
            archive.writestr("MotionGlot/" + name, data)
            manifest.append({"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        archive.writestr("MotionGlot/RELEASE_MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"archive": str(output), "source_files": len(manifest),
                      "bytes": output.stat().st_size,
                      "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}, indent=2))


if __name__ == "__main__":
    main()
