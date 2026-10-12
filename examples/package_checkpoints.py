"""Create a verified checkpoint archive for GitHub Releases."""
import argparse
import hashlib
from pathlib import Path
import tarfile
from motionglot.checkpoints import validate_bundle

parser = argparse.ArgumentParser(description="Package the MotionGlot checkpoint bundle")
parser.add_argument("--input", required=True)
parser.add_argument("--output", default="dist/motionglot-checkpoints-v0.1.0.tar.gz")
args = parser.parse_args()
directory = Path(args.input)
manifest = validate_bundle(directory)
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
with tarfile.open(output, "w:gz", compresslevel=1) as archive:
    for item in manifest["files"]:
        archive.add(directory / item["path"], arcname="release-v0.1.0/" + item["path"], recursive=False)
    archive.add(directory / "manifest.json", arcname="release-v0.1.0/manifest.json", recursive=False)
    for name in ["LICENSE", "NOTICE", "README.md"]:
        if (directory / name).is_file():
            archive.add(directory / name, arcname="release-v0.1.0/" + name, recursive=False)
    if (directory / "licenses").is_dir():
        for path in sorted((directory / "licenses").glob("*.txt")):
            archive.add(path, arcname="release-v0.1.0/licenses/" + path.name, recursive=False)
digest = hashlib.sha256()
with output.open("rb") as stream:
    for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
        digest.update(block)
output.with_suffix(output.suffix + ".sha256").write_text(digest.hexdigest() + "  " + output.name + "\n")
print(output, output.stat().st_size, digest.hexdigest())
