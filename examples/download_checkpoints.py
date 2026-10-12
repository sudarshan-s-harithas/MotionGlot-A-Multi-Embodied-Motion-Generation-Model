import argparse
from motionglot.checkpoints import download_bundle

parser = argparse.ArgumentParser(description="Download and verify MotionGlot research checkpoints")
parser.add_argument("--output", default="checkpoints/release-v0.1.0")
parser.add_argument("--revision", default="v0.1.0")
args = parser.parse_args()
print(download_bundle(args.output, revision=args.revision))
