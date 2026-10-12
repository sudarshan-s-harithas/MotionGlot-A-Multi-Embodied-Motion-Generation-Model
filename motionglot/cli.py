"""Command-line entry points. Import training dependencies only when needed."""
import argparse
import json
from .tasks import TASKS


def main(argv=None):
    parser = argparse.ArgumentParser(description="MotionGlot: tokenization, six-task preparation, GPT training and inference")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare-data", help="Create a manifest from HumanML3D and local robot annotations")
    for name in ("human-root", "robot-root", "robot-captions", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--questions", nargs="+", default=[])
    p.add_argument("--file-names")
    p.add_argument("--robot-caption-mode", choices=("recording", "per-motion"), default="recording")
    p.add_argument("--seed", type=int, default=123)
    p = sub.add_parser("build-vocab", help="Append motion, goal, task and gait tokens to GPT-2")
    p.add_argument("--base", default="openai-community/gpt2")
    p.add_argument("--output", required=True)
    p.add_argument("--human-codes", type=int, default=512)
    p.add_argument("--robot-codes", type=int, default=128)
    p = sub.add_parser("train-vq", help="Train a human or robot VQ-VAE on training motions")
    for name in ("manifest", "config", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--normalization")
    p.add_argument("--device", default="cuda")
    p.add_argument("--steps", type=int, default=200000)
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--lr-milestones", nargs="+", type=int, default=[50000, 400000])
    p.add_argument("--commitment-weight", type=float, default=0.02)
    p.add_argument("--velocity-weight", type=float, default=0.1)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--seed", type=int, default=123)
    p = sub.add_parser("build-corpus", help="Tokenize trajectories and create all six instruction tasks")
    for name in ("manifest", "tokenizer", "human-vq", "robot-vq", "normalization", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--sentiment-augmentation", action="store_true")
    p.add_argument("--seed", type=int, default=123)
    p = sub.add_parser("train-gpt", help="Train GPT-2 on the unified instruction corpus")
    for name in ("corpus", "tokenizer", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--base-model", default="openai-community/gpt2")
    p.add_argument("--from-scratch", action="store_true")
    p.add_argument("--resume")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--eval-batch-size", type=int, default=5)
    p.add_argument("--gradient-accumulation", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--seed", type=int, default=123)
    p = sub.add_parser("generate", help="Run one of the six tasks")
    p.add_argument("--task", choices=tuple(TASKS), required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--tokenizer")
    p.add_argument("--human-vq")
    p.add_argument("--robot-vq")
    p.add_argument("--normalization")
    p.add_argument("--text")
    p.add_argument("--motion")
    p.add_argument("--swap-xy", action="store_true")
    p.add_argument("--goal", nargs=2, type=float)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--sample", action="store_true")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=123)
    args = parser.parse_args(argv)
    for name in ("steps", "batch_size", "save_every", "log_every", "max_length", "max_new_tokens"):
        if hasattr(args, name) and getattr(args, name) <= 0:
            parser.error(name + " must be positive")
    if args.command == "prepare-data":
        from .data import prepare_manifest
        result = prepare_manifest(args.human_root, args.robot_root, args.robot_captions,
            args.questions, args.file_names, args.output, args.seed, args.robot_caption_mode)
    elif args.command == "build-vocab":
        from .vocabulary import build_vocabulary
        result = build_vocabulary(args.base, args.output, args.human_codes, args.robot_codes)
    elif args.command == "generate":
        from .inference import generate
        result = generate(args)
    else:
        from . import pipeline
        result = getattr(pipeline, args.command.replace("-", "_"))(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
