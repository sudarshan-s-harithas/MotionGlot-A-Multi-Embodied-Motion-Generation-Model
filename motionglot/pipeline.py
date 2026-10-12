"""VQ training, six-task instruction preparation and GPT training."""
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import random
import numpy as np
import torch
from .data import (CausalCollator, InstructionDataset, MotionWindows, load_motion,
                   read_jsonl, validate_manifest)
from .tasks import instruction, motion_string, goal_id
from .vq import VQConfig, load_normalization, load_vq, save_vq
from .vocabulary import fingerprint, validate_vocabulary


def train_vq(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    config = VQConfig(**json.loads(Path(args.config).read_text()))
    rows = list(read_jsonl(args.manifest))
    validate_manifest(rows)
    rows = [r for r in rows if r["embodiment"] == config.embodiment and r["split"] == "train"]
    if config.embodiment == "human" and not args.normalization:
        raise ValueError("Human VQ training requires --normalization")
    norm = load_normalization(args.normalization) if config.embodiment == "human" else None
    dataset = MotionWindows(rows, Path(args.manifest).parent, norm, args.window, config.unit_length)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                         num_workers=args.workers, drop_last=False)
    model = config.build().to(args.device).train()
    optimizer = (torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
                 if config.embodiment == "human" else torch.optim.Adam(model.parameters(), lr=args.lr))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    if norm is not None:
        np.savez(output / "normalization.npz", mean=norm[0], std=norm[1])
    iterator = iter(loader)
    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = batch.to(args.device)
        if config.embodiment == "human":
            rate = args.lr * min(step / max(args.warmup_steps, 1), 1.0)
            rate *= 0.05 ** sum(step >= milestone for milestone in args.lr_milestones)
            for group in optimizer.param_groups:
                group["lr"] = rate
        prediction, commitment, perplexity = model(batch)
        if config.embodiment == "human":
            reconstruction = torch.nn.functional.smooth_l1_loss(prediction, batch)
            velocity = torch.nn.functional.smooth_l1_loss(prediction[..., 4:67], batch[..., 4:67])
            loss = reconstruction + args.commitment_weight * commitment + args.velocity_weight * velocity
        else:
            loss = torch.nn.functional.l1_loss(prediction, batch) + commitment
            loss += torch.nn.functional.l1_loss(prediction.cumsum(1), batch.cumsum(1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(json.dumps({"step": step, "loss": float(loss.detach()), "perplexity": float(perplexity.detach())}), flush=True)
        if step % args.save_every == 0 or step == args.steps:
            save_vq(output / "latest.pth", model, config, step)
    return {"steps": args.steps, "motions": len(dataset)}


@torch.no_grad()
def build_corpus(args):
    from transformers import AutoTokenizer
    rows = list(read_jsonl(args.manifest))
    validate_manifest(rows)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    h_model, h_config = load_vq(args.human_vq, "human", args.device)
    r_model, r_config = load_vq(args.robot_vq, "robot", args.device)
    vocabulary = validate_vocabulary(tokenizer, h_config.nb_code, r_config.nb_code)
    norm = load_normalization(args.normalization)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    streams = {s: (output / (s + ".jsonl")).open("w", encoding="utf-8") for s in ("train", "valid", "test")}
    rng = random.Random(args.seed)
    positive_words = ["joyfully", "cheerfully", "delightfully", "playfully", "merrily"]
    out_of_grid = 0
    try:
        for row in rows:
            embodiment = row["embodiment"]
            model, config = (h_model, h_config) if embodiment == "human" else (r_model, r_config)
            motion = load_motion(row, Path(args.manifest).parent, norm if embodiment == "human" else None, config.unit_length)
            motion = motion.unsqueeze(0).to(args.device)
            codes = model.encode(motion)[0].tolist()
            tokens = motion_string(codes, embodiment)
            samples = []
            for caption in row.get("captions", []):
                if any(t in caption for t in ("<human_motion_id_", "<robot_motion_id_", "<eop>")):
                    raise ValueError("Caption contains reserved tokens")
                if embodiment == "human":
                    samples.append(("text-to-human", instruction("text-to-human", caption, tokens)))
                    samples.append(("human-to-text", instruction("human-to-text", tokens, caption)))
                else:
                    samples.append(("text-to-robot", instruction("text-to-robot", caption, tokens, "trott")))
                    samples.append(("robot-to-text", instruction("robot-to-text", tokens, caption)))
                    if args.sentiment_augmentation and row["split"] == "train":
                        positive = caption.replace("A robot", "A robot " + rng.choice(positive_words), 1)
                        if positive != caption:
                            samples.append(("text-to-robot", instruction("text-to-robot", positive, tokens, "gallop")))
            if embodiment == "human":
                for question in row.get("questions", []):
                    samples.append(("question-to-human", instruction("question-to-human", question, tokens)))
            else:
                # The original code labels goals using VQ reconstructions, not the raw endpoint.
                reconstructed, _, _ = model(motion)
                endpoint = reconstructed[0, :, :2].sum(0).cpu().tolist()
                try:
                    index = goal_id(*endpoint)
                    samples.append(("goal-to-robot", instruction("goal-to-robot", "<goal_id_{}>".format(index), tokens)))
                except ValueError:
                    out_of_grid += 1
            for task, text in samples:
                ids = tokenizer.encode(text, add_special_tokens=False)
                if len(ids) > args.max_length:
                    raise ValueError("{} {} exceeds max_length; preserve the entire instruction".format(row["id"], task))
                item = {"id": row["id"], "group": row["group"], "task": task, "text": text}
                streams[row["split"]].write(json.dumps(item, ensure_ascii=False) + "\n")
                counts[row["split"] + ":" + task] += 1
    finally:
        for stream in streams.values():
            stream.close()
    metadata = {"vocabulary": vocabulary, "counts": dict(counts), "out_of_grid_goals_skipped": out_of_grid,
                "max_length": args.max_length, "seed": args.seed,
                "human_vq_config": asdict(h_config), "robot_vq_config": asdict(r_config)}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def train_gpt(args):
    from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    meta = json.loads((Path(args.corpus) / "metadata.json").read_text())
    if meta["vocabulary"]["fingerprint"] != fingerprint(tokenizer):
        raise ValueError("Corpus and tokenizer have different token ID mappings")
    if args.from_scratch:
        config = GPT2Config.from_pretrained(args.base_model)
        config.vocab_size = len(tokenizer)
        model = GPT2LMHeadModel(config)
    else:
        model = GPT2LMHeadModel.from_pretrained(args.base_model)
        model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.motionglot_vocab_fingerprint = fingerprint(tokenizer)
    if args.max_length > model.config.n_positions or args.max_length < meta["max_length"]:
        raise ValueError("max_length must cover corpus records and fit the GPT context window")
    datasets = {s: InstructionDataset(Path(args.corpus) / (s + ".jsonl"), tokenizer, args.max_length)
                for s in ("train", "valid")}
    if not len(datasets["train"]) or not len(datasets["valid"]):
        raise ValueError("Training and validation corpora must both be nonempty")
    # Validate lengths before beginning an expensive training run.
    for dataset in datasets.values():
        for i in range(len(dataset)):
            dataset[i]
    training = TrainingArguments(output_dir=args.output, max_steps=args.steps,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation, learning_rate=args.lr,
        weight_decay=0.1, warmup_steps=args.warmup_steps, lr_scheduler_type="cosine",
        eval_strategy="steps", eval_steps=args.eval_every, save_steps=args.save_every,
        logging_steps=args.log_every, save_total_limit=3, report_to="none", seed=args.seed,
        use_cpu=args.device == "cpu", fp16=args.fp16, remove_unused_columns=False)
    trainer = Trainer(model=model, args=training, data_collator=CausalCollator(tokenizer),
                      train_dataset=datasets["train"], eval_dataset=datasets["valid"])
    trainer.train(resume_from_checkpoint=args.resume or None)
    trainer.save_model(args.output)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(args.output)
        (Path(args.output) / "motionglot_vocab.json").write_text(json.dumps(meta["vocabulary"], indent=2) + "\n")
    return {"steps": trainer.state.global_step, "train_examples": len(datasets["train"])}
