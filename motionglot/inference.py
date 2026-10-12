"""Generate any of the six tasks using recovered or newly trained models."""
import json
from pathlib import Path
import numpy as np
import torch
from .data import load_motion
from .tasks import TASKS, prompt, motion_string, goal_id
from .vq import load_normalization, load_vq
from .vocabulary import fingerprint, validate_vocabulary


def decode_motion_ids(ids, tokenizer, embodiment, nb_code):
    mapping = {tokenizer.convert_tokens_to_ids("<{}_motion_id_{}>".format(embodiment, i)): i
               for i in range(nb_code)}
    codes = [mapping[i] for i in ids if i in mapping]
    if not codes:
        raise ValueError("Generation returned no motion tokens")
    return codes


@torch.no_grad()
def generate(args):
    from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model, local_files_only=True)
    vocabulary = validate_vocabulary(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True).to(args.device).eval()
    if model.get_input_embeddings().num_embeddings != len(tokenizer):
        raise ValueError("GPT and tokenizer vocabulary sizes differ")
    expected = getattr(model.config, "motionglot_vocab_fingerprint", None)
    if expected is not None and expected != fingerprint(tokenizer):
        raise ValueError("GPT and tokenizer use different token ID mappings")
    spec = TASKS[args.task]
    decoder = config = norm = None
    if args.task in ("human-to-text", "robot-to-text"):
        embodiment = "human" if args.task == "human-to-text" else "robot"
        path = args.human_vq if embodiment == "human" else args.robot_vq
        if not args.motion or not path:
            raise ValueError("Captioning needs --motion and the corresponding VQ checkpoint")
        encoder, enc_config = load_vq(path, embodiment, args.device)
        norm = load_normalization(args.normalization) if embodiment == "human" else None
        row = {"motion": args.motion, "embodiment": embodiment, "swap_xy": args.swap_xy}
        motion = load_motion(row, normalization=norm, unit_length=enc_config.unit_length).unsqueeze(0).to(args.device)
        source = motion_string(encoder.encode(motion)[0].tolist(), embodiment)
    elif args.task == "goal-to-robot":
        if args.goal is None:
            raise ValueError("Goal generation needs --goal X Y")
        source = "<goal_id_{}>".format(goal_id(*args.goal))
    else:
        source = args.text
    if spec.output in ("human", "robot"):
        checkpoint = args.human_vq if spec.output == "human" else args.robot_vq
        if not checkpoint:
            raise ValueError("Motion generation needs the corresponding VQ checkpoint")
        decoder, config = load_vq(checkpoint, spec.output, args.device)
        expected_codes = vocabulary[spec.output + "_codes"]
        if config.nb_code != expected_codes:
            raise ValueError("VQ codebook and GPT vocabulary differ")
        if spec.output == "human":
            norm = load_normalization(args.normalization)
    encoded = tokenizer(prompt(args.task, source), return_tensors="pt", add_special_tokens=False)
    encoded = {k: v.to(args.device) for k, v in encoded.items()}
    prompt_length = encoded["input_ids"].shape[1]
    if prompt_length + args.max_new_tokens > model.config.max_position_embeddings:
        raise ValueError("Prompt and requested response exceed GPT context window")
    end = tokenizer.convert_tokens_to_ids(spec.end)
    if spec.output == "text":
        allowed = list(range(tokenizer.vocab_size)) + [end]
        gaits = []
    else:
        allowed = [tokenizer.convert_tokens_to_ids("<{}_motion_id_{}>".format(spec.output, i)) for i in range(config.nb_code)]
        allowed += [end]
        gaits = [tokenizer.convert_tokens_to_ids(t) for t in ("<trott>", "<gallop>")] if args.task == "text-to-robot" else []

    class OutputVocabulary(LogitsProcessor):
        def __call__(self, input_ids, scores):
            active = allowed + (gaits if input_ids.shape[1] == prompt_length else [])
            if spec.output != "text" and input_ids.shape[1] == prompt_length:
                active = [i for i in active if i != end]
            if spec.output != "text" and input_ids.shape[1] == prompt_length + 1 and int(input_ids[0, -1]) in gaits:
                active = [i for i in active if i != end]
            mask = torch.full_like(scores, float("-inf"))
            mask[:, active] = scores[:, active]
            return mask

    kwargs = {"max_new_tokens": args.max_new_tokens, "do_sample": args.sample,
              "eos_token_id": end, "pad_token_id": tokenizer.eos_token_id,
              "logits_processor": LogitsProcessorList([OutputVocabulary()])}
    if args.sample:
        kwargs.update(temperature=args.temperature, top_p=args.top_p)
    generated = model.generate(**encoded, **kwargs)[0, prompt_length:].tolist()
    terminated = bool(generated and generated[-1] == end)
    result = {"task": args.task, "terminated": terminated, "tokens": tokenizer.decode(generated, skip_special_tokens=False)}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if decoder is not None:
        codes = decode_motion_ids(generated, tokenizer, spec.output, config.nb_code)
        decoded = decoder.forward_decoder(torch.tensor([codes], device=args.device))[0].cpu().numpy()
        if norm is not None:
            decoded = decoded * norm[1] + norm[0]
        np.save(output.with_suffix(".npy"), decoded)
        result.update(motion_file=output.with_suffix(".npy").name, frames=len(decoded))
        if spec.output == "robot":
            result["gait"] = next((t for t in ("trott", "gallop") if "<{}>".format(t) in result["tokens"]), None)
    else:
        result["caption"] = tokenizer.decode([i for i in generated if i != end], skip_special_tokens=True)
    output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    return result
