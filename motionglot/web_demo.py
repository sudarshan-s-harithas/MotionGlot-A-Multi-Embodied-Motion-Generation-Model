"""Local Gradio demo with real inference, video rendering and downloads."""
import argparse
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import numpy as np
import torch
from .checkpoints import download_bundle, validate_bundle
from .inference import InferenceRuntime, generate
from .rendering import render_human_video, robot_figure


class MotionDemo:
    def __init__(self, checkpoint_dir, device="cpu", output_dir=None):
        self.checkpoint_dir = Path(checkpoint_dir)
        validate_bundle(self.checkpoint_dir)
        self.device = device
        self.runtime = InferenceRuntime(self.checkpoint_dir / "gpt", device=device)
        self.output_dir = Path(output_dir or Path(tempfile.gettempdir()) / "motionglot-demo")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def run(self, task, text=None, goal=None, motion=None, swap_xy=False, seed=123):
        directory = Path(tempfile.mkdtemp(prefix="request-", dir=self.output_dir))
        args = SimpleNamespace(task=task, model=str(self.checkpoint_dir / "gpt"), tokenizer=None,
            human_vq=self.checkpoint_dir / "human_vq.pth", robot_vq=self.checkpoint_dir / "robot_vq.pth",
            normalization=self.checkpoint_dir / "normalization.npz", device=self.device,
            text=text, goal=goal, motion=motion, swap_xy=swap_xy, seed=int(seed), output=str(directory / "result"),
            sample=True, max_new_tokens=64, temperature=0.8, top_p=0.95)
        with self.lock:
            result = generate(args, self.runtime)
            npy = directory / "result.npy"
            video = None
            figure = None
            if task in ("text-to-human", "question-to-human"):
                video = render_human_video(np.load(npy, allow_pickle=False), directory / "motion.mp4")
            elif task in ("text-to-robot", "goal-to-robot"):
                figure = robot_figure(np.load(npy, allow_pickle=False), goal=goal)
        result["seed"] = int(seed)
        status = "{} frames generated".format(result["frames"]) if "frames" in result else "Caption generated"
        if result.get("gait"):
            status += " · gait: " + result["gait"]
        if not result["terminated"]:
            status += " · generation limit reached"
        files = [str(directory / "result.json")]
        if npy.exists():
            files.insert(0, str(npy))
        return result, video, figure, files, status


def validate_upload(path, embodiment):
    path = Path(path)
    if path.suffix.lower() != ".npy" or path.stat().st_size > 1_000_000:
        raise ValueError("Upload a .npy motion array smaller than 1 MB")
    motion = np.load(path, allow_pickle=False)
    dimension = 263 if embodiment == "human" else 3
    numeric = np.issubdtype(motion.dtype, np.floating) or np.issubdtype(motion.dtype, np.integer)
    if motion.ndim != 2 or motion.shape[1] != dimension or not 4 <= len(motion) <= 256 or not numeric or not np.isfinite(motion).all():
        raise ValueError("Expected finite [4–256 frames,{} features] data".format(dimension))
    return str(path)


def create_app(engine):
    import gradio as gr

    def human(text, mode, seed):
        if not text or not text.strip() or len(text) > 1000:
            raise gr.Error("Enter a motion description or question, up to 1,000 characters")
        task = "question-to-human" if mode == "Situational question" else "text-to-human"
        try:
            _, video, _, files, status = engine.run(task, text=text.strip(), seed=seed)
            return video, files, status
        except ValueError as error:
            raise gr.Error(str(error)) from error

    def robot(text, mode, x, y, seed):
        task = "goal-to-robot" if mode == "Reach a goal" else "text-to-robot"
        if task == "text-to-robot" and (not text or not text.strip() or len(text) > 1000):
            raise gr.Error("Enter a robot motion description, up to 1,000 characters")
        goal = (float(x), float(y)) if task == "goal-to-robot" else None
        try:
            _, _, figure, files, status = engine.run(task, text=text.strip(), goal=goal, seed=seed)
            return figure, files, status
        except ValueError as error:
            raise gr.Error(str(error)) from error

    def caption(path, embodiment, swap_xy, seed):
        if not path:
            raise gr.Error("Upload a motion feature array first")
        embodiment = embodiment.lower()
        try:
            path = validate_upload(path, embodiment)
            result, _, _, files, status = engine.run(embodiment + "-to-text", motion=path, swap_xy=swap_xy, seed=seed)
            return result["caption"], files, status
        except (ValueError, OSError) as error:
            raise gr.Error(str(error)) from error

    with gr.Blocks(title="MotionGlot Demo", delete_cache=(3600, 3600)) as app:
        gr.Markdown("# MotionGlot\nGenerate motion for humans and quadruped robots, or describe a motion in words.\n\n"
            "[Paper](https://arxiv.org/abs/2410.16623) · "
            "[Project](https://ivl.cs.brown.edu/research/motionglot.html) · "
            "[Code and checkpoints](https://github.com/sudarshan-s-harithas/MotionGlot-A-Multi-Embodied-Motion-Generation-Model)")
        with gr.Tab("Human motion"):
            mode = gr.Radio(["Motion description", "Situational question"], value="Motion description", label="Prompt type")
            text = gr.Textbox(label="Describe a motion or ask a question", placeholder="A person walks forward and sits.", lines=2, max_length=1000)
            seed = gr.Number(value=123, precision=0, label="Random seed")
            button = gr.Button("Generate human motion", variant="primary")
            video = gr.Video(label="Generated motion", autoplay=True)
            files = gr.File(label="Download motion features and metadata", file_count="multiple")
            status = gr.Textbox(label="Result", interactive=False)
            gr.Examples(examples=[["A person walks forward.", "Motion description", 123],
                ["A person performs a cartwheel.", "Motion description", 123],
                ["How does a person practice in a gymnastics class?", "Situational question", 123]], inputs=[text, mode, seed])
            button.click(human, inputs=[text, mode, seed], outputs=[video, files, status], api_name="generate_human")
            text.submit(human, inputs=[text, mode, seed], outputs=[video, files, status], api_name=False)
        with gr.Tab("Robot motion"):
            mode = gr.Radio(["Motion description", "Reach a goal"], value="Motion description", label="Task")
            text = gr.Textbox(label="Robot motion description", value="A robot walks forward and turns right.", max_length=1000)
            with gr.Row():
                x = gr.Number(value=1, label="Goal x (m)")
                y = gr.Number(value=1, label="Goal z (m)")
            seed = gr.Number(value=123, precision=0, label="Random seed")
            button = gr.Button("Generate robot motion", variant="primary")
            plot = gr.Plot(label="Generated trajectory")
            files = gr.File(label="Download trajectory and metadata", file_count="multiple")
            status = gr.Textbox(label="Result", interactive=False)
            gr.Markdown("Goals lie in the 14 × 14 metre grid, from −7 m (inclusive) to 7 m (exclusive).")
            button.click(robot, inputs=[text, mode, x, y, seed], outputs=[plot, files, status], api_name="generate_robot")
        with gr.Tab("Motion captioning"):
            gr.Markdown("Upload a `.npy` feature array: human **263** features per frame, or robot **3** features per frame. "
                "Use the denormalized feature downloads from the generation tabs to try captioning.")
            embodiment = gr.Radio(["Human", "Robot"], value="Human", label="Embodiment")
            motion = gr.File(label="Motion feature array", file_types=[".npy"], type="filepath")
            swap = gr.Checkbox(value=False, label="Swap robot planar axes (for original QUAD-LOCO files)")
            seed = gr.Number(value=123, precision=0, label="Random seed")
            button = gr.Button("Generate caption", variant="primary")
            output = gr.Textbox(label="Generated caption", interactive=False)
            files = gr.File(label="Download caption and metadata", file_count="multiple")
            status = gr.Textbox(label="Result", interactive=False)
            button.click(caption, inputs=[motion, embodiment, swap, seed], outputs=[output, files, status], api_name="caption_motion")
        gr.Markdown("Research demo by Sudarshan Harithas and Srinath Sridhar. Motion generation is sampled; "
                    "change the seed to explore alternatives. Robot outputs are displayed as trajectories.")
    return app.queue(default_concurrency_limit=1, max_size=16)


def main(argv=None):
    if sys.version_info < (3, 10):
        raise SystemExit("The interactive demo requires Python 3.10 or newer; the research CLI also supports Python 3.8.")
    parser = argparse.ArgumentParser(description="Run the MotionGlot interactive demo locally")
    parser.add_argument("--checkpoint-dir", default="checkpoints/release-v0.1.0")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create a temporary public Gradio link")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    if args.threads <= 0:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    if not (Path(args.checkpoint_dir) / "manifest.json").exists():
        print("Downloading the public MotionGlot checkpoint bundle (approximately 660 MB)...", flush=True)
        download_bundle(args.checkpoint_dir)
    engine = MotionDemo(args.checkpoint_dir, args.device)
    app = create_app(engine)
    app.launch(server_name=args.host, server_port=args.port, share=args.share,
               allowed_paths=[str(engine.output_dir.resolve())], max_file_size="1mb")


if __name__ == "__main__":
    main()
