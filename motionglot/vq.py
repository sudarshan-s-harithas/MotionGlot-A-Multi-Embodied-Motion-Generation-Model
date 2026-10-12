"""Portable loading and training interfaces for the recovered VQ-VAE."""
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import torch
from .models.vqvae import HumanVQVAE


@dataclass
class VQConfig:
    embodiment: str = "human"
    nb_code: int = 512
    code_dim: int = 512
    output_emb_width: int = 512
    down_t: int = 2
    stride_t: int = 2
    width: int = 512
    depth: int = 3
    dilation_growth_rate: int = 3
    activation: str = "relu"
    norm: str = None
    quantizer: str = "ema_reset"
    mu: float = 0.99

    @property
    def unit_length(self):
        return self.stride_t ** self.down_t

    def build(self):
        if self.embodiment not in ("human", "robot") or self.stride_t != 2:
            raise ValueError("Use human/robot embodiment and stride_t=2 with the recovered decoder")
        if self.code_dim != self.output_emb_width:
            raise ValueError("code_dim must equal output_emb_width")
        args = SimpleNamespace(dataname="t2m" if self.embodiment == "human" else "spot",
                               quantizer=self.quantizer, mu=self.mu)
        return HumanVQVAE(args, self.nb_code, self.code_dim, self.output_emb_width,
                          self.down_t, self.stride_t, self.width, self.depth,
                          self.dilation_growth_rate, self.activation, self.norm)


def load_vq(path, embodiment, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint.get("net", checkpoint)
    state = {k.removeprefix("module.") if hasattr(k, "removeprefix") else
             (k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    metadata = json.loads(checkpoint["config_json"]) if "config_json" in checkpoint else checkpoint.get("config")
    config = VQConfig(**metadata) if metadata else VQConfig(
        embodiment=embodiment, nb_code=state["vqvae.quantizer.codebook"].shape[0])
    if config.embodiment != embodiment:
        raise ValueError("Checkpoint embodiment does not match the requested model")
    model = config.build().to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, config


def save_vq(path, model, config, step):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # Torch 1.13's weights-only reader cannot decode floats inside a metadata dict.
    torch.save({"net": model.state_dict(), "config_json": json.dumps(asdict(config)), "step": step}, path)


def load_normalization(path):
    p = Path(path)
    if p.is_dir():
        mean_path = next((p / n for n in ("mean.npy", "Mean.npy") if (p / n).exists()), None)
        std_path = next((p / n for n in ("std.npy", "Std.npy") if (p / n).exists()), None)
        if mean_path is None or std_path is None:
            raise ValueError("Normalization folder needs mean.npy/std.npy (or Mean.npy/Std.npy)")
        mean, std = np.load(mean_path), np.load(std_path)
    else:
        with np.load(p, allow_pickle=False) as data:
            mean, std = data["mean"], data["std"]
    if mean.shape != (263,) or std.shape != (263,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError("Human normalization requires finite 263-dimensional means and positive stds")
    return mean.astype(np.float32), std.astype(np.float32)
