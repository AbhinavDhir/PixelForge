"""PixelForge configuration, device selection and reproducibility helpers.

Every other module imports its defaults from here, so the CLI scripts, the
Streamlit app and the Colab notebook all agree on hyperparameters and paths.
"""
from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent

# TinyHero (AgaMiko/pixel_character_generator). data.zip is a regular file in the
# repo (not Git LFS), so the raw URL returns the 7.3 MB archive directly.
DATASET_URL = "https://github.com/AgaMiko/pixel_character_generator/raw/master/data.zip"
DATASET_REPO = "https://github.com/AgaMiko/pixel_character_generator"

# Folder names inside data.zip -> viewing direction (verified against the archive).
VIEW_NAMES = {0: "back", 1: "left", 2: "front", 3: "right"}

IMAGE_SIZE = 64
NUM_CHANNELS = 3


@dataclass
class Config:
    # --- data ---
    data_dir: str = str(PROJECT_ROOT / "data")
    views: tuple = (0, 1, 2, 3)            # train on all four directions by default
    background: tuple = (0, 0, 0)         # RGBA sprites are composited onto black (white made this DCGAN mode-collapse, see README)
    subset: int | None = None             # use only N images (smoke tests)
    hflip: bool = False                   # optional horizontal-flip augmentation

    # --- model ---
    conditional: bool = True              # G(z, y) / projection D(x, y) on [view, hair, outfit] labels
    n_classes: tuple = (4, 9, 9)          # classes per attribute (see attributes.py)
    d_cond: str = "concat"                # how D sees labels: "concat" (label maps as input channels) or "projection"
    mode_seeking: float = 0.0             # MSGAN regulariser weight: same labels + different z -> different images
    ema_decay: float = 0.998              # exponential moving average of G weights for sampling (0 = off)
    nz: int = 100                         # latent dimension
    ngf: int = 64                         # generator base channels
    ndf: int = 64                         # discriminator base channels

    # --- optimisation ---
    epochs: int = 50
    batch_size: int = 64
    lr: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    label_smoothing: float = 0.1          # one-sided: real target = 1 - label_smoothing
    real_label_jitter: float = 0.0        # real targets ~ U(r - j, r + j) with r = 1 - label_smoothing
    fake_label_max: float = 0.0           # fake targets ~ U(0, fake_label_max) (0 = hard zeros)
    instance_noise: float = 0.0           # std of Gaussian noise added to D inputs, annealed to 0 over training
    d_dropout: float = 0.0                # dropout in the discriminator (0 = off)
    d_spectral_norm: bool = False         # spectral normalisation on D's conv weights (replaces D's BatchNorm)
    d_minibatch_std: bool = False         # minibatch std-dev feature in D (anti mode collapse, Karras et al. 2018)
    seed: int = 42
    num_workers: int = 0                  # dataset lives in RAM, workers are unnecessary

    # --- logging / checkpointing ---
    sample_every: int = 5                 # save fixed-noise grid every N epochs
    checkpoint_every: int = 10            # save full checkpoint every N epochs
    keep_last: int = 3                    # keep only the N most recent epoch checkpoints (0 = all)
    num_fixed_samples: int = 64
    checkpoint_dir: str = str(PROJECT_ROOT / "checkpoints")
    output_dir: str = str(PROJECT_ROOT / "outputs")
    device: str = "auto"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["views"] = list(self.views)
        d["background"] = list(self.background)
        d["n_classes"] = list(self.n_classes)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        known = {f.name for f in fields(cls)}
        clean = {k: v for k, v in d.items() if k in known}
        if "views" in clean:
            clean["views"] = tuple(clean["views"])
        if "background" in clean:
            clean["background"] = tuple(clean["background"])
        if "n_classes" in clean:
            clean["n_classes"] = tuple(clean["n_classes"])
        if "conditional" not in d:  # checkpoints from before conditional training existed
            clean["conditional"] = False
        return cls(**clean)


def smoke_test_config(**overrides) -> Config:
    """Small, fast configuration that exercises the whole pipeline."""
    cfg = Config(
        epochs=2,
        batch_size=32,
        subset=256,
        sample_every=1,
        checkpoint_every=1,
        keep_last=1,
        num_fixed_samples=16,
        checkpoint_dir=str(PROJECT_ROOT / "checkpoints" / "smoke"),
        output_dir=str(PROJECT_ROOT / "outputs" / "smoke"),
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def get_device(preference: str = "auto") -> torch.device:
    """CUDA (Colab) -> MPS (Apple Silicon) -> CPU, unless a device is forced."""
    if preference and preference != "auto":
        if preference == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available on this machine.")
        if preference == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available on this machine.")
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        return f"CUDA - {props.name} ({props.total_memory / 1024**3:.1f} GB)"
    if device.type == "mps":
        return "MPS - Apple Silicon GPU (Metal Performance Shaders)"
    return "CPU"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
