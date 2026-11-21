"""Generate new characters (and latent interpolations) from a trained Generator.

Examples
    python generate.py --num-images 16 --seed 7
    python generate.py --num-images 9 --scale 8 --transparent
    python generate.py --interpolate --seed-a 1 --seed-b 2 --steps 10 --gif
    python generate.py --prompt "a front-facing hero with green hair and a red tunic" --num-images 8
    python generate.py --view back --hair grey --outfit grey --num-images 4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from config import PROJECT_ROOT, Config, get_device
from models import ConditionalGenerator, Generator, interpolate_latents, sample_latent
from utils import make_grid, remove_background, save_gif, tensor_to_pil, upscale

DEFAULT_CANDIDATES = ("generator_final.pt", "latest.pt")


def find_checkpoint(checkpoint_dir: str | Path = PROJECT_ROOT / "checkpoints") -> Path | None:
    d = Path(checkpoint_dir)
    for name in DEFAULT_CANDIDATES:
        if (d / name).exists():
            return d / name
    eps = sorted(d.glob("pixelforge_epoch_*.pt"))
    return eps[-1] if eps else None


def load_generator(path: str | Path, device: torch.device | str = "cpu"):
    """Load a Generator from either a full training checkpoint or a generator-only file.

    Files are loaded with map_location="cpu" and then moved to `device`, so a
    checkpoint trained on a Colab T4 (CUDA) works on a Mac (MPS) or CPU.
    Returns (generator in eval mode, Config, checkpoint metadata dict).
    """
    try:
        ck = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        ck = torch.load(path, map_location="cpu")
    if not isinstance(ck, dict) or "generator" not in ck:
        raise ValueError(f"{path} does not look like a PixelForge checkpoint (no 'generator' key).")
    cfg = Config.from_dict(ck.get("config", {}))
    G = ConditionalGenerator(cfg.nz, cfg.ngf, n_classes=cfg.n_classes) if cfg.conditional else Generator(cfg.nz, cfg.ngf)
    G.load_state_dict(ck["generator"])
    G.to(device).eval()
    meta = {k: ck.get(k) for k in ("epoch", "format", "params", "history", "ema")}
    meta["path"] = str(path)
    return G, cfg, meta


def is_conditional(G) -> bool:
    return bool(getattr(G, "conditional", False))


def make_labels(G, n: int, labels: dict | None = None, seed: int | None = None,
                device: torch.device | str = "cpu") -> torch.Tensor | None:
    """(n, 3) label tensor [view, hair, outfit]. Unspecified attributes (None) are sampled with `seed`."""
    if not is_conditional(G):
        return None
    labels = labels or {}
    g = torch.Generator().manual_seed(int(seed) + 7919 if seed is not None else 0)
    cols = []
    for i, (name, ncls) in enumerate(zip(("view", "hair", "outfit"), G.n_classes)):
        v = labels.get(name)
        cols.append(torch.full((n,), int(v)) if v is not None else torch.randint(0, ncls, (n,), generator=g))
    return torch.stack(cols, dim=1).long().to(device)


@torch.no_grad()
def generate(G: Generator, n: int, seed: int | None = None, device: torch.device | str = "cpu",
             z: torch.Tensor | None = None, labels: dict | torch.Tensor | None = None) -> list[Image.Image]:
    """Sample n latent vectors (deterministic if seed is set) and return 64x64 RGB PIL images.

    For a conditional generator, `labels` is a dict like {"view": 2, "hair": 6, "outfit": None}
    (None/missing = random) or a ready (n, 3) tensor.
    """
    if z is None:
        z = sample_latent(n, G.nz, seed=seed, device=device)
    if not is_conditional(G):
        return tensor_to_pil(G(z.to(device)))
    y = labels.to(device) if torch.is_tensor(labels) else make_labels(G, n, labels, seed, device)
    return tensor_to_pil(G(z.to(device), y))


@torch.no_grad()
def interpolate(G: Generator, seed_a: int, seed_b: int, steps: int = 10, mode: str = "linear",
                device: torch.device | str = "cpu", labels: dict | None = None) -> list[Image.Image]:
    """Decode the path between the latent vectors produced by seed_a and seed_b.

    Conditional G: attributes are held fixed along the path (from `labels`, else drawn with seed_a).
    """
    z1 = sample_latent(1, G.nz, seed=seed_a, device=device)[0]
    z2 = sample_latent(1, G.nz, seed=seed_b, device=device)[0]
    zs = interpolate_latents(z1, z2, steps, mode)
    if not is_conditional(G):
        return tensor_to_pil(G(zs))
    y = make_labels(G, 1, labels, seed_a, device).repeat(steps, 1)
    return tensor_to_pil(G(zs, y))


def save_images(images: list[Image.Image], out_dir: str | Path, prefix: str = "character",
                scale: int = 1, transparent: bool = False, background=(0, 0, 0)) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, im in enumerate(images):
        p = out / f"{prefix}_{i:03d}.png"
        upscale(im, scale).save(p)  # genuine RGB output of the GAN
        paths.append(p)
        if transparent:
            upscale(remove_background(im, background), scale).save(out / f"{prefix}_{i:03d}_transparent.png")
    return paths


def main():
    p = argparse.ArgumentParser(description="Generate PixelForge characters from a trained DCGAN")
    p.add_argument("--checkpoint", help="path to generator_final.pt / latest.pt (default: auto-detect in checkpoints/)")
    p.add_argument("--num-images", type=int, default=16, help="number of characters (4, 9 and 16 make square grids)")
    p.add_argument("--seed", type=int, default=None, help="fix for deterministic output")
    p.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "generated"))
    p.add_argument("--scale", type=int, default=4, help="nearest-neighbour upscale for the preview grid")
    p.add_argument("--sprite-scale", type=int, default=1, help="upscale for individual sprite PNGs (1 = native 64x64)")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--transparent", action="store_true",
                   help="ALSO save background-removed copies (heuristic flood fill, see README)")
    p.add_argument("--interpolate", action="store_true", help="make a latent interpolation strip instead")
    p.add_argument("--seed-a", type=int, default=0)
    p.add_argument("--seed-b", type=int, default=1)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--mode", default="linear", choices=["linear", "slerp"])
    p.add_argument("--gif", action="store_true", help="also save an animated GIF of the interpolation")
    p.add_argument("--interp-dir", default=str(PROJECT_ROOT / "outputs" / "interpolation"))
    from attributes import HAIR, OUTFIT, VIEWS
    p.add_argument("--prompt", help='description, e.g. "blonde hero in a blue outfit, facing left" (conditional model)')
    p.add_argument("--view", choices=VIEWS)
    p.add_argument("--hair", choices=HAIR)
    p.add_argument("--outfit", choices=OUTFIT)
    args = p.parse_args()

    device = get_device(args.device)
    ckpt = Path(args.checkpoint) if args.checkpoint else find_checkpoint()
    if ckpt is None or not ckpt.exists():
        raise SystemExit("No checkpoint found. Train first (python train.py --epochs 50, or the Colab notebook) "
                         "and put generator_final.pt in checkpoints/, or pass --checkpoint PATH.")
    G, cfg, meta = load_generator(ckpt, device)
    print(f"Loaded {ckpt} (epoch {meta['epoch']}) on {device}")
    labels = None
    if args.prompt or args.view or args.hair or args.outfit:
        from attributes import HAIR, OUTFIT, VIEWS, parse_prompt
        if not is_conditional(G):
            raise SystemExit("This checkpoint is unconditional: retrain with `python train.py` (conditional by default).")
        labels = parse_prompt(args.prompt)[0] if args.prompt else {"view": None, "hair": None, "outfit": None}
        for k, vocab in (("view", VIEWS), ("hair", HAIR), ("outfit", OUTFIT)):
            if getattr(args, k):
                labels[k] = vocab.index(getattr(args, k))
        print("Attributes:", ", ".join(f"{k}={v[labels[k]] if labels[k] is not None else 'random'}"
                                       for k, v in (("view", VIEWS), ("hair", HAIR), ("outfit", OUTFIT))))

    if args.interpolate:
        out = Path(args.interp_dir)
        frames = interpolate(G, args.seed_a, args.seed_b, args.steps, args.mode, device, labels=labels)
        out.mkdir(parents=True, exist_ok=True)
        tag = f"seed{args.seed_a}_to_seed{args.seed_b}_{args.mode}"
        strip = make_grid(frames, nrow=len(frames), scale=args.scale)
        strip.save(out / f"interpolation_{tag}.png")
        print(f"Saved {out / f'interpolation_{tag}.png'}")
        if args.gif:
            g = save_gif([upscale(f, args.scale) for f in frames], out / f"interpolation_{tag}.gif")
            print(f"Saved {g}")
        return

    images = generate(G, args.num_images, args.seed, device, labels=labels)
    tag = f"seed{args.seed}" if args.seed is not None else "random"
    if labels is not None:
        tag = "prompt_" + "_".join(str(labels[k]) if labels[k] is not None else "x" for k in ("view", "hair", "outfit")) \
              + f"_{tag}"
    paths = save_images(images, Path(args.output_dir) / tag, scale=args.sprite_scale,
                        transparent=args.transparent, background=cfg.background)
    grid = make_grid(images, scale=args.scale)
    grid_path = Path(args.output_dir) / f"grid_{args.num_images}_{tag}.png"
    grid.save(grid_path)
    print(f"Saved {len(paths)} sprites to {Path(args.output_dir) / tag} and grid {grid_path}")


if __name__ == "__main__":
    main()
