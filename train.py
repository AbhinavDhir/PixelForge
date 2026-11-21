"""DCGAN adversarial training for PixelForge.

Examples
    python train.py --smoke-test                 # 2 epochs on 256 images, verifies the pipeline
    python train.py --epochs 50                  # full training (default config)
    python train.py --epochs 100 --resume auto   # continue from checkpoints/latest.pt
"""
from __future__ import annotations

import argparse
import copy
import csv
import math
import os
import shutil
import time
from pathlib import Path

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from config import Config, describe_device, get_device, seed_everything, smoke_test_config
from dataset import build_dataset, make_dataloader
from models import build_models, count_parameters, sample_latent
from utils import make_grid, tensor_to_pil

EPOCH_FIELDS = ["epoch", "loss_d", "loss_g", "d_real", "d_fake", "d_fake_after_g", "sample_diversity",
                "seconds", "iterations"]
COLLAPSE_RATIO = 0.25  # warn below 25% of real variety (collapsed probes: 9-22%; healthy black run: ~37%)
COLLAPSE_RATIO_COND = 0.10  # conditional: variety across z at ONE fixed label vs. real variety over all labels
ITER_FIELDS = ["epoch", "iteration", "loss_d", "loss_g", "d_real", "d_fake"]


# --------------------------------------------------------------------------- checkpoints
def save_checkpoint(path: str | Path, G: nn.Module, D: nn.Module, opt_g, opt_d, epoch: int,
                    cfg: Config, history: list[dict], fixed_noise: torch.Tensor, history_csv: str,
                    extra: dict | None = None) -> Path:
    """Portable checkpoint: state_dicts only (no pickled model objects), tensors moved to CPU."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "format": "pixelforge-dcgan-v1",
        "epoch": int(epoch),
        "config": cfg.to_dict(),
        "generator": {k: v.detach().cpu() for k, v in G.state_dict().items()},
        "discriminator": {k: v.detach().cpu() for k, v in D.state_dict().items()},
        "opt_g": _optimizer_state_to_cpu(opt_g.state_dict()),
        "opt_d": _optimizer_state_to_cpu(opt_d.state_dict()),
        "history": history,
        "history_csv": str(history_csv),
        "fixed_noise": fixed_noise.detach().cpu(),
        "params": {"generator": count_parameters(G), "discriminator": count_parameters(D)},
        "rng_cpu": torch.get_rng_state(),
        **(extra or {}),
    }
    tmp = path.with_suffix(".tmp")
    torch.save(ckpt, tmp)
    tmp.replace(path)  # atomic: an interrupted save never corrupts the previous file
    return path


def save_generator_only(path: str | Path, G: nn.Module, cfg: Config, epoch: int, history: list[dict],
                        ema: bool = False) -> Path:
    """Small (~14 MB) inference-only file for generate.py and the Streamlit app (EMA weights if available)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": "pixelforge-generator-v1",
        "epoch": int(epoch),
        "config": cfg.to_dict(),
        "generator": {k: v.detach().cpu() for k, v in G.state_dict().items()},
        "params": {"generator": count_parameters(G)},
        "history": history,
        "ema": ema,
    }, path)
    return path


def _optimizer_state_to_cpu(sd: dict) -> dict:
    out = {"param_groups": sd["param_groups"], "state": {}}
    for k, st in sd["state"].items():
        out["state"][k] = {n: (v.detach().cpu() if torch.is_tensor(v) else v) for n, v in st.items()}
    return out


def load_checkpoint(path: str | Path) -> dict:
    """Always load onto the CPU first so CUDA-trained files open on MPS/CPU and vice versa."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # very old torch without weights_only
        return torch.load(path, map_location="cpu")


def find_latest_checkpoint(checkpoint_dir: str | Path) -> Path | None:
    d = Path(checkpoint_dir)
    if (d / "latest.pt").exists():
        return d / "latest.pt"
    eps = sorted(d.glob("pixelforge_epoch_*.pt"))
    return eps[-1] if eps else None


def _prune(checkpoint_dir: Path, keep_last: int) -> None:
    if keep_last <= 0:
        return
    eps = sorted(checkpoint_dir.glob("pixelforge_epoch_*.pt"))
    for old in eps[:-keep_last]:
        old.unlink(missing_ok=True)


# --------------------------------------------------------------------------- csv helpers
def _rewrite_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _append_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerows(rows)


def _truncate_iter_csv(path: Path, last_epoch: int) -> None:
    """On resume, drop iteration rows logged after the checkpoint we resume from."""
    if not path.exists():
        return
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if int(r["epoch"]) <= last_epoch]
    _rewrite_csv(path, ITER_FIELDS, rows)


@torch.no_grad()
def sample_diversity(G: nn.Module, fixed_noise: torch.Tensor, labels: torch.Tensor | None = None) -> float:
    """Mean per-pixel std across the fixed-noise samples (in [0,1] units). ~0 => all z give the same image.

    For a conditional G pass ONE label combination repeated, so the measure reflects variety across z only.
    """
    was_training = G.training
    G.eval()
    x = (_g(G, fixed_noise, labels).float() + 1) / 2
    if was_training:
        G.train()
    return float(x.std(dim=0).mean())


def _g(G: nn.Module, z: torch.Tensor, y: torch.Tensor | None):
    return G(z, y) if getattr(G, "conditional", False) else G(z)


def _d(D: nn.Module, x: torch.Tensor, y: torch.Tensor | None):
    return D(x, y) if getattr(D, "conditional", False) else D(x)


def fixed_condition_grid(n: int, device) -> torch.Tensor:
    """Labels for the 8x8 monitoring grid: row = hair colour, column = view (back/left/front/right x2),
    outfit cycles per column. Makes it easy to see whether G obeys the labels."""
    from attributes import HAIR, OUTFIT
    hair_rows = [HAIR.index(h) for h in ("black", "brown", "blonde", "red", "grey", "green", "blue", "purple")]
    outfits = [OUTFIT.index(o) for o in ("purple", "grey", "yellow", "blue", "red", "green", "brown", "white")]
    y = [[c % 4, hair_rows[(k // 8) % 8], outfits[c]] for k in range(n) for c in [k % 8]]
    return torch.tensor(y, dtype=torch.long, device=device)


def real_diversity(data_uint8: torch.Tensor, n: int = 64, seed: int = 0) -> float:
    """Same measure as sample_diversity() on n random real images: the reference for collapse warnings."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(data_uint8.shape[0], generator=g)[:n]
    return float((data_uint8[idx].float() / 255).std(dim=0).mean())


@torch.no_grad()
def save_progress_grid(G: nn.Module, fixed_noise: torch.Tensor, path: Path, nrow: int = 8, scale: int = 2,
                       labels: torch.Tensor | None = None) -> Path:
    was_training = G.training
    G.eval()
    imgs = tensor_to_pil(_g(G, fixed_noise, labels))
    if was_training:
        G.train()
    path.parent.mkdir(parents=True, exist_ok=True)
    make_grid(imgs, nrow=nrow, scale=scale).save(path)
    return path


# --------------------------------------------------------------------------- training
def train(cfg: Config, resume: str | None = None, show_progress: bool = True, on_epoch_end=None) -> dict:
    """Run adversarial training. Returns a summary dict with paths and the per-epoch history.

    resume: None | "auto" (checkpoint_dir/latest.pt if present) | path to a checkpoint.
    on_epoch_end(epoch, G, fixed_noise, history): optional callback (used by the notebook).
    """
    device = get_device(cfg.device)
    seed_everything(cfg.seed)
    ckpt_dir, out_dir = Path(cfg.checkpoint_dir), Path(cfg.output_dir)
    progress_dir = out_dir / "training_progress"
    for d in (ckpt_dir, out_dir, progress_dir, out_dir / "plots"):
        d.mkdir(parents=True, exist_ok=True)
    history_csv, iter_csv = out_dir / "training_history.csv", out_dir / "training_iterations.csv"

    ds = build_dataset(cfg)
    loader = make_dataloader(ds, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
                             seed=cfg.seed, pin_memory=device.type == "cuda")
    if len(loader) == 0:
        raise ValueError(f"Dataset has {len(ds)} images, fewer than batch_size={cfg.batch_size}.")

    resume_path = find_latest_checkpoint(ckpt_dir) if resume == "auto" else (Path(resume) if resume else None)
    ck = load_checkpoint(resume_path) if resume_path is not None and resume_path.exists() else None
    if ck is not None:  # architecture must match the checkpoint, whatever the CLI says
        for k in ("nz", "ngf", "ndf", "d_dropout", "d_spectral_norm", "d_minibatch_std", "conditional", "n_classes", "d_cond"):
            if k in ck["config"]:
                setattr(cfg, k, tuple(ck["config"][k]) if k == "n_classes" else ck["config"][k])
            elif k in ("d_spectral_norm", "d_minibatch_std", "conditional"):
                setattr(cfg, k, False)  # checkpoints made before these options existed

    if cfg.conditional and cfg.views != (0, 1, 2, 3):
        print("Note: conditional model trained on a subset of views; unseen views will not be learned.")
    G, D = build_models(cfg.nz, cfg.ngf, cfg.ndf, device, d_dropout=cfg.d_dropout,
                        d_spectral_norm=cfg.d_spectral_norm, d_minibatch_std=cfg.d_minibatch_std,
                        n_classes=tuple(cfg.n_classes) if cfg.conditional else None, d_cond=cfg.d_cond)
    G_ema = None
    if cfg.ema_decay > 0:  # Polyak/EMA average of generator weights, used for all sampling
        G_ema = copy.deepcopy(G).eval()
        for p in G_ema.parameters():
            p.requires_grad_(False)
    fixed_labels = fixed_condition_grid(cfg.num_fixed_samples, device) if cfg.conditional else None
    div_labels = fixed_labels[:1].repeat(cfg.num_fixed_samples, 1) if cfg.conditional else None
    G_sample = lambda: G_ema if G_ema is not None else G  # noqa: E731
    opt_g = torch.optim.Adam(G.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))
    opt_d = torch.optim.Adam(D.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))
    criterion = nn.BCEWithLogitsLoss()
    fixed_noise = sample_latent(cfg.num_fixed_samples, cfg.nz, seed=cfg.seed, device=device)

    start_epoch, history = 0, []
    if ck is not None:
        if "discriminator" not in ck:
            raise ValueError(f"{resume_path} is a generator-only file; resume needs a full checkpoint.")
        G.load_state_dict(ck["generator"])
        D.load_state_dict(ck["discriminator"])
        if G_ema is not None:
            G_ema.load_state_dict(ck.get("generator_ema", ck["generator"]))
        opt_g.load_state_dict(ck["opt_g"])   # optimizer moments are restored, not reset
        opt_d.load_state_dict(ck["opt_d"])
        start_epoch, history = int(ck["epoch"]), list(ck.get("history", []))
        fixed_noise = ck["fixed_noise"].to(device)
        torch.set_rng_state(ck["rng_cpu"])
        _rewrite_csv(history_csv, EPOCH_FIELDS, history)
        _truncate_iter_csv(iter_csv, start_epoch)
        print(f"Resumed from {resume_path} at epoch {start_epoch}")
    elif resume and resume != "auto":
        raise FileNotFoundError(f"Checkpoint not found: {resume}")
    else:
        for p in (history_csv, iter_csv):
            p.unlink(missing_ok=True)
        save_progress_grid(G_sample(), fixed_noise, progress_dir / "epoch_000.png", labels=fixed_labels)

    pg, pd_ = count_parameters(G), count_parameters(D)
    real_div = real_diversity(ds.data, n=cfg.num_fixed_samples)
    print(f"Device: {describe_device(device)}")
    print(f"Images: {len(ds)} | batches/epoch: {len(loader)} | batch size: {cfg.batch_size}")
    print(f"Generator params: {pg:,} | Discriminator params: {pd_:,}")
    print(f"Real-data diversity reference: {real_div:.4f} (collapse warning below {COLLAPSE_RATIO * real_div:.4f})")

    collapse_ratio = COLLAPSE_RATIO if not cfg.conditional else COLLAPSE_RATIO_COND
    ema_steps = 0

    @torch.no_grad()
    def ema_update():
        nonlocal ema_steps
        ema_steps += 1
        d = min(cfg.ema_decay, (1 + ema_steps) / (10 + ema_steps))  # warm-up: short memory early on
        for pe, p in zip(G_ema.parameters(), G.parameters()):
            pe.lerp_(p, 1 - d)
        for be, b in zip(G_ema.buffers(), G.buffers()):
            be.copy_(b)

    real_target_value = 1.0 - cfg.label_smoothing  # one-sided label smoothing (real labels only)
    last_epoch = start_epoch
    for epoch in range(start_epoch + 1, cfg.epochs + 1):
        G.train()
        D.train()
        t0 = time.time()
        sums = {"loss_d": 0.0, "loss_g": 0.0, "d_real": 0.0, "d_fake": 0.0, "d_fake_after_g": 0.0}
        iter_rows = []
        # instance noise (Sonderby et al. 2016): blur the boundary between real and fake for D, annealed to 0
        noise_std = cfg.instance_noise * max(0.0, 1.0 - (epoch - 1) / max(1, cfg.epochs))
        bar = tqdm(loader, desc=f"Epoch {epoch}/{cfg.epochs}", leave=False, disable=not show_progress)
        for i, (real, yb) in enumerate(bar):
            real = real.to(device, non_blocking=True)
            y = yb.to(device) if cfg.conditional else None  # fakes use the real batch's labels
            b = real.size(0)
            real_t = torch.full((b,), real_target_value, device=device)
            if cfg.real_label_jitter > 0:  # noisy labels (as in the original TinyHero DCGAN notebook)
                real_t = real_t + (torch.rand(b, device=device) * 2 - 1) * cfg.real_label_jitter
            fake_t = torch.rand(b, device=device) * cfg.fake_label_max if cfg.fake_label_max > 0 \
                else torch.zeros(b, device=device)

            # (1) Discriminator: maximise log D(x) + log(1 - D(G(z)))
            z = torch.randn(b, cfg.nz, 1, 1, device=device)
            fake = _g(G, z, y)
            opt_d.zero_grad(set_to_none=True)
            if noise_std > 0:
                d_real_in = real + noise_std * torch.randn_like(real)
                d_fake_in = fake + noise_std * torch.randn_like(fake)
            else:
                d_real_in, d_fake_in = real, fake
            out_real = _d(D, d_real_in, y)
            out_fake = _d(D, d_fake_in.detach(), y)  # detach: no generator gradients in the D step
            loss_d = criterion(out_real, real_t) + criterion(out_fake, fake_t)
            loss_d.backward()
            opt_d.step()

            # (2) Generator: non-saturating loss, maximise log D(G(z))
            opt_g.zero_grad(set_to_none=True)
            out_fake_g = _d(D, d_fake_in, y)
            loss_g = criterion(out_fake_g, torch.ones(b, device=device))
            if cfg.mode_seeking > 0:
                # Mode-seeking regulariser (Mao et al., MSGAN, CVPR 2019): for a second z with the SAME labels,
                # maximise |G(z1,y) - G(z2,y)| / |z1 - z2|, so the generator cannot ignore z.
                z2 = torch.randn_like(z)
                fake2 = _g(G, z2, y)
                ratio = (fake - fake2).abs().mean() / (z - z2).abs().mean()
                loss_g = loss_g + cfg.mode_seeking / (ratio + 1e-5)
            loss_g.backward()
            opt_g.step()
            if G_ema is not None:
                ema_update()

            ld, lg = loss_d.item(), loss_g.item()
            if not (math.isfinite(ld) and math.isfinite(lg)):
                raise FloatingPointError(
                    f"Non-finite loss at epoch {epoch}, iteration {i} (loss_d={ld}, loss_g={lg}). "
                    "Try a lower --lr or resume from the last checkpoint.")
            dr = torch.sigmoid(out_real).mean().item()
            df = torch.sigmoid(out_fake).mean().item()
            dfg = torch.sigmoid(out_fake_g).mean().item()
            for k, v in zip(sums, (ld, lg, dr, df, dfg)):
                sums[k] += v
            iter_rows.append({"epoch": epoch, "iteration": (epoch - 1) * len(loader) + i,
                              "loss_d": round(ld, 5), "loss_g": round(lg, 5),
                              "d_real": round(dr, 5), "d_fake": round(df, 5)})
            bar.set_postfix(loss_d=f"{ld:.3f}", loss_g=f"{lg:.3f}", D_x=f"{dr:.2f}", D_Gz=f"{df:.2f}")

        n = len(loader)
        row = {k: round(v / n, 5) for k, v in sums.items()}
        row.update(epoch=epoch, sample_diversity=round(sample_diversity(G_sample(), fixed_noise, div_labels), 5),
                   seconds=round(time.time() - t0, 2), iterations=n)
        history.append(row)
        _append_csv(history_csv, EPOCH_FIELDS, [row])
        _append_csv(iter_csv, ITER_FIELDS, iter_rows)
        print(f"[{epoch:3d}/{cfg.epochs}] loss_D={row['loss_d']:.4f} loss_G={row['loss_g']:.4f} "
              f"D(x)={row['d_real']:.3f} D(G(z))={row['d_fake']:.3f} div={row['sample_diversity']:.3f} "
              f"({row['seconds']:.1f}s)", flush=True)
        if row["sample_diversity"] < collapse_ratio * real_div and epoch >= 3:
            print(f"WARNING: possible mode collapse - fixed-noise samples are nearly identical "
                  f"(diversity {row['sample_diversity']:.4f} < {collapse_ratio:.0%} of real {real_div:.4f}). "
                  "Consider --instance-noise / --d-dropout or a lower learning rate.", flush=True)

        if epoch % cfg.sample_every == 0 or epoch == cfg.epochs or epoch == 1:
            save_progress_grid(G_sample(), fixed_noise, progress_dir / f"epoch_{epoch:03d}.png", labels=fixed_labels)
        # latest.pt every epoch -> a Colab disconnect loses at most one epoch
        extra = {"fixed_labels": fixed_labels.cpu() if fixed_labels is not None else None}
        if G_ema is not None:
            extra["generator_ema"] = {k: v.detach().cpu() for k, v in G_ema.state_dict().items()}
        save_checkpoint(ckpt_dir / "latest.pt", G, D, opt_g, opt_d, epoch, cfg, history, fixed_noise,
                        str(history_csv), extra=extra)
        if epoch % cfg.checkpoint_every == 0:
            # identical content to latest.pt right now -> hard link (no extra disk), copy as fallback
            numbered = ckpt_dir / f"pixelforge_epoch_{epoch:03d}.pt"
            numbered.unlink(missing_ok=True)
            try:
                os.link(ckpt_dir / "latest.pt", numbered)
            except OSError:
                shutil.copy2(ckpt_dir / "latest.pt", numbered)
            _prune(ckpt_dir, cfg.keep_last)
        if on_epoch_end is not None:
            on_epoch_end(epoch, G, fixed_noise, history)
        last_epoch = epoch

    final = save_generator_only(ckpt_dir / "generator_final.pt", G_sample(), cfg, last_epoch, history,
                                ema=G_ema is not None)
    try:
        from evaluate import plot_losses
        plot_losses(history_csv, iter_csv, out_dir / "plots" / "losses.png")
    except ImportError:
        pass  # evaluate.py not present yet (notebook writes it in a later cell)
    except Exception as e:  # plotting must never lose a finished training run
        print(f"(loss plot skipped: {e})")
    print(f"Done. Generator weights: {final}")
    return {"generator_path": str(final), "latest_path": str(ckpt_dir / "latest.pt"), "history": history,
            "history_csv": str(history_csv), "device": str(device), "epochs_completed": last_epoch,
            "params": {"generator": pg, "discriminator": pd_}}


def parse_args(argv=None):
    d = Config()
    p = argparse.ArgumentParser(description="Train the PixelForge DCGAN on TinyHero")
    p.add_argument("--smoke-test", action="store_true", help="2 epochs on 256 images, outputs in */smoke/")
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--beta1", type=float)
    p.add_argument("--beta2", type=float)
    p.add_argument("--nz", type=int, help="latent dimension")
    p.add_argument("--ngf", type=int)
    p.add_argument("--ndf", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--subset", type=int, help="train on N random images only")
    p.add_argument("--views", type=int, nargs="+", choices=[0, 1, 2, 3],
                   help="0=back 1=left 2=front 3=right (default: all). --views 2 = front only (912 images)")
    p.add_argument("--hflip", action="store_true", help="random horizontal flip augmentation")
    p.add_argument("--unconditional", action="store_true",
                   help="plain DCGAN without attribute labels (no prompt control)")
    p.add_argument("--d-cond", choices=["projection", "concat"], help="how the discriminator sees labels")
    p.add_argument("--mode-seeking", type=float, help="MSGAN regulariser weight (0 = off)")
    p.add_argument("--ema-decay", type=float, help="EMA of generator weights for sampling (0 = off, default 0.998)")
    p.add_argument("--label-smoothing", type=float, help=f"one-sided; 0 disables (default {d.label_smoothing})")
    p.add_argument("--real-label-jitter", type=float, help="real targets ~ U(r-j, r+j)")
    p.add_argument("--fake-label-max", type=float, help="fake targets ~ U(0, m)")
    p.add_argument("--background", type=int, nargs=3, metavar=("R", "G", "B"),
                   help="colour that transparent pixels are composited onto (default 0 0 0; 255 255 255 = white)")
    p.add_argument("--instance-noise", type=float, help="std of annealed Gaussian noise on D inputs (0 = off)")
    p.add_argument("--d-dropout", type=float, help="dropout probability in the discriminator (0 = off)")
    p.add_argument("--spectral-norm", dest="d_spectral_norm", action="store_true", default=None,
                   help="spectral normalisation in D (SN-DCGAN)")
    p.add_argument("--no-spectral-norm", dest="d_spectral_norm", action="store_false",
                   help="plain DCGAN discriminator with BatchNorm")
    p.add_argument("--minibatch-std", dest="d_minibatch_std", action="store_true", default=None,
                   help="minibatch std-dev feature in D (counters mode collapse)")
    p.add_argument("--no-minibatch-std", dest="d_minibatch_std", action="store_false")
    p.add_argument("--sample-every", type=int)
    p.add_argument("--checkpoint-every", type=int)
    p.add_argument("--keep-last", type=int, help="number of epoch checkpoints to keep (0 = all)")
    p.add_argument("--data-dir")
    p.add_argument("--checkpoint-dir")
    p.add_argument("--output-dir")
    p.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--num-workers", type=int)
    p.add_argument("--resume", nargs="?", const="auto", default=None,
                   help="resume from a checkpoint path, or no value / 'auto' for <checkpoint-dir>/latest.pt")
    p.add_argument("--no-progress", action="store_true", help="disable tqdm bars")
    return p.parse_args(argv)


def config_from_args(args) -> Config:
    cfg = smoke_test_config() if args.smoke_test else Config()
    for name in ("epochs", "batch_size", "lr", "beta1", "beta2", "nz", "ngf", "ndf", "seed", "subset",
                 "ema_decay", "mode_seeking", "d_cond", "label_smoothing", "real_label_jitter", "fake_label_max", "instance_noise", "d_dropout", "d_spectral_norm", "d_minibatch_std", "sample_every", "checkpoint_every", "keep_last", "data_dir",
                 "checkpoint_dir", "output_dir", "device", "num_workers"):
        v = getattr(args, name)
        if v is not None:
            setattr(cfg, name, v)
    if args.views:
        cfg.views = tuple(args.views)
    if args.background:
        cfg.background = tuple(args.background)
    if args.hflip:
        cfg.hflip = True
    if args.unconditional:
        cfg.conditional = False
    return cfg


if __name__ == "__main__":
    a = parse_args()
    train(config_from_args(a), resume=a.resume, show_progress=not a.no_progress)
