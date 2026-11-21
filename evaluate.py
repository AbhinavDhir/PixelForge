"""PixelForge evaluation: five lightweight, honest experiments.

  1. Training losses (G vs D)                    -> plots/losses.png
  2. Fixed-noise quality progression over epochs -> plots/progression.png
  3. Real vs generated side-by-side              -> plots/real_vs_generated.png
  4. Latent interpolation strips                 -> plots/interpolation_experiment.png
  5. Diversity + memorisation (nearest neighbours in pixel space)
                                                 -> plots/nearest_neighbors.png, plots/diversity.png
All numbers are computed from the actual checkpoint and dataset and written to
outputs/evaluation_metrics.json. Nothing is hard-coded.

    python evaluate.py                       # uses checkpoints/generator_final.pt
    python evaluate.py --checkpoint checkpoints/latest.pt --num-samples 1000
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.figure import Figure
from PIL import Image

from config import PROJECT_ROOT, Config, get_device

# Chart styling (validated categorical slots for a dark surface).
SURFACE, INK, INK_MUTED, GRID = "#14111f", "#f0efec", "#a9a6b8", "#2c2840"
C_GEN, C_DISC = "#3987e5", "#d95926"


def _style(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_MUTED, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_MUTED, fontsize=9)


def _legend(ax):
    leg = ax.legend(frameon=False, fontsize=9, loc="upper right")
    for t in leg.get_texts():
        t.set_color(INK)


def _fig(w, h, nrows=1, ncols=1):
    fig = Figure(figsize=(w, h), dpi=120, facecolor=SURFACE)
    axes = fig.subplots(nrows, ncols, squeeze=False)
    return fig, axes


def _read_csv(path):
    with open(path, newline="") as f:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]


def _smooth(y, k):
    if len(y) < k or k <= 1:
        return np.asarray(y)
    return np.convolve(y, np.ones(k) / k, mode="valid")


# --------------------------------------------------------------------------- Experiment 1
def plot_losses(history_csv, iter_csv=None, out_path="outputs/plots/losses.png") -> Path:
    """Generator and discriminator losses: one panel per loss (one axis each), plus D outputs."""
    hist = _read_csv(history_csv)
    it = _read_csv(iter_csv) if iter_csv and Path(iter_csv).exists() else []
    fig, ax = _fig(12, 3.6, 1, 3)
    ep = [r["epoch"] for r in hist]
    for a, key, color, name in ((ax[0, 0], "loss_g", C_GEN, "Generator loss"),
                                (ax[0, 1], "loss_d", C_DISC, "Discriminator loss")):
        if it:
            x = np.array([r["iteration"] for r in it])
            y = np.array([r[key] for r in it])
            k = max(1, len(y) // 60)
            if k > 1:
                a.plot(x, y, color=color, alpha=0.22, linewidth=1, label="per iteration")
                a.plot(x[k - 1:], _smooth(y, k), color=color, linewidth=2, label=f"moving avg ({k} iters)")
            else:
                a.plot(x, y, color=color, linewidth=2, label="per iteration")
            _style(a, name, "iteration", "BCE loss")
        else:
            a.plot(ep, [r[key] for r in hist], color=color, linewidth=2, marker="o", markersize=4, label="epoch mean")
            _style(a, name, "epoch", "BCE loss")
        _legend(a)
    a = ax[0, 2]
    a.plot(ep, [r["d_real"] for r in hist], color=C_GEN, linewidth=2, marker="o", markersize=4, label="D(x) real")
    a.plot(ep, [r["d_fake"] for r in hist], color=C_DISC, linewidth=2, marker="o", markersize=4, label="D(G(z)) fake")
    a.set_ylim(0, 1)
    _style(a, "Discriminator confidence", "epoch", "mean sigmoid output")
    _legend(a)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=SURFACE)
    return out_path


# --------------------------------------------------------------------------- Experiment 2
def progression_montage(progress_dir, out_path, n_sprites: int = 8, max_rows: int = 12) -> Path | None:
    """First row of each fixed-noise grid, stacked by epoch: same z, increasing training."""
    files = sorted(Path(progress_dir).glob("epoch_*.png"))
    if not files:
        return None
    if len(files) > max_rows:  # keep first, last, and evenly spaced in between
        idx = np.unique(np.linspace(0, len(files) - 1, max_rows).round().astype(int))
        files = [files[i] for i in idx]
    rows = []
    for f in files:
        g = Image.open(f).convert("RGB")
        # grids are made with padding=2, 64px tiles, upscale s (inferred from width, 8 per row)
        s = g.width // (8 * 64 + 9 * 2)
        tile, pad = 64 * s, 2 * s
        rows.append((int(re.findall(r"\d+", f.stem)[0]),
                     [g.crop((pad + c * (tile + pad), pad, pad + c * (tile + pad) + tile, pad + tile)) for c in range(n_sprites)]))
    fig, ax = _fig(n_sprites * 0.9 + 1, len(rows) * 0.95 + 0.6, len(rows), n_sprites)
    for r, (epoch, tiles) in enumerate(rows):
        for c, t in enumerate(tiles):
            a = ax[r, c]
            a.imshow(t, interpolation="nearest")
            a.set_xticks([]), a.set_yticks([])
            for sp in a.spines.values():
                sp.set_visible(False)
            if c == 0:
                a.set_ylabel(f"epoch {epoch}", color=INK, fontsize=8, rotation=0, ha="right", va="center")
    fig.suptitle("Same latent vectors across training (fixed noise)", color=INK, fontsize=11, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))  # leave room for the suptitle
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=SURFACE)
    return out_path


# --------------------------------------------------------------------------- helpers
def _imshow_grid(ax, images, title):
    from utils import make_grid
    ax.imshow(make_grid(images, nrow=int(np.ceil(np.sqrt(len(images))))), interpolation="nearest")
    ax.set_xticks([]), ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title, color=INK, fontsize=12, pad=8)


def _to_pil(x01: torch.Tensor) -> list[Image.Image]:
    arr = (x01.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype(np.uint8)
    return [Image.fromarray(a) for a in arr]


@torch.no_grad()
def generate_batch01(G, n, seed, device, batch=256, labels: torch.Tensor | None = None) -> torch.Tensor:
    """n generated images in [0,1]. Conditional G: `labels` (n,3) or, if None, random labels with `seed`."""
    from generate import is_conditional, make_labels
    from models import sample_latent
    z = sample_latent(n, G.nz, seed=seed, device="cpu")
    if is_conditional(G) and labels is None:
        labels = make_labels(G, n, None, seed, "cpu")
    outs = []
    for i in range(0, n, batch):
        zi = z[i:i + batch].to(device)
        x = G(zi, labels[i:i + batch].to(device)) if is_conditional(G) else G(zi)
        outs.append(((x.float().cpu() + 1) / 2).clamp(0, 1))
    return torch.cat(outs)


@torch.no_grad()
def controllability(G, device, background, per_class: int = 48, seed: int = 0) -> dict:
    """Experiment 6: does G obey the requested hair / outfit colour?

    For every class c of an attribute, generate `per_class` sprites asking for c (other attributes random),
    label them with the same automatic labeller used on the training data, and count matches.
    Baseline: the same labels compared against a *shuffled* copy of the requests (= a model that ignores them).
    """
    from attributes import HAIR, OUTFIT, label_image
    from generate import make_labels
    res = {}
    for col, name, vocab in ((1, "hair", HAIR), (2, "outfit", OUTFIT)):
        req, got = [], []
        for c in range(len(vocab)):
            y = make_labels(G, per_class, {name: c}, seed + 31 * c, "cpu")
            imgs = _to_pil(generate_batch01(G, per_class, seed + 31 * c + 1, device, labels=y))
            req += [c] * per_class
            got += [label_image(im, background)[name] for im in imgs]
        req, got = np.array(req), np.array(got)
        rng = np.random.default_rng(seed)
        per = {vocab[c]: float((got[req == c] == c).mean()) for c in range(len(vocab))}
        res[name] = {"accuracy": float((req == got).mean()),
                     "shuffled_baseline": float((rng.permutation(req) == got).mean()),
                     "chance_uniform": 1 / len(vocab), "per_class": per}
    return res


def _rmse_matrix(a: torch.Tensor, b: torch.Tensor, chunk: int = 512) -> torch.Tensor:
    """Per-pixel RMSE between every image in a and every image in b (values in [0,1])."""
    a2, b2 = a.flatten(1), b.flatten(1)
    d = a2.shape[1]
    rows = [torch.cdist(a2[i:i + chunk], b2) for i in range(0, a2.shape[0], chunk)]
    return torch.cat(rows) / np.sqrt(d)


def _pairwise_stats(x: torch.Tensor) -> dict:
    m = _rmse_matrix(x, x)
    iu = torch.triu_indices(len(x), len(x), 1)
    v = m[iu[0], iu[1]]
    return {"mean": float(v.mean()), "std": float(v.std()), "min": float(v.min()),
            "near_duplicate_pairs(<0.02)": int((v < 0.02).sum())}


def _color_hist(x: torch.Tensor, bins: int = 16) -> np.ndarray:
    h = [torch.histc(x[:, c].flatten(), bins=bins, min=0, max=1) for c in range(3)]
    h = torch.stack(h)
    return (h / h.sum(dim=1, keepdim=True)).numpy()


# --------------------------------------------------------------------------- main evaluation
def evaluate(checkpoint, cfg: Config | None = None, num_samples: int = 512, seed: int = 123,
             output_dir=None, device="auto", data_dir=None) -> dict:
    from dataset import TinyHeroDataset, download_tinyhero
    from generate import interpolate, load_generator
    from utils import make_grid

    device = get_device(device) if isinstance(device, str) else device
    G, ck_cfg, meta = load_generator(checkpoint, device)
    cfg = cfg or ck_cfg
    out = Path(output_dir or cfg.output_dir)
    plots = out / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    root = download_tinyhero(data_dir or cfg.data_dir)
    ds = TinyHeroDataset(root, views=cfg.views, background=cfg.background,
                         with_attributes=bool(cfg.conditional))  # full set, no subset
    real_all = ds.as_float01()
    metrics = {"checkpoint": str(checkpoint), "checkpoint_epoch": meta["epoch"], "device": str(device),
               "num_generated": num_samples, "num_training_images": len(ds), "seed": seed}

    # 1. losses
    hist_csv, iter_csv = out / "training_history.csv", out / "training_iterations.csv"
    if hist_csv.exists():
        plot_losses(hist_csv, iter_csv, plots / "losses.png")
        hist = _read_csv(hist_csv)
        metrics["final_epoch_losses"] = {k: hist[-1][k] for k in ("loss_d", "loss_g", "d_real", "d_fake")}
    elif meta.get("history"):
        tmp = out / "training_history_from_ckpt.csv"
        from train import EPOCH_FIELDS, _rewrite_csv
        _rewrite_csv(tmp, EPOCH_FIELDS, meta["history"])
        plot_losses(tmp, None, plots / "losses.png")

    # 2. progression
    progression_montage(out / "training_progress", plots / "progression.png")

    # 3. real vs generated
    rng0 = np.random.default_rng(seed + 1)
    gen_labels = ds.attrs[rng0.choice(len(ds), num_samples)] if cfg.conditional else None  # data label mix
    gen = generate_batch01(G, num_samples, seed, device, labels=gen_labels)
    rng = np.random.default_rng(seed)
    real_idx = rng.choice(len(ds), size=min(num_samples, len(ds)), replace=False)
    real = real_all[real_idx]
    fig, ax = _fig(12, 6.4, 1, 2)
    _imshow_grid(ax[0, 0], _to_pil(real[:36]), "REAL - TinyHero training images")
    _imshow_grid(ax[0, 1], _to_pil(gen[:36]), f"GENERATED - DCGAN (epoch {meta['epoch']})")
    fig.tight_layout()
    fig.savefig(plots / "real_vs_generated.png", facecolor=SURFACE)

    # 4. interpolation (3 seed pairs, linear vs spherical)
    pairs = [(seed, seed + 1), (seed + 2, seed + 3), (seed + 4, seed + 5)]
    fig, ax = _fig(12, 1.0 * len(pairs) * 2 + 0.6, len(pairs) * 2, 1)
    for i, (a, b) in enumerate(pairs):
        for j, mode in enumerate(("linear", "slerp")):
            frames = interpolate(G, a, b, 10, mode, device)
            axx = ax[i * 2 + j, 0]
            axx.imshow(make_grid(frames, nrow=len(frames)), interpolation="nearest")
            axx.set_xticks([]), axx.set_yticks([])
            for sp in axx.spines.values():
                sp.set_visible(False)
            axx.set_ylabel(f"z{a}->z{b}\n{mode}", color=INK, fontsize=8, rotation=0, ha="right", va="center")
    fig.suptitle("Latent interpolation: 10 decoded points between two seeds", color=INK, fontsize=11, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))  # leave room for the suptitle
    fig.savefig(plots / "interpolation_experiment.png", facecolor=SURFACE)
    # smoothness: mean RMSE between consecutive frames vs endpoints distance
    fr = torch.stack([torch.from_numpy(np.array(f)).permute(2, 0, 1).float() / 255
                                          for f in interpolate(G, pairs[0][0], pairs[0][1], 10, "linear", device)])
    step = _rmse_matrix(fr[:-1], fr[1:]).diagonal()
    metrics["interpolation"] = {"pair": list(pairs[0]),
                                "mean_consecutive_frame_rmse": float(step.mean()),
                                "max_consecutive_frame_rmse": float(step.max()),
                                "endpoint_rmse": float(_rmse_matrix(fr[:1], fr[-1:])[0, 0])}

    # 5. diversity & memorisation
    metrics["pairwise_rmse_generated"] = _pairwise_stats(gen)
    metrics["pairwise_rmse_real"] = _pairwise_stats(real)
    metrics["diversity_ratio_generated_over_real"] = (
        metrics["pairwise_rmse_generated"]["mean"] / metrics["pairwise_rmse_real"]["mean"])
    pooled_g, pooled_r = F.avg_pool2d(gen, 4), F.avg_pool2d(real, 4)  # 16x16 coarse structure
    metrics["pairwise_rmse_16x16_generated"] = _pairwise_stats(pooled_g)["mean"]
    metrics["pairwise_rmse_16x16_real"] = _pairwise_stats(pooled_r)["mean"]

    d_gen = _rmse_matrix(gen, real_all)                       # generated -> nearest training image
    nn_gen, nn_idx = d_gen.min(dim=1)
    d_real = _rmse_matrix(real, real_all)                     # real -> nearest OTHER training image
    d_real[torch.arange(len(real_idx)), torch.as_tensor(real_idx)] = float("inf")
    nn_real = d_real.min(dim=1).values
    metrics["nearest_neighbor_rmse_generated_to_train"] = {
        "mean": float(nn_gen.mean()), "median": float(nn_gen.median()), "min": float(nn_gen.min())}
    metrics["nearest_neighbor_rmse_real_to_other_train"] = {
        "mean": float(nn_real.mean()), "median": float(nn_real.median()), "min": float(nn_real.min())}
    metrics["fraction_generated_closer_than_real_median_nn"] = float((nn_gen < nn_real.median()).float().mean())
    metrics["possible_copies(nn_rmse<0.02)"] = int((nn_gen < 0.02).sum())
    hg, hr = _color_hist(gen), _color_hist(real)
    metrics["color_histogram_L1_distance"] = float(np.abs(hg - hr).sum() / 3)

    # nearest-neighbour figure: generated | 3 closest training images
    k, show = 3, 8
    top = d_gen[:show].topk(k, largest=False)
    fig, ax = _fig(2.2 * (k + 1), 2.0 * show + 0.6, show, k + 1)
    for r in range(show):
        tiles = [(gen[r], "generated", None)] + [
            (real_all[top.indices[r, j]], f"NN{j + 1} rmse={top.values[r, j]:.3f}", None) for j in range(k)]
        for c, (img, label, _) in enumerate(tiles):
            a = ax[r, c]
            a.imshow(img.permute(1, 2, 0).numpy(), interpolation="nearest")
            a.set_xticks([]), a.set_yticks([])
            for sp in a.spines.values():
                sp.set_edgecolor(C_GEN if c == 0 else GRID)
                sp.set_linewidth(2 if c == 0 else 1)
            a.set_title(label, color=INK if c == 0 else INK_MUTED, fontsize=8)
    fig.suptitle("Generated sample vs its 3 nearest TinyHero training images (pixel RMSE)",
                 color=INK, fontsize=11, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))  # leave room for the suptitle
    fig.savefig(plots / "nearest_neighbors.png", facecolor=SURFACE)

    # distribution figure: two panels, one axis each
    fig, ax = _fig(12, 3.6, 1, 2)
    bins = np.linspace(0, float(max(nn_gen.max(), nn_real.max())) * 1.05, 40)
    a = ax[0, 0]
    a.hist(nn_real.numpy(), bins=bins, color=C_DISC, alpha=0.65, label="real -> nearest other real")
    a.hist(nn_gen.numpy(), bins=bins, color=C_GEN, alpha=0.65, label="generated -> nearest real")
    _style(a, "Nearest-neighbour distance to the training set", "pixel RMSE", "count")
    _legend(a)
    a = ax[0, 1]
    iu = torch.triu_indices(min(len(gen), 300), min(len(gen), 300), 1)
    pg = _rmse_matrix(gen[:300], gen[:300])[iu[0], iu[1]].numpy()
    pr = _rmse_matrix(real[:300], real[:300])[iu[0], iu[1]].numpy()
    bins2 = np.linspace(0, max(pg.max(), pr.max()) * 1.05, 40)
    a.hist(pr, bins=bins2, color=C_DISC, alpha=0.65, label="real pairs")
    a.hist(pg, bins=bins2, color=C_GEN, alpha=0.65, label="generated pairs")
    _style(a, "Pairwise distance within each set (diversity)", "pixel RMSE", "count")
    _legend(a)
    fig.tight_layout()
    fig.savefig(plots / "diversity.png", facecolor=SURFACE)

    # 6. controllability (conditional model only)
    if cfg.conditional:
        from attributes import HAIR, OUTFIT, VIEWS
        ctrl = controllability(G, device, cfg.background)
        metrics["controllability"] = ctrl
        # labeller self-consistency on REAL images (upper bound for what this metric can show)
        from attributes import label_image as _li
        sub_idx = np.random.default_rng(0).choice(len(ds), 400, replace=False)
        real_imgs = _to_pil(real_all[sub_idx])
        for col, name in ((1, "hair"), (2, "outfit")):
            ctrl[name]["labeller_agreement_on_real"] = float(np.mean(
                [_li(im, cfg.background)[name] == int(ds.attrs[i][col]) for im, i in zip(real_imgs, sub_idx)]))
        fig, ax = _fig(12, 3.8, 1, 2)
        for a, name, vocab in ((ax[0, 0], "hair", HAIR), (ax[0, 1], "outfit", OUTFIT)):
            per = ctrl[name]["per_class"]
            xs = np.arange(len(vocab))
            a.bar(xs, [per[v] for v in vocab], color=C_GEN, width=0.6, label="requested colour obtained")
            a.axhline(ctrl[name]["shuffled_baseline"], color=C_DISC, linewidth=2, label="baseline (requests ignored)")
            a.set_xticks(xs, vocab, rotation=30, color=INK_MUTED, fontsize=8)
            a.set_ylim(0, 1)
            _style(a, f"{name.title()} control: {ctrl[name]['accuracy']:.0%} overall", None, "fraction of samples")
            _legend(a)
        fig.tight_layout()
        fig.savefig(plots / "controllability.png", facecolor=SURFACE)
        # visual grid: rows = hair colours, columns = views x2
        from generate import make_labels
        rows = []
        for h in range(len(HAIR)):
            y = make_labels(G, 8, {"hair": h, "outfit": None}, 500 + h, "cpu")
            y[:, 0] = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
            rows += _to_pil(generate_batch01(G, 8, 900 + h, device, labels=y))
        make_grid(rows, nrow=8, scale=3).save(plots / "prompt_control_grid.png")

    metrics["interpretation"] = interpret(metrics)
    metrics["notes"] = [
        "Pixel RMSE is a crude proxy: it ignores perceptual similarity and small shifts.",
        "Diversity ratio near 1 means generated samples are as spread out as real ones; much lower suggests mode collapse.",
        "Generated->train NN distances far below real->train NN distances would suggest memorisation.",
        "FID was not computed: it needs an ImageNet Inception network and thousands of samples, and is "
        "unreliable for 64x64 pixel art with ~3.6k training images.",
    ]
    (out / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def interpret(m: dict) -> list[str]:
    """Plain-language reading of the computed numbers (rules are fixed, numbers are measured)."""
    out = []
    r = m["diversity_ratio_generated_over_real"]
    if r < 0.5:
        out.append(f"Diversity ratio {r:.2f}: generated images are much less varied than real ones - likely mode collapse.")
    elif r < 0.85:
        out.append(f"Diversity ratio {r:.2f}: somewhat less varied than the real data (partial mode dropping is possible).")
    else:
        out.append(f"Diversity ratio {r:.2f}: generated samples are about as varied as real samples in pixel space.")
    g = m["nearest_neighbor_rmse_generated_to_train"]["median"]
    rr = m["nearest_neighbor_rmse_real_to_other_train"]["median"]
    copies = m["possible_copies(nn_rmse<0.02)"]
    for name, c in m.get("controllability", {}).items():
        out.append(f"{name.title()} control: requested colour obtained in {c['accuracy']:.0%} of samples vs "
                   f"{c['shuffled_baseline']:.0%} if requests were ignored (labeller agrees with itself on "
                   f"{c.get('labeller_agreement_on_real', float('nan')):.0%} of real images).")
    if copies > 0:
        out.append(f"{copies} generated samples are within RMSE 0.02 of a training image - inspect them for copying.")
    if g < 0.8 * rr:
        out.append(f"Median NN distance generated->train ({g:.3f}) is clearly below real->train ({rr:.3f}): "
                   "samples sit unusually close to training images (possible memorisation).")
    else:
        out.append(f"Median NN distance generated->train ({g:.3f}) vs real->other-train ({rr:.3f}): "
                   "no sign of copying; samples are new combinations rather than near-duplicates.")
    return out


def main():
    p = argparse.ArgumentParser(description="Evaluate a trained PixelForge generator")
    p.add_argument("--checkpoint", help="default: checkpoints/generator_final.pt (or latest.pt)")
    p.add_argument("--num-samples", type=int, default=512)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs"))
    p.add_argument("--data-dir", default=str(PROJECT_ROOT / "data"))
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    args = p.parse_args()
    from generate import find_checkpoint
    ckpt = Path(args.checkpoint) if args.checkpoint else find_checkpoint()
    if ckpt is None or not ckpt.exists():
        raise SystemExit("No checkpoint found. Train first or pass --checkpoint PATH.")
    m = evaluate(ckpt, num_samples=args.num_samples, seed=args.seed, output_dir=args.output_dir,
                 device=args.device, data_dir=args.data_dir)
    print(json.dumps(m, indent=2))
    print(f"Plots in {Path(args.output_dir) / 'plots'}; metrics in {Path(args.output_dir) / 'evaluation_metrics.json'}")


if __name__ == "__main__":
    main()
