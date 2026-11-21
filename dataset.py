"""TinyHero dataset: download, validation, preprocessing and PyTorch Dataset.

Verified structure of data.zip (AgaMiko/pixel_character_generator):
    data/0/*.png  back view   (912 files)
    data/1/*.png  left view   (912 files)
    data/2/*.png  front view  (912 files)
    data/3/*.png  right view  (912 files)
The same file name in each folder is the same character. Every file is a
64x64 RGBA PNG with binary alpha (~78% of pixels fully transparent). The RGB
values under transparent pixels are black. We alpha-composite explicitly onto a
configurable solid colour (Config.background). Black is the default: with a white
background the DCGAN mode-collapsed in every configuration we tried (see README,
"Experiment log"), while black trains stably - matching the original TinyHero notebooks.

Run `python dataset.py` to download, validate, and save statistics + a sample grid.
"""
from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from config import DATASET_URL, IMAGE_SIZE, VIEW_NAMES, Config

EXPECTED_PER_VIEW = 912


def dataset_root(data_dir: str | Path) -> Path:
    return Path(data_dir) / "tinyhero"


def _safe_extract(zf: zipfile.ZipFile, dest: Path) -> None:
    dest = dest.resolve()
    for member in zf.infolist():
        target = (dest / member.filename).resolve()
        if not str(target).startswith(str(dest)):
            raise RuntimeError(f"Unsafe path in archive: {member.filename}")
    zf.extractall(dest)


def download_tinyhero(data_dir: str | Path, url: str = DATASET_URL, force: bool = False) -> Path:
    """Download and extract TinyHero into <data_dir>/tinyhero/{0,1,2,3}/. Idempotent."""
    data_dir = Path(data_dir)
    root = dataset_root(data_dir)
    if root.exists() and not force and all((root / str(v)).is_dir() for v in VIEW_NAMES):
        return root
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "data.zip"
    if not zip_path.exists() or force:
        print(f"Downloading TinyHero from {url} ...")
        tmp = zip_path.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(zip_path)
    if not zipfile.is_zipfile(zip_path):
        head = zip_path.read_bytes()[:200]
        raise RuntimeError(
            f"{zip_path} is not a valid zip archive (first bytes: {head[:60]!r}). "
            "If it is a Git LFS pointer or an HTML page, delete it and re-download, "
            "or clone the repository with `git clone " + "https://github.com/AgaMiko/pixel_character_generator`."
        )
    staging = data_dir / "_extract"
    if staging.exists():
        shutil.rmtree(staging)
    with zipfile.ZipFile(zip_path) as zf:
        _safe_extract(zf, staging)
    src = staging / "data" if (staging / "data").is_dir() else staging
    if root.exists():
        shutil.rmtree(root)
    shutil.move(str(src), str(root))
    shutil.rmtree(staging, ignore_errors=True)
    print(f"Extracted to {root}")
    return root


def list_images(root: str | Path, views=(0, 1, 2, 3)) -> list[tuple[Path, int]]:
    root = Path(root)
    items = []
    for v in views:
        folder = root / str(v)
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing view folder {folder}. Run `python dataset.py` to download TinyHero.")
        # numeric sort so file 10.png follows 9.png
        files = sorted(folder.glob("*.png"), key=lambda p: (int(p.stem) if p.stem.isdigit() else 10**9, p.stem))
        items.extend((f, v) for f in files)
    return items


def load_rgb(path: str | Path, background=Config.background) -> Image.Image:
    """Open a sprite and alpha-composite it onto a solid background -> RGB."""
    im = Image.open(path)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (*background, 255))
        im = Image.alpha_composite(bg, im)
    im = im.convert("RGB")
    if im.size != (IMAGE_SIZE, IMAGE_SIZE):
        im = im.resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)
    return im


def to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL RGB uint8 -> float tensor (3,H,W) normalised to [-1, 1]."""
    arr = torch.from_numpy(np.asarray(img, dtype=np.uint8).copy()).permute(2, 0, 1).float()
    return arr / 127.5 - 1.0


class TinyHeroDataset(Dataset):
    """All sprites are decoded once and kept in RAM as uint8 (3648*64*64*3 = 45 MB)."""

    def __init__(self, root: str | Path, views=(0, 1, 2, 3), background=Config.background,
                 subset: int | None = None, hflip: bool = False, seed: int = 42, with_attributes: bool = False):
        self.root = Path(root)
        items = list_images(self.root, views)
        if not items:
            raise RuntimeError(f"No PNG files found under {self.root}")
        if subset is not None and subset < len(items):
            rng = np.random.default_rng(seed)
            idx = np.sort(rng.choice(len(items), size=subset, replace=False))
            items = [items[i] for i in idx]
        self.paths = [p for p, _ in items]
        self.labels = torch.tensor([v for _, v in items], dtype=torch.long)
        self.hflip = hflip
        self.background = tuple(background)
        self.with_attributes = with_attributes
        if with_attributes:  # [view, hair, outfit] per image, see attributes.py
            from attributes import dataset_labels
            self.attrs = torch.from_numpy(dataset_labels(self.root, items))
        imgs = [np.asarray(load_rgb(p, self.background), dtype=np.uint8) for p in self.paths]
        self.data = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).contiguous()  # (N,3,64,64) uint8

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, i: int):
        x = self.data[i].float() / 127.5 - 1.0
        flipped = self.hflip and bool(torch.rand(()) < 0.5)
        if flipped:
            x = torch.flip(x, dims=[2])
        if not self.with_attributes:
            return x, self.labels[i]
        y = self.attrs[i].clone()
        if flipped and y[0] in (1, 3):  # mirroring swaps left <-> right
            y[0] = 4 - y[0]
        return x, y

    def as_float01(self) -> torch.Tensor:
        """Whole dataset as float in [0,1] (used for nearest-neighbour search)."""
        return self.data.float() / 255.0


def build_dataset(cfg: Config, download: bool = True) -> TinyHeroDataset:
    root = download_tinyhero(cfg.data_dir) if download else dataset_root(cfg.data_dir)
    return TinyHeroDataset(root, cfg.views, cfg.background, cfg.subset, cfg.hflip, cfg.seed,
                           with_attributes=cfg.conditional)


def make_dataloader(ds: Dataset, batch_size: int, shuffle: bool = True, num_workers: int = 0,
                    seed: int = 42, pin_memory: bool = False, drop_last: bool = True) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      generator=g, pin_memory=pin_memory, drop_last=drop_last)


def validate_dataset(root: str | Path) -> dict:
    """Inspect every file: count per view, sizes, modes, alpha usage, pairing across views."""
    root = Path(root)
    stats = {"root": str(root), "views": {}, "sizes": Counter(), "modes": Counter(), "formats": Counter()}
    names_per_view = {}
    transparent = []
    for v, name in VIEW_NAMES.items():
        files = list((root / str(v)).glob("*"))
        pngs = [f for f in files if f.suffix.lower() == ".png"]
        names_per_view[v] = {f.stem for f in pngs}
        stats["views"][str(v)] = {"name": name, "count": len(pngs), "non_png": len(files) - len(pngs)}
        for f in pngs:
            with Image.open(f) as im:
                stats["sizes"][f"{im.width}x{im.height}"] += 1
                stats["modes"][im.mode] += 1
                stats["formats"][im.format] += 1
                if im.mode == "RGBA" and len(transparent) < 400:
                    transparent.append(float((np.asarray(im)[..., 3] == 0).mean()))
    common = set.intersection(*names_per_view.values()) if names_per_view else set()
    stats["total_images"] = sum(s["count"] for s in stats["views"].values())
    stats["characters_in_all_views"] = len(common)
    stats["mean_transparent_fraction"] = round(float(np.mean(transparent)), 4) if transparent else None
    stats["sizes"] = dict(stats["sizes"])
    stats["modes"] = dict(stats["modes"])
    stats["formats"] = dict(stats["formats"])
    stats["matches_documentation"] = (
        stats["total_images"] == 4 * EXPECTED_PER_VIEW and stats["sizes"] == {"64x64": 4 * EXPECTED_PER_VIEW}
    )
    return stats


def pixel_statistics(ds: TinyHeroDataset) -> dict:
    x = ds.data.float() / 255.0
    return {
        "num_images": len(ds),
        "tensor_shape": list(ds.data.shape),
        "channel_mean_0_1": [round(float(m), 4) for m in x.mean(dim=(0, 2, 3))],
        "channel_std_0_1": [round(float(s), 4) for s in x.std(dim=(0, 2, 3))],
        "normalized_min": round(float(ds[0][0].min()), 4),
        "normalized_max": round(float(ds[0][0].max()), 4),
        "per_view_counts": {VIEW_NAMES[int(k)]: int(v) for k, v in zip(*torch.unique(ds.labels, return_counts=True))},
        "background_rgb": list(ds.background),
    }


def save_sample_grid(ds: TinyHeroDataset, path: str | Path, n: int = 25, seed: int = 0, scale: int = 3) -> Path:
    from utils import make_grid
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    imgs = [Image.fromarray(ds.data[i].permute(1, 2, 0).numpy()) for i in idx]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    make_grid(imgs, nrow=int(np.ceil(np.sqrt(len(imgs)))), scale=scale).save(path)
    return path


def save_view_grid(ds_root: str | Path, path: str | Path, n_chars: int = 6, background=Config.background, scale: int = 3) -> Path:
    """Rows = characters, columns = back/left/front/right."""
    from utils import make_grid
    ds_root = Path(ds_root)
    names = sorted((p.stem for p in (ds_root / "2").glob("*.png")), key=lambda s: int(s) if s.isdigit() else 0)
    rng = np.random.default_rng(1)
    chosen = rng.choice(names, size=min(n_chars, len(names)), replace=False)
    imgs = [load_rgb(ds_root / str(v) / f"{c}.png", background) for c in chosen for v in VIEW_NAMES]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    make_grid(imgs, nrow=4, scale=scale).save(path)
    return path


def main():
    p = argparse.ArgumentParser(description="Download, validate and visualise TinyHero")
    p.add_argument("--data-dir", default=Config.data_dir)
    p.add_argument("--output-dir", default=Config.output_dir)
    p.add_argument("--force-download", action="store_true")
    args = p.parse_args()

    root = download_tinyhero(args.data_dir, force=args.force_download)
    stats = validate_dataset(root)
    ds = TinyHeroDataset(root)
    stats["pixels"] = pixel_statistics(ds)
    out = Path(args.output_dir)
    (out / "plots").mkdir(parents=True, exist_ok=True)
    (out / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    save_sample_grid(ds, out / "plots" / "dataset_samples.png", n=25)
    save_view_grid(root, out / "plots" / "dataset_views.png")
    print(json.dumps(stats, indent=2))
    print(f"Saved {out / 'dataset_stats.json'}, {out / 'plots' / 'dataset_samples.png'}, {out / 'plots' / 'dataset_views.png'}")


if __name__ == "__main__":
    main()
