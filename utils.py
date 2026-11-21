"""Image helpers shared by training, generation, evaluation and the app."""
from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def denormalize(x: torch.Tensor) -> torch.Tensor:
    """Map generator output from [-1, 1] back to [0, 1]."""
    return ((x.detach().float().cpu() + 1.0) / 2.0).clamp(0.0, 1.0)


def tensor_to_pil(x: torch.Tensor) -> list[Image.Image]:
    """(B,3,H,W) tensor in [-1,1] -> list of RGB PIL images (uint8)."""
    if x.dim() == 3:
        x = x.unsqueeze(0)
    arr = (denormalize(x).permute(0, 2, 3, 1).numpy() * 255.0).round().astype(np.uint8)
    return [Image.fromarray(a) for a in arr]


def upscale(img: Image.Image, scale: int) -> Image.Image:
    """Nearest-neighbour upscaling keeps hard pixel edges (no blur)."""
    if scale == 1:
        return img
    return img.resize((img.width * scale, img.height * scale), Image.NEAREST)


def make_grid(images: list[Image.Image], nrow: int | None = None, padding: int = 2,
              pad_color=(20, 16, 36), scale: int = 1) -> Image.Image:
    """Tile PIL images into a grid. nrow = images per row (default: square)."""
    if not images:
        raise ValueError("make_grid() needs at least one image")
    n = len(images)
    nrow = nrow or int(np.ceil(np.sqrt(n)))
    ncol_rows = int(np.ceil(n / nrow))
    w, h = images[0].size
    mode = images[0].mode
    fill = pad_color if mode == "RGB" else (*pad_color, 0)
    grid = Image.new(mode, (nrow * w + (nrow + 1) * padding, ncol_rows * h + (ncol_rows + 1) * padding), fill)
    for i, im in enumerate(images):
        r, c = divmod(i, nrow)
        grid.paste(im, (padding + c * (w + padding), padding + r * (h + padding)))
    return upscale(grid, scale)


def save_gif(frames: list[Image.Image], path: str | Path, duration_ms: int = 120, bounce: bool = True) -> Path:
    """Save an animated GIF (optionally ping-pong so the loop is seamless)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    seq = frames + frames[-2:0:-1] if bounce and len(frames) > 2 else frames
    seq[0].save(path, save_all=True, append_images=seq[1:], duration=duration_ms, loop=0, disposal=2)
    return path


def remove_background(img: Image.Image, bg=(0, 0, 0), tolerance: int = 24) -> Image.Image:
    """OPTIONAL post-processing: approximate transparency for a generated RGB sprite.

    The GAN only ever outputs RGB. Because training sprites were composited onto a
    solid background colour, generated backgrounds are *approximately* that colour.
    We flood-fill from the image border, marking pixels within `tolerance`
    (max abs channel difference) of `bg` as transparent.

    Limitations: with the default black background, dark outline pixels that touch
    the background can be removed too (lower `tolerance` to keep more); noisy
    backgrounds leave speckles; this is a heuristic, not something the model learned.
    """
    rgb = np.asarray(img.convert("RGB")).astype(np.int16)
    h, w, _ = rgb.shape
    close = (np.abs(rgb - np.array(bg, dtype=np.int16)).max(axis=2) <= tolerance)
    mask = np.zeros((h, w), dtype=bool)
    q = deque((y, x) for y in range(h) for x in (0, w - 1) if close[y, x])
    q.extend((y, x) for x in range(w) for y in (0, h - 1) if close[y, x])
    while q:
        y, x = q.popleft()
        if mask[y, x]:
            continue
        mask[y, x] = True
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and close[ny, nx] and not mask[ny, nx]:
                q.append((ny, nx))
    alpha = np.where(mask, 0, 255).astype(np.uint8)
    rgba = np.dstack([rgb.astype(np.uint8), alpha])
    return Image.fromarray(rgba)
