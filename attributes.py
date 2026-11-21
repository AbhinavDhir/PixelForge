"""Automatic attribute labels for TinyHero + a keyword prompt parser.

TinyHero ships no captions, only the viewing direction (folder name). To make
generation controllable we derive two more labels from the pixels themselves:

* hair  - colour of the top of the head (rows 9-17), i.e. hair or headwear
* outfit - colour of the torso (rows 33-43)

Each opaque pixel in the region is classified into a named colour (HSV rules);
dark outline pixels are ignored; the majority colour wins. Labels are therefore
approximate: a grey helmet is labelled "grey" hair, a two-colour shirt gets its
dominant colour. The prompt parser maps words in a description to these labels.
"""
from __future__ import annotations

import colorsys
import re
from pathlib import Path

import numpy as np
from PIL import Image

VIEWS = ["back", "left", "front", "right"]                     # index = TinyHero folder
HAIR = ["black", "brown", "blonde", "red", "white", "grey", "green", "blue", "purple"]
OUTFIT = ["black", "brown", "yellow", "red", "white", "grey", "green", "blue", "purple"]
ATTRIBUTES = {"view": VIEWS, "hair": HAIR, "outfit": OUTFIT}

# region boxes (y0, y1, x0, x1) in the 64x64 frame, measured from the dataset
HAIR_BOX = (9, 18, 22, 42)
OUTFIT_BOX = (33, 44, 24, 40)


def _pixel_colour(r: float, g: float, b: float) -> str | None:
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    h *= 360
    if v < 0.18:
        return None                       # outline / shadow pixel: ignore
    if v < 0.32:
        return "black"
    if s < 0.22:
        return "white" if v > 0.78 else "grey"
    if h < 18 or h >= 335:
        return "red"
    if h < 45:
        return "brown" if v < 0.62 else "yellow"   # dark orange = brown, light = blonde/yellow
    if h < 70:
        return "yellow"
    if h < 170:
        return "green"
    if h < 225 or (h < 255 and s >= 0.5):
        return "blue"                     # saturated blues (hair dye, blue cloth)
    return "purple"                       # incl. the common desaturated lavender LPC shirt


def region_colour(rgba: np.ndarray, box: tuple[int, int, int, int]) -> str | None:
    y0, y1, x0, x1 = box
    reg = rgba[y0:y1, x0:x1]
    px = reg[reg[..., 3] > 0][:, :3] / 255.0
    names = [n for n in (_pixel_colour(*p) for p in px) if n is not None]
    if not names:
        return None
    vals, counts = np.unique(names, return_counts=True)
    return str(vals[np.argmax(counts)])


def label_sprite(path: str | Path) -> dict:
    rgba = np.asarray(Image.open(path).convert("RGBA"))
    hair = region_colour(rgba, HAIR_BOX) or "black"
    outfit = region_colour(rgba, OUTFIT_BOX) or "black"
    hair = {"yellow": "blonde"}.get(hair, hair)
    return {"hair": HAIR.index(hair), "outfit": OUTFIT.index(outfit)}


# --------------------------------------------------------------------------- prompt parsing
_SYNONYMS = {
    "view": {"back": ["back", "behind", "rear", "away"], "left": ["left"], "front": ["front", "facing", "forward", "toward"],
             "right": ["right"]},
    "colour": {"black": ["black", "dark", "raven", "ebony"], "brown": ["brown", "brunette", "chestnut", "leather"],
               "blonde": ["blonde", "blond", "golden", "gold", "yellow"], "yellow": ["yellow", "gold", "golden"],
               "red": ["red", "ginger", "crimson", "scarlet", "orange"], "white": ["white", "silver", "snow"],
               "grey": ["grey", "gray", "silver", "steel", "metal", "iron", "chainmail", "knight"],
               "green": ["green", "emerald", "forest"], "blue": ["blue", "azure", "navy", "cyan"],
               "purple": ["purple", "violet", "pink", "magenta", "lilac"]},
}
_HAIR_WORDS = ["hair", "haired", "head", "helmet", "helm", "hood", "hat", "headwear", "bald", "ponytail", "braid"]
_OUTFIT_WORDS = ["outfit", "shirt", "armour", "armor", "clothes", "clothing", "tunic", "robe", "dress", "chest",
                 "torso", "body", "wearing", "dressed", "vest", "top", "cloak", "suit", "chainmail"]


def _colour_of(word: str, allowed: list[str]) -> str | None:
    for name in allowed:
        if word in _SYNONYMS["colour"].get(name, []) or word == name:
            return name
    return None


_HAIR_ONLY = {"blonde", "blond", "ginger", "brunette", "raven", "bald"}


def parse_prompt(prompt: str) -> tuple[dict, list[str]]:
    """Map a free-text description to attribute labels (None = not specified -> random).

    The prompt is split into clauses at punctuation and at "and"/"with"; inside a
    clause each colour word attaches to the nearest hair noun (hair, helmet, hood...)
    or outfit noun (tunic, armour, dress...). Words like "blonde" always mean hair.
    Returns (labels, notes) where notes explain what was understood.
    """
    text = prompt.lower()
    words = re.findall(r"[a-z]+", text)
    labels: dict = {"view": None, "hair": None, "outfit": None}

    explicit = {"back": ["back", "behind", "rear", "away"], "left": ["left"], "right": ["right"],
                "front": ["front", "frontal"]}
    for w in words:
        for name, syns in explicit.items():
            if w in syns and labels["view"] is None:
                labels["view"] = VIEWS.index(name)
    if labels["view"] is None and any(w in ("facing", "forward", "toward", "towards", "camera") for w in words):
        labels["view"] = VIEWS.index("front")

    colour_words = {w for syns in _SYNONYMS["colour"].values() for w in syns} | set(_SYNONYMS["colour"])
    for clause in re.split(r"[,;.!?\n]|\band\b|\bwith\b", text):
        cw = re.findall(r"[a-z]+", clause)
        nouns = [(j, "hair") for j, w in enumerate(cw) if w in _HAIR_WORDS] + \
                [(j, "outfit") for j, w in enumerate(cw) if w in _OUTFIT_WORDS]
        for i, w in enumerate(cw):
            if w not in colour_words:
                continue
            if w in _HAIR_ONLY:
                target = "hair"
            elif nouns:   # nearest noun in the clause; ties go to the following noun ("red tunic")
                target = min(nouns, key=lambda n: (abs(n[0] - i), n[0] < i))[1]
            else:
                target = "hair" if labels["hair"] is None else "outfit"
            vocab = HAIR if target == "hair" else OUTFIT
            c = _colour_of(w, vocab)
            if c is not None and labels[target] is None:
                labels[target] = vocab.index(c)

    if labels["hair"] is None and any(w in ("helmet", "helm", "knight") for w in words):
        labels["hair"] = HAIR.index("grey")            # metal headwear is labelled grey in the data
    if labels["outfit"] is None and any(w in ("chainmail", "armour", "armor", "knight") for w in words):
        labels["outfit"] = OUTFIT.index("grey")
    notes = [f"{k}: {vocab[labels[k]]}" if labels[k] is not None else f"{k}: (not specified - random)"
             for k, vocab in ATTRIBUTES.items()]
    return labels, notes


# --------------------------------------------------------------------------- dataset labels + generated images
def label_image(img: Image.Image, background=(0, 0, 0), tol: int = 24) -> dict:
    """Label an RGB image (e.g. a generated sprite): pixels far from the background count as opaque."""
    rgb = np.asarray(img.convert("RGB"))
    alpha = (np.abs(rgb.astype(np.int16) - np.array(background, dtype=np.int16)).max(axis=2) > tol) * 255
    rgba = np.dstack([rgb, alpha.astype(np.uint8)])
    hair = region_colour(rgba, HAIR_BOX) or "black"
    outfit = region_colour(rgba, OUTFIT_BOX) or "black"
    hair = {"yellow": "blonde"}.get(hair, hair)
    return {"hair": HAIR.index(hair), "outfit": OUTFIT.index(outfit)}


def dataset_labels(root: str | Path, items: list[tuple[Path, int]]) -> np.ndarray:
    """(N, 3) int array [view, hair, outfit]; cached in <root>/../tinyhero_attributes.csv for inspection."""
    import csv
    root = Path(root)
    cache = root.parent / "tinyhero_attributes.csv"
    rows = {}
    if cache.exists():
        with open(cache, newline="") as f:
            rows = {r["file"]: r for r in csv.DictReader(f)}
    out, changed = [], False
    for path, view in items:
        key = f"{path.parent.name}/{path.name}"
        r = rows.get(key)
        if r is None:
            lab = label_sprite(path)
            r = rows[key] = {"file": key, "view": VIEWS[view], "hair": HAIR[lab["hair"]],
                             "outfit": OUTFIT[lab["outfit"]]}
            changed = True
        out.append([VIEWS.index(r["view"]), HAIR.index(r["hair"]), OUTFIT.index(r["outfit"])])
    if changed:  # merge: never drop rows labelled earlier (e.g. a smoke-test subset then the full set)
        with open(cache, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["file", "view", "hair", "outfit"])
            w.writeheader()
            w.writerows(sorted(rows.values(), key=lambda r: r["file"]))
    return np.array(out, dtype=np.int64)
