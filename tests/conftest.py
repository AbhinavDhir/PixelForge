import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture
def fake_tinyhero(tmp_path):
    """A tiny dataset with the real TinyHero layout: data/tinyhero/{0..3}/N.png, 64x64 RGBA, binary alpha."""
    root = tmp_path / "data" / "tinyhero"
    rng = np.random.default_rng(0)
    for v in range(4):
        (root / str(v)).mkdir(parents=True)
        for i in range(12):
            a = np.zeros((64, 64, 4), dtype=np.uint8)
            a[16:56, 20:44, :3] = rng.integers(0, 255, size=(40, 24, 3))
            a[16:56, 20:44, 3] = 255
            Image.fromarray(a).save(root / str(v) / f"{i}.png")
    return tmp_path / "data"


@pytest.fixture
def real_data_dir():
    d = ROOT / "data"
    if not (d / "tinyhero" / "0").is_dir():
        pytest.skip("Real TinyHero not downloaded (run python dataset.py)")
    return d
