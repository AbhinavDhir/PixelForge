"""Prompt parsing, attribute labels and conditional generation."""
import pytest
import torch

from attributes import HAIR, OUTFIT, VIEWS, label_image, label_sprite, parse_prompt
from config import Config
from generate import generate, interpolate, load_generator, make_labels
from models import build_models
from train import train


@pytest.mark.parametrize("prompt,expected", [
    ("a front-facing hero with green hair and a red tunic", ("front", "green", "red")),
    ("knight in armour seen from behind", ("back", "grey", "grey")),
    ("blonde girl wearing a blue dress, facing left", ("left", "blonde", "blue")),
    ("hair: purple, outfit: white", (None, "purple", "white")),
    ("red-haired rogue in a brown leather vest, right side", ("right", "red", "brown")),
    ("a wizard", (None, None, None)),
])
def test_parse_prompt(prompt, expected):
    labels, notes = parse_prompt(prompt)
    got = tuple(v[labels[k]] if labels[k] is not None else None
                for k, v in (("view", VIEWS), ("hair", HAIR), ("outfit", OUTFIT)))
    assert got == expected
    assert len(notes) == 3


def test_label_image_agrees_with_alpha_labels(real_data_dir):
    """The background-based labeller (used on generated images) should agree with the alpha-based one."""
    from dataset import list_images, load_rgb
    items = list_images(real_data_dir / "tinyhero")[::40]
    agree = sum(label_image(load_rgb(p)) == label_sprite(p) for p, _ in items)
    assert agree / len(items) > 0.8


def test_conditional_generation_follows_labels():
    G, _ = build_models(n_classes=(4, 9, 9))
    G.eval()
    y = make_labels(G, 6, {"view": 2, "hair": 1, "outfit": None}, seed=3)
    assert y.shape == (6, 3) and (y[:, 0] == 2).all() and (y[:, 1] == 1).all()
    assert torch.equal(y, make_labels(G, 6, {"view": 2, "hair": 1, "outfit": None}, seed=3))  # seeded
    imgs = generate(G, 6, seed=1, labels={"view": 0})
    assert len(imgs) == 6 and imgs[0].size == (64, 64)
    assert len(interpolate(G, 1, 2, steps=8, labels={"hair": 3})) == 8


def test_unconditional_mode_still_trains(tmp_path, fake_tinyhero):
    cfg = Config(data_dir=str(fake_tinyhero), epochs=1, batch_size=16, conditional=False, ema_decay=0,
                 checkpoint_dir=str(tmp_path / "c"), output_dir=str(tmp_path / "o"), device="cpu", num_fixed_samples=16)
    train(cfg, show_progress=False)
    G, cfg2, meta = load_generator(tmp_path / "c" / "generator_final.pt")
    assert not getattr(G, "conditional", False) and not cfg2.conditional and meta["ema"] is False


def test_conditional_checkpoint_roundtrip(tmp_path, fake_tinyhero):
    cfg = Config(data_dir=str(fake_tinyhero), epochs=1, batch_size=16, checkpoint_dir=str(tmp_path / "c"),
                 output_dir=str(tmp_path / "o"), device="cpu", num_fixed_samples=16)
    train(cfg, show_progress=False)
    G, cfg2, meta = load_generator(tmp_path / "c" / "generator_final.pt")
    assert G.conditional and cfg2.conditional and meta["ema"] is True and G.n_classes == (4, 9, 9)


@pytest.mark.parametrize("d_cond", ["concat", "projection"])
def test_conditional_discriminators(d_cond):
    G, D = build_models(n_classes=(4, 9, 9), d_cond=d_cond)
    z = torch.randn(4, 100, 1, 1)
    y = torch.tensor([[0, 1, 2], [3, 8, 0], [2, 2, 2], [1, 0, 8]])
    out = D(G(z, y), y)
    assert out.shape == (4,)
    out.mean().backward()
    assert all(p.grad is not None for p in G.parameters())
    # changing only the labels must change D's score (D actually looks at y)
    with torch.no_grad():
        x = torch.randn(4, 3, 64, 64)
        assert not torch.allclose(D(x, y), D(x, (y + 1) % torch.tensor([4, 9, 9])))


def test_label_cache_merges_subsets(fake_tinyhero):
    from attributes import dataset_labels
    from dataset import list_images
    root = fake_tinyhero / "tinyhero"
    items = list_images(root)
    dataset_labels(root, items[:5])
    dataset_labels(root, items[5:9])
    lines = (root.parent / "tinyhero_attributes.csv").read_text().strip().splitlines()
    assert len(lines) == 1 + 9  # header + both subsets kept
