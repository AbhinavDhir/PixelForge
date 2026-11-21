"""Checkpointing, generation and an end-to-end smoke test:
Dataset -> DataLoader -> DCGAN -> training iterations -> checkpoint -> resume -> inference -> PNG."""
import csv

import torch
from PIL import Image

from config import Config
from generate import generate, interpolate, load_generator, save_images
from models import build_models
from train import load_checkpoint, save_checkpoint, train
from utils import make_grid, remove_background, save_gif


def _cfg(tmp_path, data_dir, **kw):
    c = Config(data_dir=str(data_dir), epochs=2, batch_size=16, sample_every=1, checkpoint_every=1,
               num_fixed_samples=16, checkpoint_dir=str(tmp_path / "ckpt"), output_dir=str(tmp_path / "out"),
               device="cpu")
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def test_checkpoint_save_load_roundtrip(tmp_path):
    G, D = build_models()
    og = torch.optim.Adam(G.parameters(), lr=2e-4)
    od = torch.optim.Adam(D.parameters(), lr=2e-4)
    # take a step so optimizer state is non-empty
    D(G(torch.randn(2, 100, 1, 1))).mean().backward(); og.step(); od.step()
    p = save_checkpoint(tmp_path / "c.pt", G, D, og, od, 3, Config(conditional=False), [{"epoch": 1, "loss_d": 1.0}],
                        torch.randn(4, 100, 1, 1), "h.csv")
    ck = load_checkpoint(p)  # weights_only=True must succeed (no pickled objects)
    assert ck["epoch"] == 3 and ck["config"]["nz"] == 100
    G2, D2 = build_models()
    G2.load_state_dict(ck["generator"]); D2.load_state_dict(ck["discriminator"])
    for a, b in zip(G.state_dict().values(), G2.state_dict().values()):
        assert torch.equal(a, b)
    og2 = torch.optim.Adam(G2.parameters(), lr=2e-4)
    og2.load_state_dict(ck["opt_g"])
    assert len(og2.state) == len(og.state) > 0
    Gl, cfg, meta = load_generator(p, "cpu")
    assert not Gl.training and meta["epoch"] == 3


def test_png_generation_and_interpolation(tmp_path):
    G, _ = build_models()
    G.eval()
    imgs = generate(G, 9, seed=5)
    assert len(imgs) == 9 and imgs[0].size == (64, 64) and imgs[0].mode == "RGB"
    assert list(imgs[0].getdata()) == list(generate(G, 9, seed=5)[0].getdata())  # deterministic seed
    paths = save_images(imgs, tmp_path / "gen", scale=2, transparent=True)
    assert Image.open(paths[0]).size == (128, 128)
    assert Image.open(tmp_path / "gen" / "character_000_transparent.png").mode == "RGBA"
    grid = make_grid(imgs, scale=4)
    assert grid.size == ((3 * 64 + 4 * 2) * 4,) * 2
    frames = interpolate(G, 1, 2, steps=10)
    assert len(frames) == 10
    assert save_gif(frames, tmp_path / "i.gif").exists()


def test_remove_background():
    im = Image.new("RGB", (64, 64), (4, 6, 2))        # near-black generated background
    im.paste((200, 0, 0), (20, 20, 40, 40))
    rgba = remove_background(im)
    assert rgba.getpixel((0, 0))[3] == 0 and rgba.getpixel((30, 30))[3] == 255


def test_end_to_end_smoke_and_resume(tmp_path, fake_tinyhero):
    cfg = _cfg(tmp_path, fake_tinyhero, epochs=2)
    res = train(cfg, show_progress=False)
    assert res["epochs_completed"] == 2
    ck_dir, out = tmp_path / "ckpt", tmp_path / "out"
    for f in ("latest.pt", "generator_final.pt", "pixelforge_epoch_001.pt", "pixelforge_epoch_002.pt"):
        assert (ck_dir / f).exists(), f
    for f in ("training_history.csv", "training_iterations.csv", "training_progress/epoch_000.png",
              "training_progress/epoch_002.png", "plots/losses.png"):
        assert (out / f).exists(), f
    rows = list(csv.DictReader(open(out / "training_history.csv")))
    assert [int(r["epoch"]) for r in rows] == [1, 2]

    # resume to epoch 3 keeps optimizer state and appends history
    before = load_checkpoint(ck_dir / "latest.pt")
    cfg.epochs = 3
    res2 = train(cfg, resume="auto", show_progress=False)
    assert res2["epochs_completed"] == 3
    after = load_checkpoint(ck_dir / "latest.pt")
    assert [h["epoch"] for h in after["history"]] == [1, 2, 3]
    step_before = before["opt_g"]["state"][0]["step"]
    step_after = after["opt_g"]["state"][0]["step"]
    assert float(step_after) > float(step_before)  # Adam step counter continued, not reset

    G, _, meta = load_generator(ck_dir / "generator_final.pt", "cpu")
    paths = save_images(generate(G, 4, seed=0), tmp_path / "png")
    assert meta["epoch"] == 3 and all(p.exists() for p in paths)


def test_sample_diversity_detects_collapse():
    from train import sample_diversity

    class Constant(torch.nn.Module):          # a fully collapsed generator
        def forward(self, z):
            return torch.zeros(z.shape[0], 3, 64, 64)

    class Varied(torch.nn.Module):            # output depends on z
        def forward(self, z):
            return torch.tanh(z.flatten(1)[:, :1, None, None]).expand(-1, 3, 64, 64)

    z = torch.randn(16, 100, 1, 1)
    assert sample_diversity(Constant(), z) == 0.0
    assert sample_diversity(Varied(), z) > 0.1


def test_stabilisers_train_and_resume_keeps_architecture(tmp_path, fake_tinyhero):
    cfg = _cfg(tmp_path, fake_tinyhero, epochs=1, d_minibatch_std=True, d_spectral_norm=True,
               instance_noise=0.1, d_dropout=0.2, real_label_jitter=0.05, fake_label_max=0.1)
    train(cfg, show_progress=False)
    # resume with a *default* config: architecture must be taken from the checkpoint, not the CLI
    cfg2 = _cfg(tmp_path, fake_tinyhero, epochs=2)
    res = train(cfg2, resume="auto", show_progress=False)
    assert res["epochs_completed"] == 2
    ck = load_checkpoint(tmp_path / "ckpt" / "latest.pt")
    assert ck["config"]["d_minibatch_std"] and ck["config"]["d_spectral_norm"]
