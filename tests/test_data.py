import torch

from config import Config
from dataset import TinyHeroDataset, load_rgb, make_dataloader, to_tensor, validate_dataset


def test_dataset_loading(fake_tinyhero):
    ds = TinyHeroDataset(fake_tinyhero / "tinyhero")
    assert len(ds) == 48
    x, y = ds[0]
    assert x.shape == (3, 64, 64) and x.dtype == torch.float32
    assert sorted(torch.unique(ds.labels).tolist()) == [0, 1, 2, 3]


def test_view_filter_and_subset(fake_tinyhero):
    assert len(TinyHeroDataset(fake_tinyhero / "tinyhero", views=(2,))) == 12
    assert len(TinyHeroDataset(fake_tinyhero / "tinyhero", subset=10)) == 10


def test_normalization_range(fake_tinyhero):
    ds = TinyHeroDataset(fake_tinyhero / "tinyhero")
    x = torch.stack([ds[i][0] for i in range(len(ds))])
    assert x.min() >= -1.0 and x.max() <= 1.0
    # transparent corner composited onto the default black background -> exactly -1 after normalisation
    assert torch.allclose(x[:, :, 0, 0], -torch.ones(len(ds), 3))
    white = TinyHeroDataset(fake_tinyhero / "tinyhero", background=(255, 255, 255))
    assert torch.allclose(white[0][0][:, 0, 0], torch.ones(3))


def test_alpha_composite_background(fake_tinyhero):
    p = fake_tinyhero / "tinyhero" / "0" / "0.png"
    assert load_rgb(p, (255, 255, 255)).getpixel((0, 0)) == (255, 255, 255)
    assert load_rgb(p, (10, 20, 30)).getpixel((0, 0)) == (10, 20, 30)
    t = to_tensor(load_rgb(p))
    assert t.shape == (3, 64, 64)


def test_dataloader_batches(fake_tinyhero):
    ds = TinyHeroDataset(fake_tinyhero / "tinyhero")
    xb, yb = next(iter(make_dataloader(ds, batch_size=16)))
    assert xb.shape == (16, 3, 64, 64) and yb.shape == (16,)


def test_validate_fake(fake_tinyhero):
    s = validate_dataset(fake_tinyhero / "tinyhero")
    assert s["total_images"] == 48 and s["sizes"] == {"64x64": 48} and s["characters_in_all_views"] == 12


def test_real_tinyhero_matches_documentation(real_data_dir):
    s = validate_dataset(real_data_dir / "tinyhero")
    assert s["total_images"] == 3648
    assert s["sizes"] == {"64x64": 3648}
    assert s["characters_in_all_views"] == 912
    assert s["matches_documentation"]
