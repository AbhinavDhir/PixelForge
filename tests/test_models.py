import pytest
import torch
import torch.nn as nn

from config import get_device
from models import (Discriminator, Generator, build_models, count_parameters, interpolate_latents,
                    sample_latent)


def test_generator_output_shape():
    G = Generator(100, 64)
    out = G(torch.randn(5, 100, 1, 1))
    assert out.shape == (5, 3, 64, 64)
    assert out.min() >= -1 and out.max() <= 1  # tanh range
    assert G(torch.randn(2, 100)).shape == (2, 3, 64, 64)  # flat z also accepted


def test_discriminator_output_shape():
    D = Discriminator(64)
    out = D(torch.randn(7, 3, 64, 64))
    assert out.shape == (7,)
    assert not isinstance(list(D.net)[-1], nn.Sigmoid)  # raw logits for BCEWithLogitsLoss


def test_parameter_counts():
    G, D = build_models(100, 64, 64)
    assert count_parameters(G) == 3_576_704
    assert count_parameters(D) == 2_765_568


def test_weight_init():
    G, _ = build_models()
    w = G.net[0].weight
    assert abs(w.mean().item()) < 5e-3 and abs(w.std().item() - 0.02) < 5e-3


def test_forward_and_backward():
    G, D = build_models()
    z = torch.randn(4, 100, 1, 1)
    loss = nn.BCEWithLogitsLoss()(D(G(z)), torch.ones(4))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in G.parameters())
    assert all(p.grad is not None for p in D.parameters())


def _snapshot(m):
    return [p.detach().clone() for p in m.parameters()]


def test_one_discriminator_step_updates_only_D():
    torch.manual_seed(0)
    G, D = build_models()
    opt_d = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    crit = nn.BCEWithLogitsLoss()
    g0, d0 = _snapshot(G), _snapshot(D)
    real = torch.rand(8, 3, 64, 64) * 2 - 1
    fake = G(torch.randn(8, 100, 1, 1))
    loss = crit(D(real), torch.full((8,), 0.9)) + crit(D(fake.detach()), torch.zeros(8))
    opt_d.zero_grad(); loss.backward(); opt_d.step()
    assert any(not torch.equal(a, b) for a, b in zip(d0, D.parameters()))
    assert all(torch.equal(a, b) for a, b in zip(g0, G.parameters()))
    assert all(p.grad is None for p in G.parameters())  # detach() blocked G gradients


def test_one_generator_step_updates_G():
    torch.manual_seed(0)
    G, D = build_models()
    opt_g = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    g0 = _snapshot(G)
    loss = nn.BCEWithLogitsLoss()(D(G(torch.randn(8, 100, 1, 1))), torch.ones(8))
    opt_g.zero_grad(); loss.backward(); opt_g.step()
    assert any(not torch.equal(a, b) for a, b in zip(g0, G.parameters()))


def test_sample_latent_deterministic():
    assert torch.equal(sample_latent(4, seed=7), sample_latent(4, seed=7))
    assert not torch.equal(sample_latent(4, seed=7), sample_latent(4, seed=8))


@pytest.mark.parametrize("mode", ["linear", "slerp"])
def test_latent_interpolation(mode):
    z1, z2 = sample_latent(1, seed=1)[0], sample_latent(1, seed=2)[0]
    zs = interpolate_latents(z1, z2, 10, mode)
    assert zs.shape == (10, 100, 1, 1)
    assert torch.allclose(zs[0], z1, atol=1e-5) and torch.allclose(zs[-1], z2, atol=1e-5)
    if mode == "linear":
        assert torch.allclose(zs[5], z1 + (z2 - z1) * 5 / 9, atol=1e-5)


def test_device_selection():
    d = get_device()
    if torch.cuda.is_available():
        assert d.type == "cuda"
    elif torch.backends.mps.is_available():
        assert d.type == "mps"
    else:
        assert d.type == "cpu"
    assert get_device("cpu").type == "cpu"
    G = Generator().to(d)
    assert G(sample_latent(2, device=d)).device.type == d.type


@pytest.mark.parametrize("kw", [dict(minibatch_std=True), dict(spectral_norm=True), dict(dropout=0.3),
                                dict(minibatch_std=True, spectral_norm=True, dropout=0.2)])
def test_discriminator_variants(kw):
    D = Discriminator(64, **kw)
    out = D(torch.randn(6, 3, 64, 64))
    assert out.shape == (6,)
    out.sum().backward()
    assert all(p.grad is not None for p in D.parameters() if p.requires_grad)


def test_dropout_keeps_state_dict_keys():
    assert Discriminator(dropout=0.0).state_dict().keys() == Discriminator(dropout=0.4).state_dict().keys()


def test_minibatch_std_sees_batch_variety():
    from models import MinibatchStdDev
    same = torch.ones(8, 4, 4, 4)
    varied = torch.randn(8, 4, 4, 4)
    assert MinibatchStdDev()(same)[:, -1].abs().max() == 0          # collapsed batch -> 0
    assert MinibatchStdDev()(varied)[:, -1].mean() > 0.5            # varied batch -> large
