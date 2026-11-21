"""DCGAN Generator and Discriminator for 64x64 RGB sprites (Radford et al., 2016).

Generator:      z (B,100,1,1) -> 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64 RGB (Tanh)
Discriminator:  64x64 RGB -> 32 -> 16 -> 8 -> 4 -> one raw logit per image
The discriminator has NO sigmoid: we train with BCEWithLogitsLoss, which fuses
sigmoid + binary cross-entropy in a numerically stable way.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def weights_init(m: nn.Module) -> None:
    """DCGAN initialisation: conv weights ~ N(0, 0.02), BN gamma ~ N(1, 0.02), beta = 0."""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, 0.0, 0.02)


class Generator(nn.Module):
    def __init__(self, nz: int = 100, ngf: int = 64, nc: int = 3):
        super().__init__()
        self.nz = nz

        def up(cin, cout):
            # kernel 4, stride 2, padding 1 doubles the spatial size exactly
            return [nn.ConvTranspose2d(cin, cout, 4, 2, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(True)]

        self.net = nn.Sequential(
            # (nz,1,1) -> (ngf*8, 4, 4)
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            *up(ngf * 8, ngf * 4),   # -> 8x8
            *up(ngf * 4, ngf * 2),   # -> 16x16
            *up(ngf * 2, ngf),       # -> 32x32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),  # -> 64x64
            nn.Tanh(),               # outputs in [-1, 1], matching normalised data
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 2:
            z = z[:, :, None, None]
        return self.net(z)


# ---- notebook-split ----


class MinibatchStdDev(nn.Module):
    """Appends one feature map holding the average std-dev of features across the batch (Karras et al., 2018).

    Real batches are varied, a collapsed generator's batch is not, so D can detect and punish mode collapse.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.std(dim=0, unbiased=False).mean() if x.size(0) > 1 else x.new_zeros(())
        return torch.cat([x, s.expand(x.size(0), 1, x.size(2), x.size(3))], dim=1)


class Discriminator(nn.Module):
    """spectral_norm=True gives the SN-DCGAN variant (Miyato et al., 2018): every conv weight is divided by
    its largest singular value, limiting how sharply D can react to small input changes. BatchNorm is then
    replaced by Identity, the usual pairing."""

    def __init__(self, ndf: int = 64, nc: int = 3, dropout: float = 0.0, spectral_norm: bool = False,
                 minibatch_std: bool = False):
        super().__init__()
        sn = nn.utils.parametrizations.spectral_norm if spectral_norm else (lambda m: m)

        def down(cin, cout):
            norm = nn.Identity() if spectral_norm else nn.BatchNorm2d(cout)
            layers = [sn(nn.Conv2d(cin, cout, 4, 2, 1, bias=False)), norm, nn.LeakyReLU(0.2, inplace=True)]
            return layers + [nn.Dropout2d(dropout)]  # always present (p=0 is identity) -> stable state_dict keys

        self.net = nn.Sequential(
            # no BatchNorm on the input layer (as recommended in the DCGAN paper)
            sn(nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)),   # 64 -> 32
            nn.LeakyReLU(0.2, inplace=True),
            *down(ndf, ndf * 2),                           # -> 16
            *down(ndf * 2, ndf * 4),                       # -> 8
            *down(ndf * 4, ndf * 8),                       # -> 4
            MinibatchStdDev() if minibatch_std else nn.Identity(),
            sn(nn.Conv2d(ndf * 8 + int(minibatch_std), 1, 4, 1, 0, bias=False)),  # -> 1x1 raw logit (no sigmoid!)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1)


class ConditionalGenerator(nn.Module):
    """Conditional DCGAN generator: G(z, y).

    Each attribute (view, hair colour, outfit colour) has a learned embedding; the
    embeddings are concatenated with z and fed to the same upsampling stack as the
    unconditional Generator. y is a (B, n_attributes) LongTensor of class indices.
    """

    def __init__(self, nz: int = 100, ngf: int = 64, nc: int = 3, n_classes=(4, 9, 9), emb_dim: int = 16):
        super().__init__()
        self.nz, self.n_classes, self.conditional = nz, tuple(n_classes), True
        self.embeds = nn.ModuleList(nn.Embedding(n, emb_dim) for n in self.n_classes)
        self.net = Generator(nz + emb_dim * len(self.n_classes), ngf, nc).net

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        e = torch.cat([emb(y[:, i]) for i, emb in enumerate(self.embeds)], dim=1)
        return self.net(torch.cat([z.flatten(1), e], dim=1)[:, :, None, None])


class ProjectionDiscriminator(nn.Module):
    """Projection discriminator (Miyato & Koyama, 2018) on the DCGAN body:

        D(x, y) = w . phi(x)  +  sum_a  embed_a(y_a) . phi(x)

    phi(x) are the 4x4 features (summed spatially). The projection term is large only
    when the image matches its labels, so the generator is pushed to obey them.
    """

    def __init__(self, ndf: int = 64, nc: int = 3, n_classes=(4, 9, 9), dropout: float = 0.0,
                 spectral_norm: bool = False, minibatch_std: bool = False, pool: str = "mean"):
        super().__init__()
        base = Discriminator(ndf, nc, dropout=dropout, spectral_norm=spectral_norm, minibatch_std=minibatch_std)
        self.n_classes, self.conditional, self.pool = tuple(n_classes), True, pool
        self.features = nn.Sequential(*list(base.net)[:-1])
        self.out = base.net[-1]                     # unconditional 4x4 conv -> 1 logit
        feat = ndf * 8 + int(minibatch_std)
        self.embeds = nn.ModuleList(nn.Embedding(n, feat) for n in self.n_classes)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        out = self.out(h).view(-1)
        phi = h.mean(dim=(2, 3)) if self.pool == "mean" else h.sum(dim=(2, 3))
        for i, emb in enumerate(self.embeds):
            out = out + (emb(y[:, i]) * phi).sum(dim=1)
        return out


class ConcatDiscriminator(nn.Module):
    """Classic conditional GAN discriminator (Mirza & Osindero, 2014): one-hot label maps are
    concatenated to the image as extra input channels, then the usual DCGAN body scores the pair."""

    def __init__(self, ndf: int = 64, nc: int = 3, n_classes=(4, 9, 9), dropout: float = 0.0,
                 spectral_norm: bool = False, minibatch_std: bool = False):
        super().__init__()
        self.n_classes, self.conditional = tuple(n_classes), True
        self.body = Discriminator(ndf, nc + sum(self.n_classes), dropout=dropout, spectral_norm=spectral_norm,
                                  minibatch_std=minibatch_std)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        maps = [nn.functional.one_hot(y[:, i], n).float() for i, n in enumerate(self.n_classes)]
        m = torch.cat(maps, dim=1)[:, :, None, None].expand(-1, -1, x.size(2), x.size(3))
        return self.body(torch.cat([x, m], dim=1))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_models(nz: int = 100, ngf: int = 64, ndf: int = 64, device: torch.device | str = "cpu",
                 d_dropout: float = 0.0, d_spectral_norm: bool = False, d_minibatch_std: bool = False,
                 n_classes: tuple | None = None, d_cond: str = "concat"):
    """n_classes=None -> unconditional DCGAN; e.g. (4, 9, 9) -> conditional G(z, y) and projection D(x, y)."""
    dkw = dict(dropout=d_dropout, spectral_norm=d_spectral_norm, minibatch_std=d_minibatch_std)
    if n_classes:
        G = ConditionalGenerator(nz, ngf, n_classes=n_classes).to(device)
        Dcls = {"projection": ProjectionDiscriminator, "concat": ConcatDiscriminator}[d_cond]
        D = Dcls(ndf, n_classes=n_classes, **dkw).to(device)
    else:
        G = Generator(nz, ngf).to(device)
        D = Discriminator(ndf, **dkw).to(device)
    G.apply(weights_init)
    D.apply(weights_init)
    return G, D


def sample_latent(n: int, nz: int = 100, seed: int | None = None, device: torch.device | str = "cpu") -> torch.Tensor:
    """Sample z ~ N(0, I) on the CPU with an explicit generator, then move to `device`.

    Sampling on the CPU means the same seed gives the same latent vectors on
    CUDA, MPS and CPU, so seeds are portable between Colab and the Mac.
    """
    g = torch.Generator(device="cpu")
    if seed is not None:
        g.manual_seed(int(seed))
    else:
        g.seed()
    return torch.randn(n, nz, 1, 1, generator=g).to(device)


def lerp(z1: torch.Tensor, z2: torch.Tensor, steps: int) -> torch.Tensor:
    """Linear interpolation z(t) = (1 - t) z1 + t z2 for t in [0, 1]."""
    t = torch.linspace(0, 1, steps, device=z1.device).view(-1, *([1] * (z1.dim())))
    return (1 - t) * z1.unsqueeze(0) + t * z2.unsqueeze(0)


def slerp(z1: torch.Tensor, z2: torch.Tensor, steps: int) -> torch.Tensor:
    """Spherical interpolation: keeps intermediate vectors at a 'typical' Gaussian norm."""
    a, b = z1.flatten(), z2.flatten()
    omega = torch.acos(torch.clamp(torch.dot(a / a.norm(), b / b.norm()), -1.0, 1.0))
    if omega.abs() < 1e-6:
        return lerp(z1, z2, steps)
    t = torch.linspace(0, 1, steps, device=z1.device)
    so = torch.sin(omega)
    out = [(torch.sin((1 - ti) * omega) / so) * a + (torch.sin(ti * omega) / so) * b for ti in t]
    return torch.stack(out).view(steps, *z1.shape)


def interpolate_latents(z1: torch.Tensor, z2: torch.Tensor, steps: int = 10, mode: str = "linear") -> torch.Tensor:
    if mode == "linear":
        return lerp(z1, z2, steps)
    if mode == "slerp":
        return slerp(z1, z2, steps)
    raise ValueError(f"Unknown interpolation mode {mode!r}")
