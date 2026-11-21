# Experiments

See the main README, section 18, "Experiment log".

### Experiment log: mode collapse and the background colour

The first full run (white background, standard DCGAN, batch 32) **mode-collapsed**. By epoch 5 all 64
fixed latent vectors decoded to the same sprite, and by epoch 23 the variation across samples was ~6% of
the real data's. D dominated from the first epoch (D(G(z)) ≈ 0.003, loss_G ≈ 6). CPU and MPS gave identical
outputs and gradients (relative difference ~1e-6), and MPS `randn` was correct, so it was not a device
bug. Grids and logs are in `outputs/experiments/01_plain_dcgan_mode_collapse/`.

Controlled 6-epoch probes (seed 42, batch 64, everything else default). "Diversity" is the per-pixel std
across 64 fixed-noise samples, as a percentage of the same measure on 64 real sprites with the same
background (white 0.113, black 0.075):

| Probe (background) | Diversity at epoch 6 | Visual check |
|---|---|---|
| Plain DCGAN (white) | 16% | |
| + instance noise 0.1 (white) | 20% | |
| + D dropout 0.3 (white) | 17% | |
| + instance noise + dropout (white) | 22% | |
| + spectral norm D (white) | 9% | D too weak (D(G(z)) ≈ 0.5, loss_G ≈ ln 2) |
| + spectral norm + noise (white) | 12% | |
| + minibatch std-dev (white) | 16% | collapsed (one sprite) |
| + minibatch std + noise + dropout (white) | 19% | collapsed (one sprite) |
| **Plain DCGAN (black)** | **37%** | **diverse: different colours, heads, bodies** |
| Original TinyHero recipe (black; nz 84, 48 ch, noisy labels, dropout 0.2) | 32% | some variety |

**Conclusion:** the background colour was the deciding factor. With ~76% of every image an exactly uniform
white field, the discriminator separated real from fake almost immediately, and the generator retreated to
one "safe" sprite. Standard stabilisers delayed this but did not prevent it. On black, the plain DCGAN
trains stably, so PixelForge uses black by default and keeps the stabilisers as optional flags.
Per-probe histories and epoch-6 grids are in `outputs/experiments/probes/`. (These are short probes on one
seed: evidence, not proof.)
