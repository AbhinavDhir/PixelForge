"""Build PixelForge_Complete.ipynb from the project modules.

The notebook must run in a fresh Colab session with no project files, so it
re-creates each module with %%writefile. Generating the notebook from the .py
files guarantees the notebook and the scripts never drift apart:

    python tools/build_notebook.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(text: str) -> None:
    cells.append({"cell_type": "markdown", "metadata": {}, "source": _lines(text.strip("\n"))})


def code(text: str) -> None:
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": _lines(text.strip("\n"))})


def _lines(s: str) -> list[str]:
    parts = s.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


def writefile(name: str, src: str | None = None, append: bool = False) -> None:
    src = src if src is not None else (ROOT / name).read_text()
    code(f"%%writefile {'-a ' if append else ''}{name}\n{src.rstrip()}\n")


models_src = (ROOT / "models.py").read_text()
models_a, models_b = models_src.split("# ---- notebook-split ----\n")

# =========================================================================== 1-6 theory
md(r"""
# 🗡️ PixelForge — Prompt-Controlled Pixel Art Character Generator (Conditional DCGAN)

<a href="https://colab.research.google.com/" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## 1. Introduction

**PixelForge** trains a **conditional Deep Convolutional GAN** from scratch on **TinyHero**, a public dataset of
3,648 retro 64×64 RPG character sprites. You describe a character in words, for example
*"a front-facing hero with green hair and a red tunic"*, and the generator draws **brand-new characters** that match.

How the prompt reaches the GAN: TinyHero has no captions, so we **label every sprite automatically** with its
viewing direction (from the folder), hair/headwear colour and outfit colour (from its pixels). A keyword parser
turns the prompt into those labels $y$, and the generator draws $G(z, y)$.

No pretrained image models, diffusion models or external APIs are used: every generated sprite is produced by
the generator network trained in this notebook.

**How to run (Google Colab Free):**
1. `Runtime → Change runtime type → T4 GPU` → Save.
2. `Runtime → Run all`.
3. Full training (100 epochs on all 3,648 images) takes about **25 minutes on an Apple M2** (measured). A T4 should be faster (not measured).
4. The final cell downloads `pixelforge_results.zip` (checkpoints, plots, sprites) for the Streamlit app.

If no GPU is allocated the notebook switches to a short **smoke test** automatically (Section 9) and says so.
""")

md(r"""
## 2. Project objectives

1. Download, validate and preprocess the TinyHero sprite dataset.
2. Derive attribute labels (view, hair colour, outfit colour) automatically and parse text prompts into them.
3. Implement a conditional DCGAN (Generator $G(z, y)$ + Discriminator $D(x, y)$) in PyTorch.
4. Train it adversarially with reproducible seeds, logging, checkpoints and resumable state.
5. Generate new 64×64 RGB characters from text prompts and export them as PNG files.
6. Explore the latent space with interpolation strips and GIFs.
7. Evaluate honestly: losses, progression, real-vs-generated, diversity, memorisation and **prompt controllability**.
8. Export a portable checkpoint that runs on CUDA (Colab), MPS (Apple Silicon) or CPU.
""")

md(r"""
## 3. What is a GAN?

A **Generative Adversarial Network** (Goodfellow et al., 2014) learns to produce samples from a data distribution
$p_{data}$ by pitting two neural networks against each other:

```
             random noise z ~ N(0, I)
                     │
                     ▼
              ┌─────────────┐   fake sprite G(z)
              │  GENERATOR  │ ───────────────────┐
              └─────────────┘                    ▼
                     ▲                    ┌───────────────┐
                     │  "how to fool D"   │ DISCRIMINATOR │ ──► real or fake?
                     └─── gradients ◄──── └───────────────┘
                                                 ▲
                    real sprite x ~ TinyHero ────┘
```

The generator never sees a real image directly. It only learns through the discriminator's gradients:
*"change your pixels in this direction to look more real."*
""")

md(r"""
## 4. Generator versus Discriminator

| | **Generator G** | **Discriminator D** |
|---|---|---|
| Input | latent vector $z \in \mathbb{R}^{100}$, $z\sim\mathcal{N}(0,I)$ | image $x \in \mathbb{R}^{3\times64\times64}$ |
| Output | image $G(z)$ in $[-1,1]$ | one logit (real vs fake score) |
| Goal | make $D(G(z))$ look *real* | separate real from generated images |
| Layers | ConvTranspose2d + BatchNorm + ReLU, Tanh output | Conv2d + BatchNorm + LeakyReLU, raw logit |
| Analogy | counterfeiter | detective |

They improve together: a better detective forces a better counterfeiter.
""")

md(r"""
## 5. Mathematical GAN objective

The original minimax game:

$$
\min_G \max_D \; V(D,G) = \mathbb{E}_{x\sim p_{data}}\big[\log D(x)\big] + \mathbb{E}_{z\sim p_z}\big[\log\big(1 - D(G(z))\big)\big]
$$

For a fixed $G$ the optimal discriminator is $D^*(x) = \dfrac{p_{data}(x)}{p_{data}(x) + p_g(x)}$, and the game is
then equivalent to minimising the Jensen–Shannon divergence between $p_{data}$ and $p_g$; the global optimum is
$p_g = p_{data}$ with $D^* = \tfrac12$.

**In practice** (and in this notebook) we use binary cross-entropy on logits $\ell$, with $\sigma$ the sigmoid:

$$
\mathcal{L}_D = -\big[\,y\log\sigma(\ell_{real}) + (1-y)\cdot 0\,\big] - \log\big(1-\sigma(\ell_{fake})\big), \qquad y = 0.9 \text{ (one-sided label smoothing)}
$$

$$
\mathcal{L}_G = -\log \sigma\big(D(G(z))\big) \quad \text{(non-saturating generator loss)}
$$

The non-saturating loss gives the generator strong gradients early in training, when $D$ easily rejects its samples
(the original $\log(1-D(G(z)))$ would saturate and produce near-zero gradients).
""")

md(r"""
## 6. What is DCGAN?

**DCGAN** (Radford, Metz & Chintala, 2016) is a set of architectural rules that made GANs train reliably on images:

* Replace pooling with **strided convolutions** (D) and **transposed convolutions** (G) so the networks learn
  their own down/up-sampling.
* Use **BatchNorm** in both networks (except the G output layer and the D input layer).
* No fully connected hidden layers.
* **ReLU** in the generator, **Tanh** on its output; **LeakyReLU (0.2)** in the discriminator.
* Adam with **lr = 0.0002, β₁ = 0.5**; weights initialised from $\mathcal{N}(0, 0.02)$.

PixelForge follows these rules exactly at 64×64 resolution:

```
G:  z(100×1×1) → 512×4×4 → 256×8×8 → 128×16×16 → 64×32×32 → 3×64×64 (Tanh)
D:  3×64×64 → 64×32×32 → 128×16×16 → 256×8×8 → 512×4×4 → 1 logit
```
""")

# =========================================================================== setup
md(r"""
## ⚙️ Setup — configuration (edit this cell only)

All tunable settings live here. Defaults are the full-training configuration for a Colab T4 GPU.
""")
code(r"""
# ======================= PIXELFORGE CONFIGURATION =======================
EPOCHS            = 100     # 100 = recommended (prompt model); 50 = quicker
BATCH_SIZE        = 64      # 64 on a Colab GPU; use 16-32 on Mac MPS
LEARNING_RATE     = 2e-4    # Adam learning rate for both networks
LATENT_DIM        = 100     # size of the noise vector z
NUM_GENERATED     = 16      # characters to generate at the end (4, 9 or 16 make square grids)
CHECKPOINT_EVERY  = 10      # save a numbered checkpoint every N epochs (latest.pt is saved every epoch)
SAMPLE_EVERY      = 5       # save/display a fixed-noise grid every N epochs
LABEL_SMOOTHING   = 0.1     # one-sided: real label = 0.9 (0 disables)
SEED              = 42
VIEWS             = (0, 1, 2, 3)   # 0=back 1=left 2=front 3=right; (2,) = front-facing only (912 images)

CONDITIONAL       = True    # prompt control: G(z, y) with y = (view, hair colour, outfit colour)
D_COND            = "concat" # how D sees labels: "concat" (works best here) or "projection"
EMA_DECAY         = 0.998   # moving average of generator weights for smoother samples (0 = off)

SMOKE_TEST        = False   # True = 2 quick epochs on 256 images to verify the pipeline
ALLOW_FULL_TRAINING_WITHOUT_CUDA = False   # without a CUDA GPU the notebook falls back to the smoke test
RESUME            = True    # continue from checkpoints/latest.pt if this session already has one
SAVE_TO_DRIVE     = False   # optional: copy the results ZIP to Google Drive (asks for permission)
INCLUDE_FULL_CHECKPOINT_IN_ZIP = True      # latest.pt (~75 MB, G+D+optimizers) for resuming later
# =========================================================================
""")

md(r"""
### Environment: install only missing packages and create the project folder

Colab already ships PyTorch with CUDA, NumPy, Pillow, Matplotlib, pandas and tqdm, so normally
nothing is installed. The project modules are written into `./pixelforge/` with `%%writefile`, so this notebook
needs **no files from your computer**.
""")
code(r"""
import importlib, importlib.util, os, subprocess, sys
from pathlib import Path

required = {"torch": "torch", "numpy": "numpy", "PIL": "pillow",
            "matplotlib": "matplotlib", "pandas": "pandas", "tqdm": "tqdm"}
missing = [pip for mod, pip in required.items() if importlib.util.find_spec(mod) is None]
if missing:
    print("Installing missing packages:", missing)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
else:
    print("All required packages are already installed.")

IN_COLAB = "google.colab" in sys.modules or importlib.util.find_spec("google.colab") is not None
PROJECT_DIR = Path("pixelforge").resolve()
PROJECT_DIR.mkdir(exist_ok=True)
os.chdir(PROJECT_DIR)
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
print("Working directory:", Path.cwd(), "| Colab:", IN_COLAB)


def fresh(name):
    # import (or re-import after %%writefile) a project module
    if name in sys.modules:
        return importlib.reload(sys.modules[name])
    return importlib.import_module(name)
""")

md("Project modules: configuration/device selection and shared image utilities.")
writefile("config.py")
writefile("utils.py")

# =========================================================================== 7 dataset
md(r"""
## 7. Dataset exploration — TinyHero

**Source:** [AgaMiko/pixel_character_generator](https://github.com/AgaMiko/pixel_character_generator) (`data.zip`, 7.3 MB).
Characters were assembled from the **Universal LPC Spritesheet**.
**License:** dual **CC-BY-SA 3.0** and **GNU GPL 3.0** (LPC art rules) — attribution required, share-alike.

What the documentation says: 912 characters × 4 directions = 3,648 images at 64×64.
We **verify** this below instead of assuming it. Verified layout of the archive:

| folder | view |
|---|---|
| `data/0/` | back |
| `data/1/` | left |
| `data/2/` | front |
| `data/3/` | right |

The dataset module below downloads, safely extracts and validates the archive.
""")
writefile("dataset.py")
code(r"""
import json
import numpy as np
from IPython.display import Image as IPImage, display

config = fresh("config"); utils = fresh("utils"); dataset = fresh("dataset")
root = dataset.download_tinyhero(config.PROJECT_ROOT / "data")
stats = dataset.validate_dataset(root)
print(json.dumps(stats, indent=2))
assert stats["total_images"] > 0, "Dataset extraction failed"
if not stats["matches_documentation"]:
    print("WARNING: dataset differs from the documented 3,648 64x64 images - see stats above.")
""")
code(r"""
full_ds = dataset.TinyHeroDataset(root)
pix = dataset.pixel_statistics(full_ds)
stats["pixels"] = pix
Path("outputs/plots").mkdir(parents=True, exist_ok=True)
Path("outputs/dataset_stats.json").write_text(json.dumps(stats, indent=2))
print(json.dumps(pix, indent=2))

dataset.save_sample_grid(full_ds, "outputs/plots/dataset_samples.png", n=25, scale=3)
dataset.save_view_grid(root, "outputs/plots/dataset_views.png")
print("25 random REAL training images:")
display(IPImage("outputs/plots/dataset_samples.png"))
print("Same character from 4 directions (columns: back, left, front, right):")
display(IPImage("outputs/plots/dataset_views.png"))
""")

# =========================================================================== 8 preprocessing
md(r"""
## 8. Preprocessing

1. **RGBA → RGB.** Every sprite is RGBA with binary alpha (~76 % transparent). We alpha-composite onto an
   explicit solid background, **black** by default. White keeps outlines more visible, but in our experiments
   a white background made this DCGAN **mode-collapse** (all latent vectors → one sprite by epoch 5), while black
   trained stably. This matches the original TinyHero notebooks. Change it with `Config.background`.
2. **Normalise to [-1, 1]:** $x' = x/127.5 - 1$, matching the generator's Tanh output range.
3. **In-memory dataset:** all 3,648 sprites are decoded once into a uint8 tensor (≈45 MB), so the DataLoader needs no workers.
4. **All four views** are used for training to maximise data (front-only would leave just 912 images).
""")
code(r"""
import matplotlib.pyplot as plt
from PIL import Image

sample_path = sorted((root / "2").glob("*.png"))[5]
raw = Image.open(sample_path)
fig, ax = plt.subplots(1, 3, figsize=(9, 3.2))
for a, (im, t) in zip(ax, [(raw, f"original {raw.mode}"),
                           (dataset.load_rgb(sample_path, (255, 255, 255)), "on white (collapsed GAN)"),
                           (dataset.load_rgb(sample_path, cfg_bg := tuple(config.Config().background)),
                            f"on {cfg_bg} (used)")]):
    a.imshow(im, interpolation="nearest"); a.set_title(t); a.axis("off")
plt.show()

loader = dataset.make_dataloader(full_ds, batch_size=BATCH_SIZE)
xb, yb = next(iter(loader))
print("batch:", tuple(xb.shape), xb.dtype, "| range [%.2f, %.2f]" % (xb.min(), xb.max()),
      "| views in batch:", np.bincount(yb.numpy(), minlength=4))
""")

md(r"""
### 8b. Automatic attribute labels (what makes prompts possible)

TinyHero has **no text descriptions**, so we derive labels from the pixels:

| attribute | source | classes |
|---|---|---|
| view | folder name (exact) | back, left, front, right |
| hair / headwear colour | majority colour of rows 9–17 (top of head) | black, brown, blonde, red, white, grey, green, blue, purple |
| outfit colour | majority colour of rows 33–43 (torso) | black, brown, yellow, red, white, grey, green, blue, purple |

Each opaque pixel is classified by HSV rules (dark outline pixels ignored) and the majority wins.
**Limitations:** labels are approximate. A grey helmet counts as grey "hair", and a shirtless character's
"outfit" is their skin colour. The keyword parser maps a prompt to these labels. Words it doesn't know
(e.g. "wizard", "sword") are ignored.
""")
writefile("attributes.py")
code(r"""
attributes = fresh("attributes")
labels = attributes.dataset_labels(root, dataset.list_images(root))
for col, (name, vocab) in enumerate(attributes.ATTRIBUTES.items()):
    counts = np.bincount(labels[:, col], minlength=len(vocab))
    print(f"{name:7s}", ", ".join(f"{v}={c}" for v, c in zip(vocab, counts)))
# visual check: 8 real sprites per hair colour (rows)
items = dataset.list_images(root)
rng = np.random.default_rng(0); rows = []
for h in range(len(attributes.HAIR)):
    idx = np.where(labels[:, 1] == h)[0]
    rows += [dataset.load_rgb(items[i][0]) for i in rng.choice(idx, 8, replace=len(idx) < 8)]
print("Rows = hair labels:", ", ".join(attributes.HAIR))
display(utils.make_grid(rows, nrow=8, scale=2))
for p in ["a front-facing hero with green hair and a red tunic", "knight in armour seen from behind",
          "blonde girl wearing a blue dress, facing left"]:
    print(f"{p!r:55} -> {attributes.parse_prompt(p)[1]}")
""")

# =========================================================================== 9 GPU
md(r"""
## 9. GPU setup

Device priority: **CUDA** (Colab T4) → **MPS** (Apple Silicon) → **CPU**.
If no CUDA GPU is available, full training would take far too long, so the notebook **automatically switches to
smoke-test mode** (2 epochs, 256 images). This verifies the code path but does **not** produce a good model.
To fix it in Colab: `Runtime → Change runtime type → T4 GPU`, then *Run all* again.
""")
code(r"""
import torch
device = config.get_device()
print("PyTorch", torch.__version__, "| device:", config.describe_device(device))
if device.type == "cuda":
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv"])

if not SMOKE_TEST and device.type != "cuda" and not ALLOW_FULL_TRAINING_WITHOUT_CUDA:
    print("\n" + "!" * 78)
    print("No CUDA GPU detected -> switching to SMOKE-TEST mode (2 epochs on 256 images).")
    print("The pipeline will be verified, but the resulting model is NOT fully trained.")
    print("In Colab: Runtime > Change runtime type > T4 GPU, then Run all.")
    print("!" * 78)
    SMOKE_TEST = True

if SMOKE_TEST:
    cfg = config.smoke_test_config(nz=LATENT_DIM, lr=LEARNING_RATE, seed=SEED, views=tuple(VIEWS),
                                   label_smoothing=LABEL_SMOOTHING, conditional=CONDITIONAL, d_cond=D_COND,
                                   ema_decay=EMA_DECAY)
else:
    cfg = config.Config(epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE, nz=LATENT_DIM,
                        checkpoint_every=CHECKPOINT_EVERY, sample_every=SAMPLE_EVERY, seed=SEED,
                        views=tuple(VIEWS), label_smoothing=LABEL_SMOOTHING, conditional=CONDITIONAL,
                        d_cond=D_COND, ema_decay=EMA_DECAY)
config.seed_everything(cfg.seed)
MODE = "SMOKE TEST" if SMOKE_TEST else "FULL TRAINING"
print(f"\nMode: {MODE} | epochs={cfg.epochs} batch={cfg.batch_size} subset={cfg.subset}")
print("checkpoints ->", cfg.checkpoint_dir, "\noutputs     ->", cfg.output_dir)
""")

# =========================================================================== 10-12 models
md(r"""
## 10. Generator implementation

The generator upsamples a 100-d noise vector into a 64×64 RGB sprite. A `ConvTranspose2d` with kernel 4,
stride 2, padding 1 exactly doubles the spatial size: $H_{out} = (H_{in}-1)\cdot 2 - 2 + 4 = 2H_{in}$.
The first layer (stride 1, padding 0) turns the 1×1 input into a 4×4 feature map.
`weights_init` applies the DCGAN initialisation; `sample_latent` draws $z$ on the CPU with a seeded
generator so the **same seed gives the same characters on CUDA, MPS and CPU**.

The **conditional generator** (`ConditionalGenerator`, next cell) reuses exactly this stack. Each label
(view, hair, outfit) is turned into a learned 16-d embedding and concatenated with $z$, so the first layer
sees $100 + 3 \times 16 = 148$ inputs.
""")
writefile("models.py", models_a + "# ---- notebook-split ----\n")
md(r"""
## 11. Discriminator implementation

The discriminator mirrors the generator with strided `Conv2d` layers and **LeakyReLU(0.2)** (so gradients still
flow for negative activations). There is **no BatchNorm on the first layer** and **no sigmoid at the end**:
it returns a raw logit for `BCEWithLogitsLoss`.

**Conditional discriminator.** `ConcatDiscriminator` appends the labels to the image as 22 one-hot
"label maps" (4 + 9 + 9 channels, constant over the image). D then judges *"is this a real sprite with these
attributes?"*, so the generator is rewarded only when it follows the labels. A projection discriminator
(Miyato & Koyama, 2018) is also implemented. In our experiments it overpowered the generator on this small
dataset (samples collapsed by epoch 6), so concatenation is the default. This cell also adds the latent
interpolation helpers (`lerp`, `slerp`).
""")
writefile("models.py", models_b, append=True)

md("## 12. Model architecture\n\nBuild both networks, apply DCGAN initialisation and trace the tensor shapes layer by layer.")
code(r"""
models = fresh("models")
G, D = models.build_models(cfg.nz, cfg.ngf, cfg.ndf, device,
                            n_classes=tuple(cfg.n_classes) if cfg.conditional else None, d_cond=cfg.d_cond)
print(G); print(D)

def trace(seq, x):
    print(f"{'input':<18} {tuple(x.shape[1:])}")
    for m in seq:
        x = m(x)
        if isinstance(m, (torch.nn.ConvTranspose2d, torch.nn.Conv2d)):
            print(f"{m.__class__.__name__:<18} {tuple(x.shape[1:])}")
    return x

with torch.no_grad():
    g_in = G.net[0].in_channels  # nz (+ label embeddings if conditional)
    print(f"\nGenerator shape trace (input = z{' + label embeddings' if cfg.conditional else ''})")
    fake = trace(G.net, torch.zeros(1, g_in, 1, 1, device=device))
    d_body = D.body.net if hasattr(D, "body") else (D.features if hasattr(D, "features") else D.net)
    d_in = d_body[0].in_channels
    print(f"\nDiscriminator shape trace (input = image{' + label maps' if d_in > 3 else ''})")
    trace(d_body, torch.cat([fake, torch.zeros(1, d_in - 3, 64, 64, device=device)], 1))
pG, pD = models.count_parameters(G), models.count_parameters(D)
print(f"\nGenerator parameters:     {pG:,}  (~{pG*4/1e6:.1f} MB fp32)")
print(f"Discriminator parameters: {pD:,}  (~{pD*4/1e6:.1f} MB fp32)")
""")

# =========================================================================== 13 losses
md(r"""
## 13. Loss functions

`BCEWithLogitsLoss` combines the sigmoid and binary cross-entropy in one numerically stable operation
(log-sum-exp trick), which is why the discriminator outputs raw logits.

* **Discriminator:** real images → target **0.9** (one-sided label smoothing, Salimans et al. 2016 — stops D
  from becoming over-confident), generated images → target **0**.
* **Generator:** generated images → target **1** (non-saturating loss: "make D call my images real").
""")
code(r"""
crit = torch.nn.BCEWithLogitsLoss()
logits = torch.tensor([3.0, -2.0, 0.0])
targets = torch.tensor([0.9, 0.0, 1.0])
manual = -(targets * torch.log(torch.sigmoid(logits)) + (1 - targets) * torch.log(1 - torch.sigmoid(logits))).mean()
print(f"BCEWithLogitsLoss = {crit(logits, targets):.6f} | manual sigmoid+BCE = {manual:.6f}")
print("At the theoretical equilibrium D outputs 0.5 everywhere: loss_D = 2*ln2 = %.3f, loss_G = ln2 = %.3f"
      % (2 * np.log(2), np.log(2)))
""")

# =========================================================================== 14 training
md(r"""
## 14. Training loop

For every batch:

1. Load real sprites, sample $z\sim\mathcal{N}(0,I)$ and generate fakes $G(z)$.
2. **Update D** on the real batch and on `fake.detach()` — detaching stops the D loss from changing G.
3. **Update G** by passing the (non-detached) fakes through D with target "real".
4. Log both losses and $D(x)$, $D(G(z))$; abort with a clear error if a loss becomes NaN/Inf.

Every epoch, `latest.pt` (G, D, both Adam states, epoch, config, history, fixed noise, RNG) is saved, so a Colab
disconnect loses at most one epoch — re-running the notebook with `RESUME = True` continues where it stopped
**without resetting the optimizer state**. Numbered checkpoints are saved every `CHECKPOINT_EVERY` epochs, and a
grid of 64 fixed latent vectors is rendered every `SAMPLE_EVERY` epochs. Checkpoints contain only
`state_dict`s and plain Python types, so they load with `weights_only=True` on any device.
""")
writefile("train.py")
code(r"""
train = fresh("train")

def show_progress(epoch, G, fixed_noise, history):
    if epoch == 1 or epoch % cfg.sample_every == 0 or epoch == cfg.epochs:
        p = Path(cfg.output_dir) / "training_progress" / f"epoch_{epoch:03d}.png"
        if p.exists():
            print(f"Fixed-noise samples after epoch {epoch}:")
            display(IPImage(str(p), width=520))

result = train.train(cfg, resume="auto" if RESUME else None, on_epoch_end=show_progress)
print(f"\n{MODE} finished: {result['epochs_completed']} epochs on {result['device']}")
""")

# =========================================================================== 15 losses plot
md(r"""
## 15. Generator and discriminator losses

`evaluate.py` holds all plotting and evaluation code (used again in Section 19).

**Why the loss alone cannot tell us sample quality:** the two losses measure a moving two-player game, not
distance to the data distribution. When G improves, D's task gets harder and its loss rises; when D improves,
G's loss rises — even if images are getting better. Losses can look stable while the generator mode-collapses.
We therefore always inspect samples, progression grids and diversity statistics as well.
""")
writefile("evaluate.py")
code(r"""
evaluate = fresh("evaluate")
out = Path(cfg.output_dir)
loss_png = evaluate.plot_losses(out / "training_history.csv", out / "training_iterations.csv", out / "plots" / "losses.png")
display(IPImage(str(loss_png)))
import pandas as pd
hist_df = pd.read_csv(out / "training_history.csv")
display(hist_df.tail(10))
""")

# =========================================================================== 16 progress
md(r"""
## 16. Training progress visualizations

The same 64 latent vectors are decoded after different numbers of epochs. Look for: blobs → silhouettes →
heads/bodies/legs → coloured clothing, hair and equipment.
""")
code(r"""
prog_png = evaluate.progression_montage(out / "training_progress", out / "plots" / "progression.png")
if prog_png:
    display(IPImage(str(prog_png)))
final_grid = sorted((out / "training_progress").glob("epoch_*.png"))[-1]
print("Final fixed-noise grid:", final_grid.name)
display(IPImage(str(final_grid)))
""")

# =========================================================================== 17 generation
md(r"""
## 17. New character generation

Load the inference-only `generator_final.pt` (CPU first via `map_location="cpu"`, then moved to the device),
sample new latent vectors with a fixed seed and decode them. Outputs are converted from $[-1,1]$ to $[0,255]$ and
saved as genuine **RGB** PNGs; previews use nearest-neighbour upscaling to keep pixel edges crisp.
Transparent copies (optional) are made with a documented flood-fill heuristic, **not** learned by the model.
""")
writefile("generate.py")
code(r"""
generate = fresh("generate")
G_inf, cfg_inf, meta = generate.load_generator(Path(cfg.checkpoint_dir) / "generator_final.pt", device)
print("Loaded generator from epoch", meta["epoch"])
GEN_SEED = 2024
chars = generate.generate(G_inf, NUM_GENERATED, seed=GEN_SEED, device=device)
gen_dir = out / "generated"
paths = generate.save_images(chars, gen_dir / f"seed{GEN_SEED}", transparent=True, background=cfg.background)
grid = utils.make_grid(chars, scale=4)
grid.save(gen_dir / f"grid_{NUM_GENERATED}_seed{GEN_SEED}.png")
print(f"Saved {len(paths)} sprites to {gen_dir / f'seed{GEN_SEED}'}")
display(grid)
""")

md(r"""
### 17b. Prompt → character

`parse_prompt` turns the description into labels; unspecified attributes are random. Each row below is one
prompt, decoded with 8 different latent vectors $z$ (same labels, different characters). **Edit `PROMPTS` to try your own.**
""")
code(r"""
PROMPTS = ["a front-facing hero with green hair and a red tunic",
           "knight in armour seen from behind",
           "blonde adventurer in a blue outfit, facing left",
           "purple-haired mage in a white robe",
           "red-haired rogue in a brown leather vest, right side"]
if getattr(G_inf, "conditional", False):
    prompt_dir = out / "generated" / "prompts"; prompt_dir.mkdir(parents=True, exist_ok=True)
    for k, p in enumerate(PROMPTS):
        labels, notes = attributes.parse_prompt(p)
        imgs = generate.generate(G_inf, 8, seed=100 + k, device=device, labels=labels)
        row = utils.make_grid(imgs, nrow=8, scale=3)
        row.save(prompt_dir / f"prompt_{k}.png")
        print(f"{p!r}  ->  {', '.join(notes)}"); display(row)
else:
    print("This generator is unconditional (CONDITIONAL = False): prompts cannot be used.")
""")

# =========================================================================== 18 interpolation
md(r"""
## 18. Latent space interpolation

Pick two latent vectors $z_1, z_2$ and decode points along the line between them:
$z(t) = (1-t)\,z_1 + t\,z_2$. Because $G$ is a continuous function, nearby latent vectors map to related images,
so a well-trained generator shows a gradual morph (body shape, colours, equipment) rather than random jumps.
We also show **slerp** (spherical interpolation), which keeps $\|z\|$ close to the typical norm of a 100-d Gaussian.
Not every transition is semantically meaningful — some intermediate frames can look less coherent.
""")
code(r"""
interp_dir = out / "interpolation"
interp_dir.mkdir(parents=True, exist_ok=True)
for a, b in [(7, 2024), (1, 99), (42, 314)]:
    frames = generate.interpolate(G_inf, a, b, steps=10, mode="linear", device=device)
    strip = utils.make_grid(frames, nrow=len(frames), scale=3)
    strip.save(interp_dir / f"interpolation_seed{a}_to_seed{b}_linear.png")
    utils.save_gif([utils.upscale(f, 4) for f in frames], interp_dir / f"interpolation_seed{a}_to_seed{b}.gif")
    print(f"seed {a} -> seed {b} (linear)"); display(strip)
frames = generate.interpolate(G_inf, 7, 2024, steps=10, mode="slerp", device=device)
print("seed 7 -> seed 2024 (slerp)"); display(utils.make_grid(frames, nrow=10, scale=3))
display(IPImage(str(interp_dir / "interpolation_seed7_to_seed2024.gif")))
""")

# =========================================================================== 19 evaluation
md(r"""
## 19. Evaluation

Five experiments, all computed from this run (nothing is hard-coded):

1. **Training loss** — Section 15.
2. **Visual quality progression** — Section 16.
3. **Real vs generated** — side-by-side, clearly labelled grids.
4. **Latent interpolation** — strips for 3 seed pairs, linear vs slerp, plus frame-to-frame smoothness.
5. **Diversity & memorisation** — 512 generated vs 512 real images:
   * mean pairwise pixel RMSE within each set (diversity ratio ≈ 1 is good, ≪ 1 suggests mode collapse);
   * nearest-neighbour distance from each generated image to the **whole training set**, compared with the
     distance from real images to their nearest *other* training image (much smaller → memorisation);
   * count of near-copies (RMSE < 0.02) and colour-histogram distance.

**FID is intentionally not computed**: it requires an ImageNet-pretrained Inception network and many thousands
of samples, and is unreliable for 64×64 pixel art with only 3.6k training images.
""")
code(r"""
metrics = evaluate.evaluate(Path(cfg.checkpoint_dir) / "generator_final.pt", cfg, num_samples=512,
                            output_dir=out, device=device)
for name in ["controllability", "prompt_control_grid", "real_vs_generated", "interpolation_experiment",
             "nearest_neighbors", "diversity"]:
    if (out / "plots" / f"{name}.png").exists():
        print(name); display(IPImage(str(out / "plots" / f"{name}.png")))
""")

# =========================================================================== 20 results
md(r"""
## 20. Results

The table below is filled in from `evaluation_metrics.json` produced by the cell above — for **this** run.
If the notebook ran in smoke-test mode, these numbers describe a barely-trained model and must not be reported
as final results.
""")
code(r"""
print(f"Mode: {MODE} | epochs trained: {metrics['checkpoint_epoch']} | device: {metrics['device']}")
rows = {
    "Final loss_D / loss_G": f"{metrics.get('final_epoch_losses', {}).get('loss_d', float('nan')):.3f} / "
                             f"{metrics.get('final_epoch_losses', {}).get('loss_g', float('nan')):.3f}",
    "Mean pairwise RMSE generated": f"{metrics['pairwise_rmse_generated']['mean']:.4f}",
    "Mean pairwise RMSE real": f"{metrics['pairwise_rmse_real']['mean']:.4f}",
    "Diversity ratio (gen/real)": f"{metrics['diversity_ratio_generated_over_real']:.3f}",
    "Median NN RMSE generated->train": f"{metrics['nearest_neighbor_rmse_generated_to_train']['median']:.4f}",
    "Median NN RMSE real->other train": f"{metrics['nearest_neighbor_rmse_real_to_other_train']['median']:.4f}",
    "Near-copies (NN RMSE < 0.02)": metrics["possible_copies(nn_rmse<0.02)"],
    "Colour histogram L1 distance": f"{metrics['color_histogram_L1_distance']:.4f}",
    "Interpolation: mean frame-to-frame RMSE": f"{metrics['interpolation']['mean_consecutive_frame_rmse']:.4f}",
}
for name, c in metrics.get("controllability", {}).items():
    rows[f"{name.title()} control: requested colour obtained"] = f"{c['accuracy']:.1%}"
    rows[f"{name.title()} control: if requests ignored (baseline)"] = f"{c['shuffled_baseline']:.1%}"
display(pd.DataFrame(rows.items(), columns=["metric", "value"]))
print("\nInterpretation (rules applied to the measured numbers):")
for line in metrics["interpretation"]:
    print(" -", line)
""")

# =========================================================================== 21-23
md(r"""
## 21. Limitations

* **Small dataset** (3,648 images, 912 unique characters) → limited variety; four views of each character are
  strongly correlated.
* **Unconditional model** — you cannot request a direction, class or item. A conditional GAN would add that.
* **64×64 resolution** and DCGAN capacity: details such as faces and weapons are often blurry or incoherent;
  transposed convolutions can create checkerboard artifacts.
* **GAN instability** — results depend on seeds and hyperparameters; mode collapse is possible.
* **Evaluation metrics are simple** pixel-space proxies; they do not measure perceptual quality.
* **Transparency** is not learned: the optional background removal is a heuristic flood fill.
""")
md(r"""
## 22. Conclusion

We trained a conditional DCGAN from scratch on TinyHero and used it to create new pixel-art RPG characters from
short text descriptions.
The notebook verifies the dataset, implements the standard DCGAN architecture and training procedure with
logging, checkpoints and resume support, and evaluates the model with loss curves, fixed-noise progression,
real-vs-generated grids, latent interpolation, and diversity / nearest-neighbour memorisation checks.
The exported `generator_final.pt` runs on CUDA, Apple MPS or CPU and powers the PixelForge Streamlit studio.
""")
md(r"""
## 23. Viva questions and answers (short version — see `VIVA_GUIDE.md` for 25+ detailed answers)

**Q: How does a text prompt become an image without a text model?** The prompt is parsed by keyword rules into
the three labels the GAN was trained on (view, hair colour, outfit colour). The conditional generator $G(z, y)$
draws a character with those labels; $z$ supplies the variation. Words outside that vocabulary are ignored.

**Q: Where do the labels come from?** View comes from the dataset folders. Hair and outfit colour are measured
automatically from the pixels (majority HSV colour of fixed head and torso regions), so they're approximate.

**Q: How do you know the prompt actually controls the output?** Controllability experiment: request each
colour, label the generated sprites with the same automatic labeller, and compare the hit rate with a baseline
where the requests are shuffled (a model that ignores them).

**Q: What does the generator learn?** A mapping from a simple distribution $\mathcal{N}(0,I)$ to the distribution
of sprites, so that samples it produces are indistinguishable (to D) from real ones.

**Q: Why detach the fake images in the D step?** So the discriminator loss does not compute gradients for the
generator — each network is updated only by its own objective.

**Q: Why no sigmoid in D?** `BCEWithLogitsLoss` applies the sigmoid internally in a numerically stable way.

**Q: Why Tanh + [-1, 1] normalisation?** Tanh outputs lie in [-1, 1]; scaling real data to the same range lets
D compare like with like and keeps inputs zero-centred.

**Q: What is mode collapse?** G produces only a few kinds of output that fool D, ignoring the rest of the data
distribution. We check for it with the diversity ratio and by inspecting samples.

**Q: How do you know the model is not copying training images?** Nearest-neighbour search over all 3,648
training images: generated samples are compared with their closest real image, and those distances are compared
with real-to-real nearest-neighbour distances.

**Q: Why use latent interpolation?** Smooth transitions indicate a continuous, structured latent space rather than
a lookup table of memorised sprites.
""")

# =========================================================================== export
md(r"""
## 📦 Export results

Packages the checkpoints, history, plots, progression grids, generated sprites and interpolations into
`pixelforge_results.zip`, mirroring the project layout. On your Mac, unzip it **inside the project folder**:

```bash
cd pixelforge
unzip ~/Downloads/pixelforge_results.zip -d .
streamlit run app.py
```
""")
code(r"""
import zipfile
zip_path = PROJECT_DIR / "pixelforge_results.zip"
ck = Path(cfg.checkpoint_dir)
files = [ck / "generator_final.pt"] + ([ck / "latest.pt"] if INCLUDE_FULL_CHECKPOINT_IN_ZIP else [])
files += [p for p in out.rglob("*") if p.is_file()]
files.append(Path("outputs/dataset_stats.json"))
files += [p for p in Path("outputs/plots").glob("dataset_*.png")]
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in dict.fromkeys(files):
        if f.exists():
            zf.write(f, Path(f).resolve().relative_to(PROJECT_DIR))
print(f"{zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB), {len(zipfile.ZipFile(zip_path).namelist())} files")

if SAVE_TO_DRIVE and IN_COLAB:
    from google.colab import drive
    import shutil
    drive.mount("/content/drive")
    dst = Path("/content/drive/MyDrive/PixelForge")
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy(zip_path, dst / zip_path.name)
    print("Copied to", dst / zip_path.name)

if IN_COLAB:
    from google.colab import files as colab_files
    colab_files.download(str(zip_path))
else:
    print("Not running in Colab - the ZIP is at", zip_path)
""")

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4", "toc_visible": True, "name": "PixelForge_Complete.ipynb"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}
out_path = ROOT / "PixelForge_Complete.ipynb"
out_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(f"Wrote {out_path} ({len(cells)} cells)")
