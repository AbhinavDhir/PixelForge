# PixelForge — Viva Preparation Guide

## Part A — 5-minute presentation script

**0:00 – 0:40 · The problem.**
"Games need hundreds of character sprites, and drawing each one by hand is slow. PixelForge lets you *describe*
a character, for example 'a front-facing hero with green hair and a red tunic', and a GAN I trained from scratch
draws new ones that match. No pretrained models, no text-to-image APIs: a conditional DCGAN in PyTorch."

**0:40 – 1:30 · The data and the labels.**
"TinyHero has 912 characters, each drawn from 4 directions: 3,648 sprites at 64×64. I checked every file
instead of trusting the documentation. The dataset has **no captions**, so I made labels automatically. The
direction comes from the folder. Hair colour is the majority colour of the top of the head, and outfit colour is
the majority colour of the torso, classified with HSV rules. I checked the labels visually per colour. A keyword
parser turns a prompt into these three labels. It's honest about its limits: it knows directions and nine
colours each for hair and outfit. Words like 'wizard' are ignored because the data has no such label."

**1:30 – 2:30 · The model.**
"A GAN has two networks. The **Generator** takes 100 random numbers *plus the labels* (each turned into a small
learned embedding) and grows a 4×4 feature map into a 64×64 image with transposed convolutions. The
**Discriminator** sees the image *plus the labels*, as extra input channels, and decides 'is this a real sprite
with these attributes?'. So the generator only wins if its character looks real **and** matches the request.
The random numbers give variety: same prompt, different characters."

**2:30 – 3:20 · Training, and what went wrong.**
"Adam with learning rate 0.0002 and β₁ = 0.5, binary cross-entropy on logits, label smoothing, and an
exponential moving average of the generator weights. Two problems came up, and I fixed both with controlled
experiments. On a white background the GAN **mode-collapsed**: every input gave the same character. Switching to
black fixed it. A **projection discriminator** for the labels was too strong and also collapsed. Feeding labels
as input channels trained stably. Training logs a diversity measure every epoch and warns if collapse starts."

**3:20 – 4:20 · Demo** (Streamlit).
1. *Prompt Studio*: type a description → see the parsed attributes → 8 new characters; change one word, see
   the change.
2. *Latent Space Explorer*: keep the attributes fixed and morph between two random characters.
3. *Training Dashboard*: losses, the diversity monitor, and the progress grid (rows = hair colours).
4. *Evaluation*: controllability scores, real vs generated, nearest neighbours showing nothing is copied.

**4:20 – 5:00 · Evaluation and limits.**
"I measure **controllability**: request each colour, label the outputs with the same labeller, and compare the
hit rate with a baseline where the requests are shuffled. I also measure diversity against real data, and use
nearest-neighbour search against all 3,648 training sprites to rule out memorisation. Limits: labels are
approximate, the vocabulary is small, and it's 64×64. Next steps: richer labels such as weapons and classes from the
LPC generator metadata, a learned text encoder, and a stronger architecture."

---

## Part B — Pipeline in plain language

```
TinyHero zip ──► validate (3,648 × 64×64 RGBA) ──► paste on black, scale to [-1,1] ──► batches of 64
           └──► auto-label: view (folder), hair colour, outfit colour ──► labels y ─────────┤
 "green hair, red tunic" ──► keyword parser ──► y                                           ▼
     random z (100 numbers) + y ──► GENERATOR ──► fake sprite ──┐  (D also sees y)
                                   ▲                        ├──► DISCRIMINATOR ──► real/fake score
                                   │                        │          │
                                   └──── gradients ◄────────┘◄── loss ◄┘
every epoch: log losses · save latest.pt · every 5: sample grid · every 10: numbered checkpoint
after training: generator_final.pt ──► generate.py / app.py / evaluate.py
```

---

## Part C — Technical questions and answers

### 1. What is generative AI?
Models that learn the probability distribution of data, $p(x)$, and can **sample new data** from it
(images, text, audio), as opposed to discriminative models that learn $p(y \mid x)$ to predict labels.
Families include GANs, VAEs, autoregressive models (GPT), normalising flows and diffusion models.

### 2. What is a GAN?
A Generative Adversarial Network (Goodfellow et al., 2014) is two networks trained in a minimax game. A
**generator** $G$ maps noise $z$ to samples. A **discriminator** $D$ classifies samples as real or generated.
$D$ maximises its classification accuracy and $G$ minimises it. At the theoretical equilibrium
$p_g = p_{data}$ and $D(x) = 0.5$ everywhere.

### 3. How is a GAN different from a VAE?
| | GAN | VAE |
|---|---|---|
| Training signal | adversarial (learned discriminator) | reconstruction + KL divergence (ELBO) |
| Encoder | none (generator only) | yes: $q(z \mid x)$ |
| Likelihood | implicit, no density | lower bound on $\log p(x)$ |
| Samples | sharp | often blurry (pixel-wise loss averages) |
| Stability | harder (mode collapse, oscillation) | stable |
| Latent inference | not built in | encode a real image to $z$ |

### 4. What is DCGAN?
Radford, Metz & Chintala (2016): architectural guidelines that make convolutional GANs train stably. Strided
convolutions instead of pooling, transposed convolutions for upsampling, BatchNorm in both networks (except
the G output and D input), no fully connected hidden layers, ReLU in G with Tanh output, LeakyReLU in D,
Adam(lr 2e-4, β₁ = 0.5), and N(0, 0.02) weight initialisation.

### 5. What does the Generator learn?
A deterministic function $G_\theta: \mathbb{R}^{100} \to \mathbb{R}^{3\times64\times64}$ that pushes the simple
Gaussian distribution onto the distribution of sprites. Early layers decide global structure (pose,
silhouette, palette) and later layers add local detail (outlines, shading, clothing pixels). It never sees
real images, only gradients through $D$.

### 6. What does the Discriminator learn?
A binary classifier that outputs a logit for "real" vs "generated". It learns features that separate the two,
such as broken outlines, wrong colours or noise. These become the training signal for G. In theory the optimal
$D^*(x) = p_{data}(x)/(p_{data}(x)+p_g(x))$.

### 7. What is adversarial training?
Alternating optimisation of two opposing objectives. In PixelForge, each batch has one D step (real → 0.9,
detached fake → 0) and one G step (fake → 1 through D). Each network's improvement makes the other's task
harder, which drives the generator towards the data distribution.

### 8. Why use random latent vectors?
They make the generator a **sampler**. Different $z$ give different characters, so we can generate unlimited
variety. A Gaussian is easy to sample, smooth and isotropic. The randomness is the source of diversity. A
fixed input would always produce the same image.

### 9. What is latent space?
The 100-dimensional space of $z$. After training, it is organised: nearby points give similar characters,
and directions can loosely correspond to attributes such as hair colour or armour. In PixelForge a seed fixes
$z$, so a seed is effectively a "character ID".

### 10. Why use ConvTranspose2d?
It is a learnable upsampling layer (the gradient of a convolution). With kernel 4, stride 2, padding 1 it
exactly doubles spatial size, $(H-1)\cdot2 - 2 + 4 = 2H$, so 4→8→16→32→64. The network learns *how* to
upsample instead of using fixed interpolation. Downside: overlapping kernels can cause checkerboard
artifacts.

### 11. Why use BatchNorm?
It normalises each channel to zero mean and unit variance over the batch, then rescales with learnable γ, β.
This stabilises and speeds up training, helps gradients flow in deep generators, and reduces the chance that
G collapses to a single output. DCGAN omits it in the G output layer and the D input layer, where it causes
sample oscillation.

### 12. Why use LeakyReLU in the Discriminator?
ReLU outputs zero gradient for negative inputs ("dead" units). The generator's only learning signal is the
gradient flowing *back through D*, so D should pass gradients for all inputs. LeakyReLU(0.2) keeps a slope of
0.2 for negatives. G uses plain ReLU, as recommended by the DCGAN paper.

### 13. Why use Tanh in the Generator?
Tanh bounds outputs to (−1, 1), matching the normalised data range, and it is zero-centred with useful
gradients. A bounded output also stops the generator producing arbitrary pixel values.

### 14. Why normalise images into [-1, 1]?
So real and generated images live in the same range (Tanh output). Otherwise D could separate them trivially
by value range. Zero-centred inputs also make optimisation better conditioned. We invert with
$x = (y+1)/2 \cdot 255$ when saving PNGs.

### 15. What is BCEWithLogitsLoss?
Binary cross-entropy that takes **raw logits** and applies the sigmoid internally:
$\ell(x, y) = -[y \log\sigma(x) + (1-y)\log(1-\sigma(x))]$, computed with the log-sum-exp trick so large logits
don't overflow or give $\log 0$. That is why the discriminator has **no sigmoid layer**. Adding one would apply
the sigmoid twice.

### 16. What is mode collapse?
The generator maps many different $z$ to the same few outputs that currently fool D, ignoring the rest of the
data distribution, for example every character becoming the same bald hero. We detect it with the
**diversity ratio** (mean pairwise distance of generated vs real images; ≪ 1 means collapse) and by looking at
grids. Our 2-epoch smoke model correctly scored 0.06. Remedies include label smoothing, minibatch
discrimination, WGAN-GP or unrolled GANs.

### 17. Why can GAN training be unstable?
* It is a **minimax game**, not a minimisation. Gradient descent on two players can oscillate or circle instead of converging.
* If D becomes too strong, $D(G(z)) \to 0$ and G's gradients vanish (hence the non-saturating loss and label smoothing).
* The target keeps moving: each update changes the other network's objective.
* Mode collapse, and high sensitivity to learning rates, architecture and initialisation.
We mitigate with DCGAN rules, Adam β₁ = 0.5, label smoothing, NaN detection and frequent checkpoints.

### 18. How do we evaluate generated images?
No single number is enough. PixelForge uses: (1) loss curves for training health, (2) fixed-noise
progression grids, (3) labelled real-vs-generated grids, (4) interpolation smoothness, (5) diversity
statistics and nearest-neighbour distances to the training set (memorisation check). Standard metrics are
**FID** and **KID** (distance between Inception feature distributions) and **Inception Score**. We did not use
FID because it needs many thousands of samples and an ImageNet network that suits 64×64 pixel art poorly.
KID would be the better choice for small datasets.

### 19. How does latent interpolation work?
Sample $z_1, z_2$, compute $z(t) = (1-t)z_1 + t z_2$ for $t \in [0, 1]$ (8–12 steps), and decode each with G.
Because G is continuous, nearby latent points give similar images, so we see a morph. **Slerp** (spherical
interpolation) is an alternative. In 100-d, Gaussian samples concentrate near radius $\sqrt{100} = 10$, and
the linear midpoint has a smaller norm, which G saw less in training. Not every intermediate frame is
guaranteed to be a coherent character.

### 20. How would you improve the model?
Conditional GAN on the view label; WGAN-GP or hinge loss with spectral normalisation; upsample + conv instead
of transposed conv (no checkerboard); DiffAugment for the small dataset; EMA of generator weights; an RGBA
output for learned transparency; KID/LPIPS metrics; a StyleGAN-style mapping network for controllable
attributes.

### 21. Why detach the fake images in the discriminator step?
`fake.detach()` cuts the graph, so $\mathcal{L}_D$'s backward pass
doesn't compute or accumulate gradients in G's parameters. This saves compute and stops D's objective from
moving G. A unit test (`test_one_discriminator_step_updates_only_D`) checks G's gradients stay `None`.

### 22. What is the non-saturating generator loss?
Instead of minimising $\log(1 - D(G(z)))$, which has a near-zero gradient when D confidently rejects fakes
early in training, G maximises $\log D(G(z))$, implemented as BCE with target 1. It has the same fixed point
but much stronger gradients early on.

### 23. What is one-sided label smoothing and why only one side?
Real targets are 0.9 instead of 1, which stops D from becoming over-confident and giving G extreme gradients.
Only real labels are smoothed. Smoothing fake labels (e.g. 0.1) would reward D for assigning probability to
fake samples and can stop G from improving (Salimans et al., 2016).

### 24. Why Adam with β₁ = 0.5 instead of 0.9?
Lower momentum reacts faster to the constantly changing adversarial objective. Radford et al. found that 0.9
caused oscillation and instability.

### 25. How do you know the outputs are not just copied training images?
Nearest-neighbour search over **all 3,648 training images** in pixel space. For each generated image we take
the RMSE to its closest training image and compare that distribution with the distance from real images to
their nearest *other* training image. If generated images sat much closer to the training set than real
images sit to each other, or had near-zero distances, that would indicate memorisation. The
`nearest_neighbors.png` figure shows each sample next to its 3 nearest real images.

### 26. Why use all four viewing directions?
More data: 3,648 vs 912 images. GANs on small datasets overfit the discriminator quickly. The model then
generates characters facing different directions without control. A conditional GAN would add that control.

### 27. Why does a checkpoint store optimizer state?
Adam keeps per-parameter first and second moment estimates and a step counter. Resuming without them resets
the effective learning rate and momentum, which causes a jump in training dynamics. PixelForge restores both
optimizers (a test checks the step counter continues). Checkpoints are loaded with `map_location="cpu"` so a
CUDA-trained file opens on a Mac.

### 28. How does seeding make results reproducible across devices?
Latent vectors are sampled on the CPU with a seeded `torch.Generator` and then moved to the device, so seed 7
produces the same $z$ on CUDA, MPS and CPU. Training also seeds Python, NumPy, PyTorch and the DataLoader
shuffling. GPU kernels are not bit-exact across hardware, so weights trained on different devices can differ
slightly.

### 29. Did you encounter mode collapse? How did you diagnose and fix it?
Yes. The first full run used a white background and collapsed: by epoch 5 every latent vector produced the
same sprite. Diagnosis: (1) visual grids of 64 fixed latent vectors were identical; (2) the logs showed
D winning from epoch 1 (D(G(z)) ≈ 0.003); (3) a diversity measure (pixel std across samples) fell to ~6% of
real data; (4) CPU and MPS gave identical outputs and gradients, ruling out a device bug. I then ran
controlled 6-epoch probes. Instance noise, dropout, minibatch std-dev and spectral normalisation did not
fix it on white. Switching to a **black background** did (37% of real diversity vs 9–22%). Likely reason:
76% of each real image is a perfectly uniform white field that a Tanh generator can't reproduce exactly, so D
wins trivially and G retreats to one safe sprite. Training now logs this diversity measure every epoch and
warns automatically when it drops.

### 30. What is a conditional GAN and how is yours conditioned?
A GAN where both networks also receive side information $y$: $G(z, y)$ and $D(x, y)$ (Mirza & Osindero, 2014).
G gets three 16-d learned label embeddings concatenated with $z$. D gets 22 one-hot label maps concatenated to
the image channels. D must then judge whether the image is real *and* consistent with $y$, so G learns to obey $y$.

### 31. Your dataset has no text. How can you generate from a prompt?
Two steps. (1) **Automatic labels**: direction from the dataset folders, hair and outfit colour from the pixels
(majority HSV colour of the head and torso regions, with dark outline pixels ignored). I checked them visually
per class. (2) A **keyword parser** maps prompt words to those labels: colour words attach to the nearest hair or
outfit noun in the same clause. It isn't a language model, so it only understands what we have labels for.

### 32. Why concatenated label maps rather than a projection discriminator?
I tried both. The projection discriminator adds an inner product between label embeddings and D's features. On
3.6k images it made D far too strong (D(G(z)) ≈ 0.02 and identical samples by epoch 6). With concatenated label
maps, D(G(z)) stayed around 0.2 and samples stayed varied and label-dependent, so I used that. Evidence is in
`outputs/experiments/conditional_probes/`.

### 33. How do you measure whether the prompt works?
Controllability: for each colour, generate 48 sprites requesting it, label them with the same automatic
labeller, and compute the hit rate. The baseline compares the same outputs against **shuffled** requests, which
is what a model ignoring the prompt would score. I also report the labeller's agreement with itself on real
images as an upper bound. Direction is checked visually, since there's no direction classifier.

### 34. What is the EMA of generator weights?
A running average $\theta_{EMA} \leftarrow d\,\theta_{EMA} + (1-d)\,\theta$ (d = 0.998), updated after every G step
and used for sampling. It smooths out the oscillation of adversarial training, so samples are usually cleaner
than from the raw last weights. BatchNorm statistics are copied from the live generator.

### 35. Why do several variations of the same prompt look so similar?
Conditional mode collapse: with the labels fixed, the generator mostly ignores $z$. Each exact label combination
covers only a few dozen training sprites, so the easiest way to fool D is to draw one good character per label
set. I tried the mode-seeking regulariser (MSGAN), which penalises similar outputs for different $z$. It
increased pixel variety, but as noise, and weakened label-following, so I kept the cleaner model and documented
the limitation. Leaving one attribute unspecified gives varied results.

### 36. Why not just use CLIP or a text encoder?
A pretrained text-image model would bring external knowledge the brief excluded. CLIP also works poorly on
64×64 pixel art and needs a large download. A learned text encoder needs captioned data, which TinyHero doesn't
have. The keyword-to-label approach is honest about what the data supports.

---

## Part D — Quick facts to memorise

| Fact | Value |
|---|---|
| Dataset | TinyHero: 912 characters × 4 views = 3,648 RGBA 64×64 PNGs |
| License | CC-BY-SA 3.0 / GPL 3.0 (LPC) |
| Generator / Discriminator parameters | conditional: 3,970,272 / 2,788,096 (plain DCGAN: 3,576,704 / 2,765,568) |
| Prompt vocabulary | 4 directions, 9 hair colours, 9 outfit colours |
| Controllability (trained model) | hair 78.7% vs 11.6% baseline; outfit 71.8% vs 10.4% baseline |
| Prompt model training | 100 epochs, ~32 min on Apple M2 (MPS) |
| Known weakness | little variety when all three attributes are fixed (conditional mode collapse) |
| Latent dimension | 100 |
| Optimiser | Adam, lr 0.0002, β = (0.5, 0.999) |
| Loss | BCEWithLogitsLoss; real target 0.9 |
| Default training | 50 epochs, batch 64, ~57 iterations/epoch |
| Equilibrium losses | $\mathcal{L}_D = 2\ln 2 \approx 1.386$, $\mathcal{L}_G = \ln 2 \approx 0.693$ when D outputs 0.5 |
| Measured M2 speed | ~14 s/epoch on MPS (~12 min for 50 epochs) |
| Tests | 41 pytest tests, including a full train → checkpoint → resume → generate → PNG test |
| Background | black (white caused mode collapse; see Q29) |
