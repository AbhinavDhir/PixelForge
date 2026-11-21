"""PIXELFORGE - AI RPG Character Studio (Streamlit).

    streamlit run app.py

Every character shown under "Generated" comes from the trained DCGAN generator
loaded from checkpoints/. If no checkpoint exists the app says so; it never
substitutes dataset images for generated ones.
"""
from __future__ import annotations

import base64
import csv
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image

from config import PROJECT_ROOT, VIEW_NAMES, Config, describe_device, get_device
from attributes import ATTRIBUTES, HAIR, OUTFIT, VIEWS, parse_prompt
from generate import generate, interpolate, is_conditional, load_generator
from models import count_parameters
from utils import make_grid, remove_background, save_gif, upscale

CKPT_DIR = PROJECT_ROOT / "checkpoints"
OUT_DIR = PROJECT_ROOT / "outputs"
DATA_DIR = PROJECT_ROOT / "data"

st.set_page_config(page_title="PixelForge - AI RPG Character Studio", page_icon="🗡️", layout="wide")

# --------------------------------------------------------------------------- styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&family=VT323&display=swap');
:root { --pf-bg:#0b0a14; --pf-panel:#16132a; --pf-panel2:#1e1a38; --pf-purple:#a855f7; --pf-blue:#3b82f6;
        --pf-cyan:#22d3ee; --pf-ink:#ece9f7; --pf-muted:#a9a6b8; --pf-line:#2c2840; }
.stApp { background:
   radial-gradient(1200px 500px at 10% -10%, rgba(168,85,247,.16), transparent 60%),
   radial-gradient(900px 500px at 110% 0%, rgba(59,130,246,.14), transparent 60%), var(--pf-bg); }
h1, h2, h3 { font-family:'Press Start 2P', monospace !important; letter-spacing:.5px; }
h1 { font-size:1.6rem !important; line-height:1.5 !important; }
h2 { font-size:1.05rem !important; line-height:1.6 !important; color:var(--pf-ink); }
h3 { font-size:.8rem !important; line-height:1.7 !important; color:#c4b5fd; }
.pf-hero { padding:22px 26px; border:3px solid var(--pf-purple); background:linear-gradient(135deg,#1a1433,#0f1530);
   box-shadow: 6px 6px 0 0 var(--pf-blue); margin-bottom:22px; image-rendering:pixelated; }
.pf-hero .t { font-family:'Press Start 2P', monospace; font-size:1.55rem; color:#fff;
   text-shadow: 3px 3px 0 var(--pf-purple), 6px 6px 0 rgba(59,130,246,.6); }
.pf-hero .s { font-family:'VT323', monospace; font-size:1.45rem; color:var(--pf-cyan); margin-top:10px; }
.pf-card { background:var(--pf-panel); border:2px solid var(--pf-line); padding:10px; text-align:center;
   box-shadow:4px 4px 0 0 #000; margin-bottom:6px; transition: transform .08s, border-color .08s; }
.pf-card:hover { border-color:var(--pf-purple); transform: translate(-2px,-2px); box-shadow:6px 6px 0 0 var(--pf-blue); }
.pf-card img { width:100%; image-rendering:pixelated; display:block; }
.pf-cap { font-family:'VT323', monospace; font-size:1.15rem; color:var(--pf-muted); margin-top:6px; }
.pf-badge { display:inline-block; font-family:'Press Start 2P', monospace; font-size:.55rem; padding:6px 8px;
   border:2px solid currentColor; margin-bottom:8px; }
.pf-real { color:#f59e0b; } .pf-gen { color:var(--pf-cyan); }
.pf-chip { display:inline-block; font-family:'VT323', monospace; font-size:1.25rem; padding:2px 10px; margin:0 8px 8px 0;
   border:2px solid var(--pf-purple); background:rgba(168,85,247,.15); color:#fff; }
.pf-chip.rand { border-color:var(--pf-line); color:var(--pf-muted); background:transparent; }
.pf-strip img { width:100%; image-rendering:pixelated; border:3px solid var(--pf-purple); box-shadow:5px 5px 0 0 var(--pf-blue); }
.pf-note { font-family:'VT323', monospace; font-size:1.25rem; color:var(--pf-muted); }
.pf-stat { background:var(--pf-panel); border:2px solid var(--pf-line); padding:12px 14px; box-shadow:4px 4px 0 0 #000; }
.pf-stat .k { font-family:'VT323', monospace; font-size:1.15rem; color:var(--pf-muted); }
.pf-stat .v { font-family:'Press Start 2P', monospace; font-size:.95rem; color:#fff; margin-top:6px; }
.pf-layer { display:flex; align-items:center; gap:10px; overflow-x:auto; padding:10px 2px 16px; }
.pf-block { flex:0 0 auto; display:flex; flex-direction:column; justify-content:flex-end; align-items:center; }
.pf-box { border:2px solid var(--pf-purple); background:linear-gradient(180deg, rgba(168,85,247,.35), rgba(59,130,246,.25));
   box-shadow:3px 3px 0 0 #000; }
.pf-box.d { border-color:var(--pf-blue); background:linear-gradient(180deg, rgba(59,130,246,.35), rgba(34,211,238,.18)); }
.pf-lbl { font-family:'VT323', monospace; font-size:1.05rem; color:var(--pf-ink); margin-top:6px; text-align:center; white-space:nowrap; }
.pf-op { font-family:'VT323', monospace; color:var(--pf-muted); font-size:1rem; text-align:center; white-space:nowrap; }
section[data-testid="stSidebar"] { background:#100d1f; border-right:2px solid var(--pf-line); }
.stButton>button, .stDownloadButton>button { font-family:'Press Start 2P', monospace; font-size:.62rem;
   border:2px solid var(--pf-purple); border-radius:0; box-shadow:3px 3px 0 0 var(--pf-blue); }
.stButton>button:hover, .stDownloadButton>button:hover { border-color:var(--pf-cyan); color:var(--pf-cyan); }
.stButton>button:active { transform: translate(2px,2px); box-shadow:1px 1px 0 0 var(--pf-blue); }
@media (max-width: 640px) { .pf-hero .t { font-size:1rem; } h1 { font-size:1.1rem !important; } }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- helpers
def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def img_html(img: Image.Image, cls: str = "") -> str:
    b64 = base64.b64encode(png_bytes(img)).decode()
    return f'<img class="{cls}" src="data:image/png;base64,{b64}"/>'


def card(img: Image.Image, caption: str, scale: int = 4, max_width: int | None = None) -> None:
    style = f' style="max-width:{max_width}px;margin:0 auto 6px"' if max_width else ""
    st.markdown(f'<div class="pf-card"{style}>{img_html(upscale(img, scale))}<div class="pf-cap">{caption}</div></div>',
                unsafe_allow_html=True)


def badge(kind: str) -> None:
    if kind == "real":
        st.markdown('<span class="pf-badge pf-real">REAL · TINYHERO DATASET</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="pf-badge pf-gen">GENERATED · DCGAN OUTPUT</span>', unsafe_allow_html=True)


def stat(col, k: str, v: str) -> None:
    col.markdown(f'<div class="pf-stat"><div class="k">{k}</div><div class="v">{v}</div></div>', unsafe_allow_html=True)


def find_checkpoints() -> list[Path]:
    if not CKPT_DIR.exists():
        return []
    files = [p for p in CKPT_DIR.rglob("*.pt")]
    order = {"generator_final.pt": 0, "latest.pt": 1}
    # trained (non-smoke) files first, then by preferred name, then newest
    return sorted(files, key=lambda p: ("smoke" in p.parts, order.get(p.name, 2), -p.stat().st_mtime))


@st.cache_resource(show_spinner="Loading generator weights...")
def cached_generator(path: str, mtime: float, device_str: str):
    return load_generator(path, torch.device(device_str))


@st.cache_resource(show_spinner="Loading TinyHero dataset...")
def cached_dataset(data_dir: str, background: tuple):
    from dataset import TinyHeroDataset, dataset_root
    root = dataset_root(data_dir)
    if not (root / "0").is_dir():
        return None
    return TinyHeroDataset(root, background=background)


@st.cache_data(show_spinner=False)
def cached_validation(root: str):
    from dataset import validate_dataset
    return validate_dataset(root)


def dataset_images(ds, idx) -> list[Image.Image]:
    return [Image.fromarray(ds.data[i].permute(1, 2, 0).numpy()) for i in idx]


def read_history() -> tuple[list[dict], list[dict]]:
    def rd(p):
        if not p.exists():
            return []
        with open(p, newline="") as f:
            return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]
    return rd(OUT_DIR / "training_history.csv"), rd(OUT_DIR / "training_iterations.csv")


def no_checkpoint_message() -> None:
    st.error("No trained generator found in `checkpoints/`. Generated characters can only come from a trained "
             "model, so this section is unavailable until you add one.")
    st.markdown("""
**How to get a checkpoint**

1. Open `PixelForge_Complete.ipynb` in Google Colab → *Runtime → Change runtime type → T4 GPU* → *Run all*.
2. The last cell downloads `pixelforge_results.zip`.
3. Unzip it into this project folder so that `checkpoints/generator_final.pt` exists:
   ```bash
   unzip ~/Downloads/pixelforge_results.zip -d .
   ```
4. Refresh this page.

Or train locally (MPS/CPU): `python train.py --epochs 50` — or upload a `.pt` file in the sidebar.
""")


# --------------------------------------------------------------------------- sidebar
device = get_device()
with st.sidebar:
    st.markdown('<div style="font-family:\'Press Start 2P\';font-size:1rem;color:#fff;'
                'text-shadow:2px 2px 0 #a855f7;margin:6px 0 2px">PIXELFORGE</div>'
                '<div class="pf-note">AI RPG Character Studio</div>', unsafe_allow_html=True)
    page = st.radio("Navigate", ["Prompt Studio", "Character Generator", "Latent Space Explorer", "Training Dashboard",
                                 "Dataset Explorer", "Model Architecture", "Evaluation"], label_visibility="collapsed")
    st.divider()
    ckpts = find_checkpoints()
    uploaded = st.file_uploader("Load a checkpoint (.pt)", type=["pt"])
    if uploaded is not None:
        up_path = CKPT_DIR / "uploaded" / Path(uploaded.name).name
        up_path.parent.mkdir(parents=True, exist_ok=True)
        up_path.write_bytes(uploaded.getvalue())
        ckpts = [up_path] + [c for c in ckpts if c != up_path]
    G = cfg = meta = None
    if ckpts:
        choice = st.selectbox("Generator checkpoint", ckpts, format_func=lambda p: str(p.relative_to(CKPT_DIR)))
        try:
            G, cfg, meta = cached_generator(str(choice), choice.stat().st_mtime, str(device))
            st.caption(f"Epoch **{meta['epoch']}** · nz={cfg.nz} · ngf={cfg.ngf}")
            if (meta["epoch"] or 0) < 10:
                st.warning(f"This checkpoint was trained for only {meta['epoch']} epoch(s) "
                           "(e.g. a smoke test). Expect noisy, blob-like output.")
        except Exception as e:  # invalid/corrupt file -> clear message instead of a stack trace
            st.error(f"Could not load {choice.name}: {e}")
    else:
        st.info("No checkpoint yet - see the Character Generator page for instructions.")
    st.caption(f"Device: {describe_device(device)}")

bg = tuple(cfg.background) if cfg else Config().background
if meta:
    _sub = Path(meta["path"]).resolve().parent.relative_to(CKPT_DIR.resolve()) \
        if Path(meta["path"]).resolve().is_relative_to(CKPT_DIR.resolve()) else Path(".")
    if str(_sub) not in (".", "uploaded") and (OUT_DIR / _sub).is_dir():
        OUT_DIR = OUT_DIR / _sub

def make_models():
    """G and D with exactly the loaded checkpoint's architecture (for layer tables and parameter counts)."""
    from models import build_models
    c = cfg or Config()
    return build_models(c.nz, c.ngf, c.ndf, "cpu", d_dropout=c.d_dropout, d_spectral_norm=c.d_spectral_norm,
                        d_minibatch_std=c.d_minibatch_std, n_classes=tuple(c.n_classes) if c.conditional else None,
                        d_cond=c.d_cond)


def d_layers(D):
    """The sequential conv body of any discriminator variant."""
    return D.body.net if hasattr(D, "body") else (D.features if hasattr(D, "features") else D.net)


st.markdown('<div class="pf-hero"><div class="t">PIXELFORGE</div>'
            '<div class="s">&gt; AI RPG Character Studio — new 64×64 heroes from a Deep Convolutional GAN_</div></div>',
            unsafe_allow_html=True)

# --------------------------------------------------------------------------- 0. prompt studio
if page == "Prompt Studio":
    st.header("Prompt Studio")
    if G is None:
        no_checkpoint_message()
        st.stop()
    if not is_conditional(G):
        st.warning("The loaded checkpoint is an unconditional DCGAN: it cannot follow descriptions. "
                   "Select a conditional checkpoint (trained with `python train.py`, the default) in the sidebar.")
        st.stop()
    st.markdown('<p class="pf-note">Describe a character. The prompt is parsed into the attributes the model was trained '
                'on - <b>viewing direction</b>, <b>hair / headwear colour</b> and <b>outfit colour</b> - and the '
                'conditional generator draws new characters with them. Anything else in the prompt is ignored.</p>',
                unsafe_allow_html=True)
    examples = ["a front-facing hero with green hair and a red tunic", "knight in armour seen from behind",
                "blonde adventurer in a blue outfit, facing left", "purple-haired mage in a white robe",
                "red-haired rogue in a brown leather vest, right side"]
    if "prompt" not in st.session_state:
        st.session_state.prompt = examples[0]
    ex_cols = st.columns(len(examples))
    for c, ex in zip(ex_cols, examples):
        if c.button(ex, key=f"ex_{ex}", width="stretch"):
            st.session_state.prompt = ex
    prompt = st.text_input("Describe your character", key="prompt", max_chars=200)
    parsed, _ = parse_prompt(prompt)
    st.markdown("<div>" + "".join(
        f'<span class="pf-chip">{k}: {v[parsed[k]]}</span>' if parsed[k] is not None
        else f'<span class="pf-chip rand">{k}: random</span>' for k, v in ATTRIBUTES.items()) + "</div>",
        unsafe_allow_html=True)
    with st.expander("Adjust attributes manually"):
        oc = st.columns(3)
        final = {}
        for col, (k, vocab) in zip(oc, ATTRIBUTES.items()):
            opts = ["(from prompt / random)"] + vocab
            choice = col.selectbox(k.title(), opts, key=f"ovr_{k}")
            final[k] = vocab.index(choice) if choice != opts[0] else parsed[k]
    c1, c2, c3 = st.columns([1, 1, 1])
    n = c1.selectbox("Variations", [4, 8, 12, 16], index=1)
    if "pseed" not in st.session_state:
        st.session_state.pseed = 7
    if c2.button("🎲 New variations", width="stretch"):
        st.session_state.pseed = int(np.random.randint(0, 2**31 - 1))
    seed = c3.number_input("Seed", 0, 2**31 - 1, key="pseed")
    imgs = generate(G, n, seed=int(seed), device=device, labels=final)
    if all(v is not None for v in final.values()):
        st.info("All three attributes are fixed, so the variations differ only in details (shade, small accessories). "
                "The model relies mostly on the labels: it learned little within-label variety from 3.6k sprites. "
                "Leave an attribute out of the prompt to see more varied characters.")
    badge("gen")
    per_row = 4
    for r in range(0, len(imgs), per_row):
        cols = st.columns(per_row)
        for j, im in enumerate(imgs[r:r + per_row]):
            with cols[j]:
                card(im, f"#{r + j:02d}", 4)
                st.download_button("⬇ PNG", png_bytes(im), f"pixelforge_prompt_{seed}_{r + j:02d}.png", "image/png",
                                   key=f"pdl{r + j}", width="stretch")
    grid = make_grid(imgs, nrow=per_row, scale=4, pad_color=(11, 10, 20))
    st.download_button("⬇ Download all as one PNG", png_bytes(grid), f"pixelforge_prompt_{seed}.png", "image/png")
    st.markdown('<p class="pf-note">How it works: <i>prompt → keyword parser → labels y = (view, hair, outfit) → '
                'G(z, y)</i>. Each variation uses a different random z with the same labels. Labels were extracted '
                'automatically from the training sprites, so control is approximate - see the Evaluation page for the '
                'measured controllability.</p>', unsafe_allow_html=True)

# --------------------------------------------------------------------------- A. generator
elif page == "Character Generator":
    st.header("Character Generator")
    if G is None:
        no_checkpoint_message()
        st.stop()
    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1, 1])
    if "seed" not in st.session_state:
        st.session_state.seed = 42
    if c2.button("🎲 Random seed", width="stretch"):
        st.session_state.seed = int(np.random.randint(0, 2**31 - 1))
    seed = c1.number_input("Seed", min_value=0, max_value=2**31 - 1, step=1, key="seed")
    n = c3.selectbox("Characters", [4, 8, 9, 12, 16, 25], index=4)
    scale = c4.selectbox("Preview zoom", [3, 4, 6, 8], index=1, format_func=lambda s: f"{s}× ({64 * s}px)")
    transparent = st.toggle("Remove background (heuristic flood-fill, not learned - see README)", value=False)
    if st.button("⚔️ FORGE CHARACTERS", type="primary") or "gen" not in st.session_state \
            or st.session_state.get("gen_key") != (seed, n, meta["path"]):
        st.session_state.gen = generate(G, n, seed=int(seed), device=device)
        st.session_state.gen_key = (seed, n, meta["path"])
    imgs = st.session_state.gen
    if transparent:
        imgs = [remove_background(im, bg) for im in imgs]
    badge("gen")
    per_row = 4 if n in (4, 8, 12, 16) else 5 if n == 25 else 3
    for r in range(0, len(imgs), per_row):
        cols = st.columns(per_row)
        for j, im in enumerate(imgs[r:r + per_row]):
            i = r + j
            with cols[j]:
                card(im, f"seed {seed} · #{i:02d}", scale)
                st.download_button("⬇ PNG", png_bytes(upscale(im, 1)), f"pixelforge_seed{seed}_{i:02d}.png",
                                   "image/png", key=f"dl{i}", width="stretch")
    st.divider()
    d1, d2 = st.columns(2)
    grid = make_grid(imgs, nrow=per_row, scale=scale, pad_color=(11, 10, 20))
    d1.download_button("⬇ Download grid PNG", png_bytes(grid), f"pixelforge_grid_seed{seed}.png", "image/png",
                       width="stretch")
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w") as zf:
        for i, im in enumerate(imgs):
            zf.writestr(f"pixelforge_seed{seed}_{i:02d}.png", png_bytes(im))
            zf.writestr(f"preview_x8/pixelforge_seed{seed}_{i:02d}.png", png_bytes(upscale(im, 8)))
    d2.download_button("⬇ Download all sprites (ZIP)", zbuf.getvalue(), f"pixelforge_seed{seed}.zip",
                       "application/zip", width="stretch")
    st.markdown('<p class="pf-note">Each sprite is the raw 64×64 RGB output of the generator for one latent '
                'vector z ~ N(0, I) drawn with this seed. Previews are enlarged with nearest-neighbour '
                'scaling so pixels stay sharp. Same seed → same characters, on CPU, MPS or CUDA.</p>',
                unsafe_allow_html=True)

# --------------------------------------------------------------------------- B. latent explorer
elif page == "Latent Space Explorer":
    st.header("Latent Space Explorer")
    if G is None:
        no_checkpoint_message()
        st.stop()
    c1, c2, c3, c4 = st.columns(4)
    sa = c1.number_input("Seed A (start)", 0, 2**31 - 1, 7)
    sb = c2.number_input("Seed B (end)", 0, 2**31 - 1, 2024)
    steps = c3.slider("Frames", 8, 12, 10)
    mode = c4.selectbox("Interpolation", ["linear", "slerp"],
                        format_func=lambda m: "Linear (lerp)" if m == "linear" else "Spherical (slerp)")
    fixed = None
    if is_conditional(G):
        lc = st.columns(3)
        fixed = {}
        for col, (k, vocab) in zip(lc, ATTRIBUTES.items()):
            ch = col.selectbox(f"{k.title()} (held fixed)", ["random"] + vocab, key=f"lat_{k}")
            fixed[k] = vocab.index(ch) if ch != "random" else None
    frames = interpolate(G, int(sa), int(sb), steps, mode, device, labels=fixed)
    e1, e2, e3 = st.columns([1, 2, 1])
    with e1:
        badge("gen")
        card(frames[0], f"z_A (seed {sa})", 4)
    with e3:
        badge("gen")
        card(frames[-1], f"z_B (seed {sb})", 4)
    with e2:
        t = st.slider("Scrub t", 0, steps - 1, steps // 2, format="frame %d")
        card(frames[t], f"t = {t / (steps - 1):.2f}", 4, max_width=300)
    strip = make_grid(frames, nrow=steps, scale=3, pad_color=(11, 10, 20))
    st.markdown(f'<div class="pf-strip">{img_html(strip)}</div>', unsafe_allow_html=True)
    st.write("")
    d1, d2 = st.columns(2)
    d1.download_button("⬇ Download strip PNG", png_bytes(strip), f"interp_{sa}_{sb}_{mode}.png", "image/png",
                       width="stretch")
    gif_path = OUT_DIR / "interpolation" / f"app_interp_{sa}_{sb}_{mode}.gif"
    save_gif([upscale(f, 4) for f in frames], gif_path)
    d2.download_button("⬇ Download GIF", gif_path.read_bytes(), gif_path.name, "image/gif", width="stretch")
    st.latex(r"z(t) = (1-t)\,z_A + t\,z_B, \quad t\in[0,1] \qquad\text{(slerp keeps } \|z\| \text{ typical for a Gaussian)}")
    st.markdown('<p class="pf-note">The generator is a smooth (continuous) function, so nearby latent vectors '
                'decode to related images. Smooth transitions suggest the model learned a structured latent '
                'space rather than memorising individual sprites. Intermediate frames are not guaranteed '
                'to be meaningful characters, and no single direction is guaranteed to control one attribute.</p>',
                unsafe_allow_html=True)

# --------------------------------------------------------------------------- C. training dashboard
elif page == "Training Dashboard":
    st.header("Training Dashboard")
    hist, iters = read_history()
    if not hist and meta and meta.get("history"):
        hist = meta["history"]
    if not hist:
        st.info("No training history found (`outputs/training_history.csv`). Train the model or unzip the Colab "
                "results into this folder.")
    else:
        last = hist[-1]
        cols = st.columns(6)
        stat(cols[0], "Epochs trained", f"{int(last['epoch'])}")
        stat(cols[1], "Final G loss", f"{last['loss_g']:.3f}")
        stat(cols[2], "Final D loss", f"{last['loss_d']:.3f}")
        stat(cols[3], "D(x) real", f"{last['d_real']:.2f}")
        stat(cols[4], "D(G(z)) fake", f"{last['d_fake']:.2f}")
        stat(cols[5], "Sec / epoch", f"{np.mean([h['seconds'] for h in hist]):.1f}")
        st.write("")
        import pandas as pd
        if iters:
            df = pd.DataFrame(iters)
            k = max(1, len(df) // 80)
            df["G (moving avg)"] = df["loss_g"].rolling(k, min_periods=1).mean()
            df["D (moving avg)"] = df["loss_d"].rolling(k, min_periods=1).mean()
            x = "iteration"
        else:
            df = pd.DataFrame(hist)
            df["G (moving avg)"], df["D (moving avg)"] = df["loss_g"], df["loss_d"]
            x = "epoch"
        g1, g2 = st.columns(2)
        with g1:
            st.subheader("Generator loss")
            st.line_chart(df, x=x, y="G (moving avg)", color="#3987e5", height=260)
        with g2:
            st.subheader("Discriminator loss")
            st.line_chart(df, x=x, y="D (moving avg)", color="#d95926", height=260)
        if "sample_diversity" in hist[0]:
            st.subheader("Sample diversity per epoch (mode-collapse monitor)")
            st.line_chart(pd.DataFrame(hist), x="epoch", y="sample_diversity", color="#3987e5", height=220)
            st.markdown('<p class="pf-note">Pixel std across 64 fixed-noise samples' +
                        (' that all request the same attributes (so only z varies)' if cfg and cfg.conditional else '') +
                        '. A drop towards 0 means every latent vector produces the same sprite (mode collapse). '
                        '64 random real sprites on black score about 0.07.</p>', unsafe_allow_html=True)
        st.subheader("Discriminator outputs per epoch")
        dfe = pd.DataFrame(hist).rename(columns={"d_real": "D(x) real", "d_fake": "D(G(z)) fake"})
        st.line_chart(dfe, x="epoch", y=["D(x) real", "D(G(z)) fake"], color=["#3987e5", "#d95926"], height=240)
        st.markdown('<p class="pf-note">GAN losses are a two-player game: they do not steadily go to zero, and a '
                    'lower generator loss does not by itself mean better images. Healthy training shows both '
                    'losses staying bounded without one network winning completely (D(x)→1 and D(G(z))→0 '
                    'would mean the discriminator has won and the generator gets no useful gradient).</p>',
                    unsafe_allow_html=True)
    st.subheader("Training progression (fixed noise)")
    prog = sorted((OUT_DIR / "training_progress").glob("epoch_*.png"))
    if prog:
        badge("gen")
        i = st.select_slider("Epoch", options=list(range(len(prog))), value=len(prog) - 1,
                             format_func=lambda j: prog[j].stem.replace("epoch_", "epoch "))
        cond_note = (" Rows request hair colours black, brown, blonde, red, grey, green, blue, purple; columns cycle "
                     "back, left, front, right; each column requests a different outfit colour."
                     if cfg and cfg.conditional else "")
        st.image(Image.open(prog[i]), caption=f"{prog[i].name}: the same 64 latent vectors decoded after "
                 f"{int(prog[i].stem.split('_')[1])} epochs.{cond_note}", width=640)
    else:
        st.info("No progression grids found in `outputs/training_progress/`.")
    st.subheader("Model parameters")
    p1, p2, p3 = st.columns(3)
    nz, ngf, ndf = (cfg.nz, cfg.ngf, cfg.ndf) if cfg else (100, 64, 64)
    _G, _D = make_models()
    stat(p1, "Generator params", f"{count_parameters(_G):,}")
    stat(p2, "Discriminator params", f"{count_parameters(_D):,}")
    stat(p3, "Weights size (G, fp32)", f"{count_parameters(_G) * 4 / 1e6:.1f} MB")

# --------------------------------------------------------------------------- D. dataset explorer
elif page == "Dataset Explorer":
    st.header("Dataset Explorer · TinyHero")
    ds = cached_dataset(str(DATA_DIR), bg)
    if ds is None:
        st.warning("TinyHero is not downloaded yet.")
        if st.button("Download TinyHero (7.3 MB)"):
            from dataset import download_tinyhero
            with st.spinner("Downloading..."):
                download_tinyhero(DATA_DIR)
            st.cache_resource.clear()
            st.rerun()
        st.stop()
    v = cached_validation(str(DATA_DIR / "tinyhero"))
    cols = st.columns(5)
    stat(cols[0], "Images", f"{v['total_images']:,}")
    stat(cols[1], "Characters", f"{v['characters_in_all_views']}")
    stat(cols[2], "Views", "4")
    stat(cols[3], "Resolution", "64×64")
    stat(cols[4], "Source mode", "/".join(v["modes"]))
    st.write("")
    st.markdown(f"""
TinyHero ([AgaMiko/pixel_character_generator](https://github.com/AgaMiko/pixel_character_generator)) contains
**{v['characters_in_all_views']} characters × 4 viewing directions = {v['total_images']} sprites**, built from
the Universal LPC Spritesheet. Each character has a random sex, body type, skin colour and equipment.
Files are 64×64 **RGBA** PNGs with binary alpha (~{(v['mean_transparent_fraction'] or 0) * 100:.0f}% of pixels transparent).
PixelForge alpha-composites them onto a solid RGB background ({bg}) and normalises pixels to **[-1, 1]**.

*License:* dual **CC-BY-SA 3.0** / **GNU GPL 3.0** (LPC art rules). See `LICENSES/`.
""")
    view = st.radio("View", ["all", *VIEW_NAMES.values()], horizontal=True)
    if st.button("🔀 Shuffle samples"):
        st.session_state.ds_seed = int(np.random.randint(0, 10**6))
    rng = np.random.default_rng(st.session_state.get("ds_seed", 0))
    pool = np.arange(len(ds)) if view == "all" else \
        np.where(ds.labels.numpy() == [k for k, n in VIEW_NAMES.items() if n == view][0])[0]
    idx = rng.choice(pool, size=25, replace=False)
    badge("real")
    st.markdown(f'<div class="pf-strip">{img_html(make_grid(dataset_images(ds, idx), nrow=5, scale=4, pad_color=(11, 10, 20)))}</div>',
                unsafe_allow_html=True)
    st.subheader("Same character, four directions")
    badge("real")
    per_view = len(ds) // 4
    chars = rng.choice(per_view, size=4, replace=False)
    rows = [ds.data[c + k * per_view] for c in chars for k in range(4)]
    imgs = [Image.fromarray(r.permute(1, 2, 0).numpy()) for r in rows]
    st.markdown(f'<div class="pf-strip">{img_html(make_grid(imgs, nrow=8, scale=3, pad_color=(11, 10, 20)))}</div>',
                unsafe_allow_html=True)
    st.caption("Columns alternate back / left / front / right for each character.")
    with st.expander("Full validation report"):
        st.json(v)

# --------------------------------------------------------------------------- E. architecture
elif page == "Model Architecture":
    st.header("Model Architecture")
    nz, ngf, ndf = (cfg.nz, cfg.ngf, cfg.ndf) if cfg else (100, 64, 64)

    def diagram(stages, cls=""):
        html = '<div class="pf-layer">'
        for i, (label, c, s, op) in enumerate(stages):
            h = max(14, int(160 * s / 64)) if s > 1 else 14
            w = max(10, int(14 + 50 * np.log2(max(c, 2)) / 9))
            html += (f'<div class="pf-block"><div class="pf-box {cls}" style="width:{w}px;height:{h}px"></div>'
                     f'<div class="pf-lbl">{label}</div></div>')
            if op:
                html += f'<div class="pf-op">{op}<br>→</div>'
        return html + "</div>"

    cond = bool(cfg and cfg.conditional)
    g_lab = 16 * len(cfg.n_classes) if cond else 0          # label embeddings fed to G
    d_lab = sum(cfg.n_classes) if cond and cfg.d_cond == "concat" else 0  # one-hot label maps fed to D
    if cond:
        st.markdown('<p class="pf-note">Conditional model: labels y = (view, hair colour, outfit colour). G receives '
                    f'z plus three 16-d label embeddings; D receives the image plus {sum(cfg.n_classes)} one-hot label '
                    'maps, so it judges "real sprite <i>with these attributes</i>?".</p>', unsafe_allow_html=True)
    st.subheader("Generator G(z, y): noise + labels -> sprite" if cond else "Generator G(z): noise -> sprite")
    st.markdown(diagram([
        (f"z {nz} + y {g_lab}" if g_lab else f"z {nz}×1×1", nz, 1, "ConvT 4×4 s1<br>BN · ReLU"), (f"{ngf*8}×4×4", ngf * 8, 4, "ConvT 4×4 s2<br>BN · ReLU"),
        (f"{ngf*4}×8×8", ngf * 4, 8, "ConvT 4×4 s2<br>BN · ReLU"), (f"{ngf*2}×16×16", ngf * 2, 16, "ConvT 4×4 s2<br>BN · ReLU"),
        (f"{ngf}×32×32", ngf, 32, "ConvT 4×4 s2<br>Tanh"), ("3×64×64 RGB", 3, 64, None)]), unsafe_allow_html=True)
    st.subheader("Discriminator D(x, y): sprite + labels -> real/fake" if cond else "Discriminator D(x): sprite -> real/fake")
    st.markdown(diagram([
        (f"RGB 3 + labels {d_lab}" if d_lab else "3×64×64 RGB", 3, 64, "Conv 4×4 s2<br>LeakyReLU 0.2"), (f"{ndf}×32×32", ndf, 32, "Conv 4×4 s2<br>BN · LReLU"),
        (f"{ndf*2}×16×16", ndf * 2, 16, "Conv 4×4 s2<br>BN · LReLU"), (f"{ndf*4}×8×8", ndf * 4, 8, "Conv 4×4 s2<br>BN · LReLU"),
        (f"{ndf*8}×4×4", ndf * 8, 4, "Conv 4×4 s1<br>(no sigmoid)"), ("1 logit", 1, 1, None)], "d"), unsafe_allow_html=True)
    a1, a2 = st.columns(2)
    with a1:
        st.subheader("Layer table")
        def layer_rows(seq, x):
            rows = []
            for m in seq:
                x = m(x)
                n = sum(p.numel() for p in m.parameters())
                rows.append({"layer": m.__class__.__name__, "output": "×".join(map(str, x.shape[1:])), "params": n})
            return rows
        with torch.no_grad():
            _G, _D = make_models()
            _G.eval(), _D.eval()
            st.caption(f"Generator - {count_parameters(_G):,} parameters (input: {_G.net[0].in_channels} channels)")
            st.dataframe(layer_rows(_G.net, torch.zeros(1, _G.net[0].in_channels, 1, 1)), hide_index=True, width="stretch")
            body = d_layers(_D)
            st.caption(f"Discriminator - {count_parameters(_D):,} parameters (input: {body[0].in_channels} channels)")
            st.dataframe(layer_rows(body, torch.zeros(1, body[0].in_channels, 64, 64)), hide_index=True, width="stretch")
    with a2:
        st.subheader("How GAN training works")
        st.markdown("""
Two networks play a minimax game:

* **Discriminator D** sees real TinyHero sprites and generator fakes and learns to output a high logit for real, low for fake.
* **Generator G** turns random noise **z** into a sprite and is rewarded when **D** is fooled.

Each training step: (1) update D on a real batch and a *detached* fake batch, (2) update G through D using the
non-saturating loss. As D gets better at spotting fakes, G is pushed to produce more realistic characters.
""")
        st.latex(r"\min_G \max_D \; \mathbb{E}_{x}[\log D(x)] + \mathbb{E}_{z}[\log(1-D(G(z)))]")
        st.latex(r"\mathcal{L}_D = \mathrm{BCE}(D(x), 0.9) + \mathrm{BCE}(D(G(z)), 0)")
        st.latex(r"\mathcal{L}_G = \mathrm{BCE}(D(G(z)), 1)")
        st.subheader("Hyperparameters")
        c = cfg or Config()
        stabilisers = [n for n, on in (("instance noise", c.instance_noise > 0), ("dropout", c.d_dropout > 0),
                                       ("spectral norm", c.d_spectral_norm), ("minibatch std", c.d_minibatch_std)) if on]
        st.dataframe([{"parameter": k, "value": str(val)} for k, val in {
            "latent dim (nz)": c.nz, "G base channels (ngf)": c.ngf, "D base channels (ndf)": c.ndf,
            "batch size": c.batch_size, "learning rate": c.lr, "Adam betas": (c.beta1, c.beta2),
            "epochs (configured)": c.epochs, "one-sided label smoothing": c.label_smoothing,
            "loss": "BCEWithLogitsLoss", "init": "N(0, 0.02) conv, N(1, 0.02) BN", "seed": c.seed,
            "views": [VIEW_NAMES[v] for v in c.views], "background RGB": c.background,
            "D stabilisers": ", ".join(stabilisers) or "none"}.items()], hide_index=True, width="stretch")

# --------------------------------------------------------------------------- F. evaluation
elif page == "Evaluation":
    st.header("Evaluation")
    if G is None:
        no_checkpoint_message()
        st.stop()
    ds = cached_dataset(str(DATA_DIR), bg)
    if ds is None:
        st.warning("Download TinyHero first (Dataset Explorer page).")
        st.stop()
    from evaluate import _pairwise_stats, _rmse_matrix, generate_batch01
    seed = st.number_input("Evaluation seed", 0, 10**6, 123)
    st.subheader("Real vs generated")
    gen = generate_batch01(G, 256, int(seed), device)
    rng = np.random.default_rng(int(seed))
    ridx = rng.choice(len(ds), 256, replace=False)
    real_all = ds.as_float01()
    real = real_all[ridx]
    to_pil = lambda x: [Image.fromarray((t.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)) for t in x]
    r1, r2 = st.columns(2)
    with r1:
        badge("real")
        st.markdown(f'<div class="pf-strip">{img_html(make_grid(to_pil(real[:16]), nrow=4, scale=3, pad_color=(11,10,20)))}</div>', unsafe_allow_html=True)
    with r2:
        badge("gen")
        st.markdown(f'<div class="pf-strip">{img_html(make_grid(to_pil(gen[:16]), nrow=4, scale=3, pad_color=(11,10,20)))}</div>', unsafe_allow_html=True)

    st.subheader("Diversity analysis (256 samples, pixel RMSE)")
    pg, pr = _pairwise_stats(gen), _pairwise_stats(real)
    d_gen = _rmse_matrix(gen, real_all)
    nn_gen, nn_idx = d_gen.min(dim=1)
    d_real = _rmse_matrix(real, real_all)
    d_real[torch.arange(len(ridx)), torch.as_tensor(ridx)] = float("inf")
    nn_real = d_real.min(dim=1).values
    cols = st.columns(4)
    stat(cols[0], "Pairwise dist (gen)", f"{pg['mean']:.3f}")
    stat(cols[1], "Pairwise dist (real)", f"{pr['mean']:.3f}")
    stat(cols[2], "Diversity ratio", f"{pg['mean'] / pr['mean']:.2f}")
    stat(cols[3], "Near-copies (<0.02)", f"{int((nn_gen < 0.02).sum())}")
    st.write("")
    cols = st.columns(2)
    stat(cols[0], "Median NN dist gen→train", f"{nn_gen.median():.3f}")
    stat(cols[1], "Median NN dist real→other train", f"{nn_real.median():.3f}")
    st.markdown('<p class="pf-note">Diversity ratio ≈ 1 → generated samples are as spread out as real ones; '
                '≪ 1 → mode collapse. If generated samples were much closer to their nearest training image than real '
                'images are to theirs, the model would be memorising.</p>', unsafe_allow_html=True)

    st.subheader("Nearest-neighbour comparison")
    st.caption("Left: generated sample. Right: the 3 closest TinyHero training images by pixel RMSE.")
    top = d_gen[:6].topk(3, largest=False)
    for r in range(6):
        cs = st.columns([1, 0.15, 1, 1, 1])
        with cs[0]:
            card(to_pil(gen[r:r + 1])[0], "GENERATED", 2)
        for j in range(3):
            with cs[2 + j]:
                card(to_pil(real_all[top.indices[r, j]][None])[0], f"real NN{j+1} · {top.values[r, j]:.3f}", 2)

    saved = OUT_DIR / "evaluation_metrics.json"
    if saved.exists() and "controllability" in json.loads(saved.read_text()):
        ctrl = json.loads(saved.read_text())["controllability"]
        st.subheader("Prompt controllability (offline evaluation)")
        cc = st.columns(4)
        stat(cc[0], "Hair: requested obtained", f"{ctrl['hair']['accuracy']:.0%}")
        stat(cc[1], "Hair: if ignored", f"{ctrl['hair']['shuffled_baseline']:.0%}")
        stat(cc[2], "Outfit: requested obtained", f"{ctrl['outfit']['accuracy']:.0%}")
        stat(cc[3], "Outfit: if ignored", f"{ctrl['outfit']['shuffled_baseline']:.0%}")
        if (OUT_DIR / "plots" / "controllability.png").exists():
            st.image(str(OUT_DIR / "plots" / "controllability.png"))
        if (OUT_DIR / "plots" / "prompt_control_grid.png").exists():
            st.image(str(OUT_DIR / "plots" / "prompt_control_grid.png"),
                     caption="Rows: requested hair colour (" + ", ".join(HAIR) + "). Columns: back, left, front, right (x2).")
    if saved.exists():
        with st.expander("Saved offline evaluation (python evaluate.py)"):
            st.json(json.loads(saved.read_text()))
            for name in ("controllability", "prompt_control_grid", "real_vs_generated", "progression",
                         "nearest_neighbors", "diversity", "interpolation_experiment"):
                p = OUT_DIR / "plots" / f"{name}.png"
                if p.exists():
                    st.image(str(p), caption=name)

    st.subheader("Limitations")
    st.markdown("""
* **Loss ≠ quality.** GAN losses describe the game between G and D, not image quality.
* **Pixel distances are crude.** RMSE ignores perceptual similarity; a 1-pixel shift looks "far".
* **No FID.** FID needs an ImageNet Inception network and many thousands of samples; at 64×64 pixel art with
  3.6k training images it would be unreliable.
* **Small dataset, 64×64 resolution, unconditional model** - you cannot ask for a specific class, weapon or direction.
* **Artifacts.** DCGANs can produce checkerboard patterns (transposed convolutions) and blurry or incoherent details.
""")
