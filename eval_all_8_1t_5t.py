#!/usr/bin/env python3
"""
eval_all8_1t_5t.py  —  1-timestep vs 5-timestep vs full on all 8 models
=========================================================================
Runs the motivational 1t / 5t / full trajectory experiment on the complete
set of 8 models across all three axes:

  SD v1.4, v1.5, v2.1, Dreamshaper-8, Realistic Vision v5   (Axis 1 — data)
  SDXL, PixArt-α                                             (Axis 2 — arch)
  SDXL-Turbo                                                 (Axis 3 — procedure)

For each plan (1step, 5step) we run 8-way classification and
report: argmin Top-1, silhouette, distance ratio, LR, kNN.

The progression 1t → 5t → full directly motivates Takeaway 3:
  "Model identity is in the trajectory, not a single timestep."

Usage:
  python eval_all8_1t_5t.py --proto_n 50 --test_n 500 --end 550

  # Smoke test
  python eval_all8_1t_5t.py --proto_n 5 --test_n 20 --end 25

  # Run only one plan
  python eval_all8_1t_5t.py --plans 1step --proto_n 50 --test_n 500 --end 550
"""

import argparse, os, csv, gc, json, random, hashlib
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Dict
import os
# Must be set BEFORE torch is imported
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Commented out - set via command line or env var

import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    PixArtAlphaPipeline,
    DPMSolverMultistepScheduler,
    DDIMScheduler,
    EulerAncestralDiscreteScheduler,
)
from datasets import load_dataset

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import silhouette_score, accuracy_score, confusion_matrix
from scipy.spatial.distance import cdist
from collections import Counter


# =============================================================================
# All 8 models
# =============================================================================
ALL_MODELS = {
    # ── Axis 1: same architecture, different training data ──────────────────
    "v14": {
        "id":       "CompVis/stable-diffusion-v1-4",
        "type":     "sd15",
        "arch":     "UNet-860M",
        "axis":     1,
        "gen_size": 512,
        "gen_steps": 50, "gen_cfg": 7.5,
        "desc":     "SD v1.4",
    },
    "v15": {
        "id":       "runwayml/stable-diffusion-v1-5",
        "type":     "sd15",
        "arch":     "UNet-860M",
        "axis":     1,
        "gen_size": 512,
        "gen_steps": 50, "gen_cfg": 7.5,
        "desc":     "SD v1.5",
    },
    "v21": {
        "id":       "Manojb/stable-diffusion-2-1-base",
        "type":     "sd15",
        "arch":     "UNet-860M",
        "axis":     1,
        "gen_size": 512,
        "gen_steps": 50, "gen_cfg": 7.5,
        "desc":     "SD v2.1",
    },
    "ds8": {
        "id":       "Lykon/dreamshaper-8",
        "type":     "sd15",
        "arch":     "UNet-860M",
        "axis":     1,
        "gen_size": 512,
        "gen_steps": 50, "gen_cfg": 7.5,
        "desc":     "Dreamshaper-8",
    },
    "rv5": {
        "id":       "SG161222/Realistic_Vision_V5.1_noVAE",
        "type":     "sd15",
        "arch":     "UNet-860M",
        "axis":     1,
        "gen_size": 512,
        "gen_steps": 50, "gen_cfg": 7.5,
        "desc":     "Realistic Vision v5",
    },
    # ── Axis 2: same data family, different architecture ────────────────────
    "sdxl": {
        "id":       "stabilityai/stable-diffusion-xl-base-1.0",
        "type":     "sdxl",
        "arch":     "UNet-2.6B",
        "axis":     2,
        "gen_size": 1024,
        "gen_steps": 50, "gen_cfg": 7.5,
        "desc":     "SDXL",
    },
    "pixart": {
        "id":       "PixArt-alpha/PixArt-XL-2-512x512",
        "type":     "pixart",
        "arch":     "DiT-XL/2",
        "axis":     2,
        "gen_size": 512,
        "gen_steps": 50, "gen_cfg": 4.5,
        "desc":     "PixArt-α",
    },
    # ── Axis 3: same architecture + data, different training procedure ───────
    "turbo": {
        "id":       "stabilityai/sdxl-turbo",
        "type":     "sdxl_turbo",
        "arch":     "UNet-2.6B",
        "axis":     3,
        "gen_size": 512,
        "gen_steps": 4, "gen_cfg": 0.0,
        "desc":     "SDXL-Turbo",
    },
}

MODEL_KEYS   = list(ALL_MODELS.keys())
FIXED_VAE_ID = "runwayml/stable-diffusion-v1-5"
PROBE_SIZE   = 512


# =============================================================================
# ProbeConfig
# =============================================================================
@dataclass
class ProbeConfig:
    rings:         int         = 8
    probe_repeats: int         = 6
    timestep_plan: str         = "midlate"
    noise_scales:  List[float] = field(default_factory=lambda: [1.0, 1.25, 1.5])
    normalize:     bool        = True
    probe_seed:    int         = 123


def choose_timestep_indices(plan: str, T: int) -> List[int]:
    if plan == "1step":
        return [T // 2]
    if plan == "5step":
        start = int(0.2 * T); end = T - 1
        return sorted(set(
            min(start + int(i * (end - start) / 4), T-1) for i in range(5)))
    if plan == "midlate":
        ms = int(0.2*T); me = int(0.5*T)
        mids = list(range(ms, me, max(1, (me-ms)//5)))
        late = list(range(int(0.5*T), T, max(1, (T-int(0.5*T))//10)))
        return sorted(set(min(i, T-1) for i in mids+late))
    raise ValueError(f"Unknown plan: {plan!r}  (valid: 1step, 5step)")


def _probe_hash(conf: ProbeConfig, model_id: str) -> str:
    key = (
        f"model={model_id}"
        f"|rings={conf.rings}"
        f"|repeats={conf.probe_repeats}"
        f"|scales={'_'.join(str(s) for s in conf.noise_scales)}"
        f"|plan={conf.timestep_plan}"
        f"|norm={int(conf.normalize)}"
        f"|seed={conf.probe_seed}"
    )
    return hashlib.md5(key.encode()).hexdigest()[:12]


# =============================================================================
# Reproducibility
# =============================================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# FFT helpers
# =============================================================================
def fftshift2(x):  return torch.fft.fftshift(x, dim=(-2, -1))
def ifftshift2(x): return torch.fft.ifftshift(x, dim=(-2, -1))


def radial_ring_masks(h, w, device, dtype, K):
    yy = torch.linspace(-1., 1., h, device=device, dtype=dtype)
    xx = torch.linspace(-1., 1., w, device=device, dtype=dtype)
    Y, X = torch.meshgrid(yy, xx, indexing="ij")
    R    = (X*X + Y*Y).sqrt() / (2.**0.5)
    edges = torch.linspace(0., 1., K+1, device=device, dtype=dtype)
    return [((R.clamp(0,1) >= edges[k]) &
             (R.clamp(0,1) <  edges[k+1])).to(dtype) for k in range(K)]


def apply_freq_mask(noise, mask):
    X   = fftshift2(torch.fft.fft2(noise, dim=(-2,-1)))
    X   = ifftshift2(X * mask[None, None])
    out = torch.fft.ifft2(X, dim=(-2,-1)).real
    eps = 1e-8
    s0  = noise.std(dim=(-3,-2,-1), keepdim=True).clamp_min(eps)
    s1  = out.std(  dim=(-3,-2,-1), keepdim=True).clamp_min(eps)
    return out * (s0 / s1)


def band_energy(residual, masks):
    X    = fftshift2(torch.fft.fft2(residual, dim=(-2,-1)))
    mag2 = X.real**2 + X.imag**2
    return torch.stack([(mag2 * m[None,None]).mean() for m in masks])


# =============================================================================
# Wrappers
# =============================================================================
class SD15Wrapper:
    def __init__(self, model_id: str, device, dtype):
        self.device = device; self.dtype = dtype
        self._pipe  = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=dtype,
            safety_checker=None, requires_safety_checker=False,
        ).to(device)
        self._pipe.set_progress_bar_config(disable=True)
        try:
            self._pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self._pipe.scheduler.config)
        except ValueError as e:
            if "final_sigmas_type" in str(e):
                cfg = dict(self._pipe.scheduler.config)
                cfg["final_sigmas_type"] = "sigma_min"
                self._pipe.scheduler = DPMSolverMultistepScheduler.from_config(cfg)
            else:
                raise
        self._probe_sched = DDIMScheduler.from_pretrained(
            model_id, subfolder="scheduler")

    def generate(self, prompt, seed, size=512):
        set_seed(seed)
        return self._pipe(prompt, height=size, width=size,
                          num_inference_steps=50, guidance_scale=7.5).images[0]

    @torch.no_grad()
    def encode_to_latent(self, img):
        vae_dev = next(self._pipe.vae.parameters()).device
        x = self._pipe.image_processor.preprocess(
            img.resize((PROBE_SIZE, PROBE_SIZE), Image.LANCZOS)
        ).to(vae_dev, self.dtype)
        return self._pipe.vae.encode(x).latent_dist.mean \
               * self._pipe.vae.config.scaling_factor

    @torch.no_grad()
    def unet_forward(self, zin, t, cond):
        unet_dev = next(self._pipe.unet.parameters()).device
        return self._pipe.unet(
            zin.to(unet_dev), t,
            encoder_hidden_states=cond.to(unet_dev),
            return_dict=False)[0]

    def get_probe_scheduler(self, device):
        self._probe_sched.set_timesteps(50, device=device); return self._probe_sched

    def get_empty_cond(self, device):
        enc_dev = next(self._pipe.text_encoder.parameters()).device
        cond, _ = self._pipe.encode_prompt(
            "", device=enc_dev, num_images_per_prompt=1,
            do_classifier_free_guidance=False, negative_prompt=None)
        return cond

    def free_memory(self):
        del self._pipe; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()


class SDXLWrapper:
    def __init__(self, model_id: str, device, dtype):
        self.device = device; self.dtype = dtype
        self._pipe  = StableDiffusionXLPipeline.from_pretrained(
            model_id, torch_dtype=dtype, use_safetensors=True,
        ).to(device)
        self._pipe.set_progress_bar_config(disable=True)
        shared = StableDiffusionPipeline.from_pretrained(
            FIXED_VAE_ID, torch_dtype=dtype,
            safety_checker=None, requires_safety_checker=False)
        self._vae     = shared.vae.to(device)
        self._vscale  = shared.vae.config.scaling_factor
        self._imgproc = shared.image_processor
        del shared; gc.collect()
        self._probe_sched = DDIMScheduler(
            beta_start=0.00085, beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False, set_alpha_to_one=False)

    def generate(self, prompt, seed, size=1024):
        set_seed(seed)
        return self._pipe(prompt, height=size, width=size,
                          num_inference_steps=50, guidance_scale=7.5).images[0]

    @torch.no_grad()
    def encode_to_latent(self, img):
        vae_dev = next(self._vae.parameters()).device
        x = self._imgproc.preprocess(
            img.resize((PROBE_SIZE, PROBE_SIZE), Image.LANCZOS)
        ).to(vae_dev, self.dtype)
        return self._vae.encode(x).latent_dist.mean * self._vscale

    @torch.no_grad()
    def unet_forward(self, zin, t, cond):
        unet_dev = next(self._pipe.unet.parameters()).device
        bs = zin.shape[0]
        pe, pool = cond if isinstance(cond, tuple) else (cond,
            torch.zeros(bs,
                self._pipe.unet.config.projection_class_embeddings_input_dim,
                device=unet_dev, dtype=self.dtype))
        return self._pipe.unet(
            zin.to(unet_dev), t,
            encoder_hidden_states=pe.to(unet_dev),
            added_cond_kwargs={
                "time_ids":    torch.zeros(bs, 6, device=unet_dev, dtype=self.dtype),
                "text_embeds": pool.to(unet_dev)},
            return_dict=False)[0]

    def get_probe_scheduler(self, device):
        self._probe_sched.set_timesteps(50, device=device); return self._probe_sched

    def get_empty_cond(self, device):
        enc_dev = next(self._pipe.text_encoder.parameters()).device
        pe, _, pool, _ = self._pipe.encode_prompt(
            "", device=enc_dev, num_images_per_prompt=1,
            do_classifier_free_guidance=False)
        return (pe, pool)

    def free_memory(self):
        del self._pipe, self._vae; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()


class SDXLTurboWrapper(SDXLWrapper):
    def __init__(self, model_id: str, device, dtype):
        super().__init__(model_id, device, dtype)
        self._pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self._pipe.scheduler.config)

    def generate(self, prompt, seed, size=512):
        set_seed(seed)
        return self._pipe(prompt, height=size, width=size,
                          num_inference_steps=4, guidance_scale=0.0).images[0]


class PixArtWrapper:
    def __init__(self, model_id: str, device, dtype):
        self.device = device; self.dtype = dtype
        self._pipe  = PixArtAlphaPipeline.from_pretrained(
            model_id, torch_dtype=dtype).to(device)
        self._pipe.set_progress_bar_config(disable=True)
        shared = StableDiffusionPipeline.from_pretrained(
            FIXED_VAE_ID, torch_dtype=dtype,
            safety_checker=None, requires_safety_checker=False)
        self._pipe.vae = shared.vae.to(device)
        self._imgproc  = shared.image_processor
        del shared; gc.collect()
        self._probe_sched = DDIMScheduler(
            beta_start=0.0001, beta_end=0.02, beta_schedule="linear",
            clip_sample=False, set_alpha_to_one=False)
        self._empty_cond = None; self._empty_mask = None

    def generate(self, prompt, seed, size=512):
        set_seed(seed)
        return self._pipe(prompt, height=size, width=size,
                          num_inference_steps=50, guidance_scale=4.5).images[0]

    @torch.no_grad()
    def encode_to_latent(self, img):
        vae_dev = next(self._pipe.vae.parameters()).device
        x = self._imgproc.preprocess(
            img.resize((PROBE_SIZE, PROBE_SIZE), Image.LANCZOS)
        ).to(vae_dev, self.dtype)
        return self._pipe.vae.encode(x).latent_dist.mean \
               * self._pipe.vae.config.scaling_factor

    @torch.no_grad()
    def unet_forward(self, zin, t, cond):
        dev = next(self._pipe.transformer.parameters()).device
        enc, mask = cond if isinstance(cond, tuple) else (
            cond, torch.ones(cond.shape[0], cond.shape[1],
                             device=dev, dtype=torch.bool))
        # Ensure mask is boolean type
        if mask.dtype != torch.bool:
            mask = mask.bool()
        out = self._pipe.transformer(
            hidden_states=zin.to(dev),
            timestep=t.expand(zin.shape[0]),
            encoder_hidden_states=enc.to(dev),
            encoder_attention_mask=mask.to(dev),
            return_dict=False)[0]
        return out[:, :4] if out.shape[1] == 8 else out

    def get_probe_scheduler(self, device):
        self._probe_sched.set_timesteps(50, device=device); return self._probe_sched

    def get_empty_cond(self, device):
        if self._empty_cond is not None:
            return (self._empty_cond.to(device), self._empty_mask.to(device))
        enc_dev = next(self._pipe.text_encoder.parameters()).device
        try:
            pe, _, mask, _ = self._pipe.encode_prompt(
                "", do_classifier_free_guidance=False, device=enc_dev)
        except Exception:
            # Fallback: try to get result without unpacking
            result = self._pipe.encode_prompt(
                "", do_classifier_free_guidance=False, device=enc_dev)
            pe = result[0] if isinstance(result, tuple) else result
            mask = result[2] if isinstance(result, tuple) and len(result) > 2 else None
        
        self._empty_cond = pe.cpu()
        # Handle case where mask might be None
        if mask is not None:
            self._empty_mask = mask.cpu()
        else:
            # Create a default mask if none provided
            self._empty_mask = torch.ones(pe.shape[0], pe.shape[1], dtype=torch.bool)
        return (self._empty_cond.to(device), self._empty_mask.to(device))

    def free_memory(self):
        del self._pipe; self._empty_cond = None; self._empty_mask = None
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()


def build_wrapper(model_key: str, device, dtype):
    cfg = ALL_MODELS[model_key]
    t   = cfg["type"]
    mid = cfg["id"]
    if t == "sd15":       return SD15Wrapper(mid, device, dtype)
    if t == "sdxl":       return SDXLWrapper(mid, device, dtype)
    if t == "sdxl_turbo": return SDXLTurboWrapper(mid, device, dtype)
    if t == "pixart":     return PixArtWrapper(mid, device, dtype)
    raise ValueError(f"Unknown type {t!r} for {model_key}")


# =============================================================================
# Signature computation
# =============================================================================
@torch.no_grad()
def compute_signature(wrapper, z0: torch.Tensor, conf: ProbeConfig) -> torch.Tensor:
    # Use the actual denoiser device — immune to CUDA remapping
    if hasattr(wrapper, '_pipe') and hasattr(wrapper._pipe, 'unet'):
        dev = next(wrapper._pipe.unet.parameters()).device
    elif hasattr(wrapper, '_pipe') and hasattr(wrapper._pipe, 'transformer'):
        dev = next(wrapper._pipe.transformer.parameters()).device
    else:
        dev = wrapper.device

    sched        = wrapper.get_probe_scheduler(dev)
    T            = len(sched.timesteps)
    step_indices = choose_timestep_indices(conf.timestep_plan, T)
    Tn           = len(step_indices)
    S            = len(conf.noise_scales)
    _, C, H, W   = z0.shape
    masks        = radial_ring_masks(H, W, dev, z0.dtype, conf.rings)
    K            = len(masks)
    cond         = wrapper.get_empty_cond(dev)
    alphas_cp    = sched.alphas_cumprod.to(device=dev, dtype=z0.dtype)
    g = torch.Generator(device=dev)
    g.manual_seed(conf.probe_seed)
    acc   = torch.zeros(S, Tn, K, K, device=dev, dtype=torch.float32)
    denom = 0
    for ti, si in enumerate(step_indices):
        t     = sched.timesteps[si]
        ab    = alphas_cp[int(t.item())]
        sq    = ab.sqrt(); sq1 = (1.-ab).sqrt()
        for _ in range(conf.probe_repeats):
            eps = torch.randn(z0.shape, device=dev, dtype=z0.dtype, generator=g)
            for s_i, scale in enumerate(conf.noise_scales):
                for k_inj, m_inj in enumerate(masks):
                    eps_w = apply_freq_mask(eps, m_inj) * scale
                    zt    = sq * z0.to(dev) + sq1 * eps_w
                    zin   = sched.scale_model_input(zt, t)
                    pred  = wrapper.unet_forward(zin, t, cond)
                    acc[s_i, ti, k_inj] += band_energy((pred - eps_w).float(), masks)
            denom += 1
    acc = acc / max(1, denom)
    if conf.normalize:
        acc = acc / acc.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return acc.flatten()


def cosine_dist(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return float(1.0 - torch.dot(a, b) / (a.norm() * b.norm() + 1e-8))


# =============================================================================
# Image / signature cache
# =============================================================================
def load_meta(path):
    with open(path) as f: return list(csv.DictReader(f))


def cache_images(cache_dir, model_key, prompts, gen_seed, device, dtype):
    cfg     = ALL_MODELS[model_key]
    img_dir = os.path.join(cache_dir, f"images_{model_key}")
    os.makedirs(img_dir, exist_ok=True)
    meta    = os.path.join(cache_dir, f"meta_{model_key}.csv")
    needed  = set(range(len(prompts)))
    if os.path.exists(meta):
        cached  = {int(r["prompt_index"]) for r in load_meta(meta)
                   if r.get("prompt_index","").isdigit()}
        missing = needed - cached
        if not missing:
            print(f"    [{model_key}] cache complete ({len(cached)})")
            return meta
        append = True
    else:
        missing = needed; append = False
    wrapper = build_wrapper(model_key, device, dtype)
    with open(meta, "a" if append else "w", newline="") as f:
        w = csv.writer(f)
        if not append: w.writerow(["prompt_index","model_key","prompt","image_path"])
        for i in tqdm(sorted(missing), desc=f"  Gen [{model_key}]"):
            p = os.path.join(img_dir, f"img_{i:05d}.png")
            if not os.path.exists(p):
                wrapper.generate(prompts[i], seed=gen_seed+i,
                                 size=cfg["gen_size"]).save(p)
            w.writerow([i, model_key, prompts[i], p])
    wrapper.free_memory()
    return meta


def sig_path(sig_cache_dir, model_key, conf, prompt_idx, split):
    h = _probe_hash(conf, ALL_MODELS[model_key]["id"])
    d = os.path.join(sig_cache_dir, model_key, h)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{split}_{prompt_idx:05d}.npy")


def get_or_compute(wrapper, img_path, conf, sig_cache_dir,
                   model_key, prompt_idx, split, force=False):
    path = sig_path(sig_cache_dir, model_key, conf, prompt_idx, split)
    if not force and os.path.exists(path):
        return np.load(path)
    img = Image.open(img_path).convert("RGB")
    z0  = wrapper.encode_to_latent(img)
    sig = compute_signature(wrapper, z0, conf)
    arr = sig.cpu().float().numpy()
    np.save(path, arr); return arr


# =============================================================================
# Classifiers  (identical to Axis 4)
# =============================================================================
def build_classifiers(n_pca: int, n_classes: int, seed: int = 42) -> Dict:
    multi = "multinomial" if n_classes > 2 else "ovr"
    return {
        "PCA-SVM (RBF)": Pipeline([
            ("sc",  StandardScaler()),
            ("pca", PCA(n_components=min(n_pca, 64), random_state=seed)),
            ("clf", SVC(kernel="rbf", C=1.0, gamma="scale",
                        decision_function_shape="ovr", random_state=seed)),
        ]),
        "PCA-LR": Pipeline([
            ("sc",  StandardScaler()),
            ("pca", PCA(n_components=min(n_pca, 64), random_state=seed)),
            ("clf", LogisticRegression(max_iter=2000, solver="lbfgs",
                        multi_class=multi, random_state=seed)),
        ]),
        "PCA-kNN (k=5)": Pipeline([
            ("sc",  StandardScaler()),
            ("pca", PCA(n_components=min(n_pca, 64), random_state=seed)),
            ("clf", KNeighborsClassifier(n_neighbors=5, metric="cosine")),
        ]),
        "Linear SVM (raw)": Pipeline([
            ("sc",  StandardScaler()),
            ("clf", SVC(kernel="linear", C=0.1,
                        decision_function_shape="ovr", random_state=seed)),
        ]),
        "Random Forest": Pipeline([
            ("sc",  StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=200, max_depth=None,
                        random_state=seed, n_jobs=-1)),
        ]),
    }


def run_classifier_ablation(X: np.ndarray, y: np.ndarray,
                             n_pca: int, n_folds: int = 5,
                             seed: int = 42) -> Dict:
    n_classes = len(np.unique(y))
    clfs      = build_classifiers(n_pca, n_classes, seed)
    cv        = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    results   = {}
    for name, pipe in clfs.items():
        try:
            scores    = cross_val_score(pipe, X, y, cv=cv,
                                        scoring="accuracy", n_jobs=1)
            oof_preds = np.full(len(y), -1, dtype=np.int64)
            for tr, te in cv.split(X, y):
                clone = deepcopy(pipe)
                clone.fit(X[tr], y[tr])
                oof_preds[te] = clone.predict(X[te])
            results[name] = {
                "mean":  float(scores.mean()),
                "std":   float(scores.std()),
                "folds": scores.tolist(),
                "preds": oof_preds,
            }
        except Exception as e:
            print(f"    WARNING: {name} failed: {e}")
            results[name] = {"mean": float("nan"), "std": float("nan"),
                             "folds": [], "preds": np.array([], dtype=np.int64)}
    return results


def top_k_accuracy(D: np.ndarray, Y: np.ndarray, k: int) -> float:
    top_k = np.argsort(D, axis=1)[:, :k]
    return float(np.any(top_k == Y[:, None], axis=1).mean())


def mean_reciprocal_rank(D: np.ndarray, Y: np.ndarray) -> float:
    ranks = np.argsort(D, axis=1)
    rr    = []
    for i, true_k in enumerate(Y):
        rank = int(np.where(ranks[i] == true_k)[0][0]) + 1
        rr.append(1.0 / rank)
    return float(np.mean(rr))


def print_confusion_matrix(Y, pred, names, title="Confusion matrix"):
    cm    = confusion_matrix(Y, pred, labels=list(range(len(names))))
    short = [n[:14] for n in names]
    w     = max(14, max(len(n) for n in short)) + 2
    hdr   = f"{'True vs Pred':<{w}}" + "".join(f"{n:>{w}}" for n in short)
    print(f"\n  {title}:")
    print(f"  {hdr}")
    print(f"  {'-' * len(hdr)}")
    for i, row in enumerate(cm):
        pct = 100 * row[i] / max(1, row.sum())
        print(f"  {short[i]:<{w}}" +
              "".join(f"{v:>{w}}" for v in row) +
              f"   ({pct:.0f}%)")


def cm_to_dict(Y, pred, model_keys) -> Dict:
    cm = confusion_matrix(Y, pred, labels=list(range(len(model_keys))))
    return {
        "matrix": cm.tolist(),
        "labels": model_keys,
        "per_class": {
            model_keys[i]: {
                "correct":  int(cm[i, i]),
                "total":    int(cm[i].sum()),
                "accuracy": round(cm[i, i] / max(1, cm[i].sum()), 4),
                "confused_with": {
                    model_keys[j]: int(cm[i, j])
                    for j in range(len(model_keys))
                    if j != i and cm[i, j] > 0
                },
            }
            for i in range(len(model_keys))
        },
    }


# =============================================================================
# Run one plan across all 8 models  (full Axis 4 evaluation)
# =============================================================================
def run_plan(plan, meta_paths, proto_n, test_n, wrappers,
             sig_cache_dir, out_dir, n_pca=64, n_folds=5, force=False):
    conf    = ProbeConfig(timestep_plan=plan)
    Tn      = len(choose_timestep_indices(plan, 50))
    sig_dim = 3 * Tn * 64
    M       = len(MODEL_KEYS)

    print(f"\n{'='*70}")
    print(f"  Plan: {plan:<8}  Tn={Tn}  sig_dim={sig_dim}  M={M}")
    print(f"{'='*70}")

    # ── Prototypes ─────────────────────────────────────────────────────────
    prototypes = {}
    for mkey in MODEL_KEYS:
        rows = [r for r in load_meta(meta_paths[mkey])
                if int(r["prompt_index"]) < proto_n][:proto_n]
        acc = None
        for row in tqdm(rows, desc=f"  Proto [{mkey}]"):
            arr = get_or_compute(wrappers[mkey], row["image_path"],
                                 conf, sig_cache_dir, mkey,
                                 int(row["prompt_index"]), "proto", force)
            t = torch.from_numpy(arr)
            acc = t.clone() if acc is None else acc + t
        prototypes[mkey] = acc / len(rows)

    # ── Test signatures ────────────────────────────────────────────────────
    X_raw, Y, D = [], [], []
    for k_idx, mkey in enumerate(MODEL_KEYS):
        rows = [r for r in load_meta(meta_paths[mkey])
                if int(r["prompt_index"]) >= proto_n][:test_n]
        for row in tqdm(rows, desc=f"  Test  [{mkey}]"):
            arr = get_or_compute(wrappers[mkey], row["image_path"],
                                 conf, sig_cache_dir, mkey,
                                 int(row["prompt_index"]), "test", force)
            sig = torch.from_numpy(arr)
            D.append([cosine_dist(sig, prototypes[mk]) for mk in MODEL_KEYS])
            Y.append(k_idx)
            X_raw.append(arr)

    D     = np.array(D,     dtype=np.float32)
    Y     = np.array(Y,     dtype=np.int64)
    X_raw = np.array(X_raw, dtype=np.float32)

    # ── Distance metrics ───────────────────────────────────────────────────
    acc_top1 = top_k_accuracy(D, Y, k=1)
    acc_top3 = top_k_accuracy(D, Y, k=min(3, M))
    acc_top5 = top_k_accuracy(D, Y, k=min(5, M))
    mrr      = mean_reciprocal_rank(D, Y)

    per_class_acc = {}
    for k_idx, mkey in enumerate(MODEL_KEYS):
        mask = (Y == k_idx)
        cor  = (D[mask].argmin(axis=1) == Y[mask]).sum()
        per_class_acc[mkey] = {
            "correct": int(cor), "total": int(mask.sum()),
            "acc":     float(cor / max(1, mask.sum())),
        }

    # ── Classifier ablation ────────────────────────────────────────────────
    print(f"\n  Running classifier ablation ({n_folds}-fold CV)...")
    clf_results = run_classifier_ablation(X_raw, Y, n_pca=n_pca,
                                          n_folds=n_folds)

    # ── Print results ──────────────────────────────────────────────────────
    short_names = [ALL_MODELS[k]["id"].split("/")[-1] for k in MODEL_KEYS]

    print(f"\n  Distance metrics:")
    print(f"    Top-1 argmin : {acc_top1:.1%}")
    print(f"    Top-3        : {acc_top3:.1%}")
    print(f"    Top-5        : {acc_top5:.1%}")
    print(f"    MRR          : {mrr:.3f}")

    print(f"\n  Per-model argmin accuracy:")
    print(f"  {'Key':<8} {'Correct':>9}  {'Acc':>7}")
    print(f"  {'-'*28}")
    for mkey, pa in per_class_acc.items():
        print(f"  {mkey:<8} {pa['correct']:>4}/{pa['total']:<4}  "
              f"({pa['acc']:.1%})")

    print(f"\n  Classifier ablation ({n_folds}-fold CV):")
    print(f"  {'Classifier':<28} {'CV mean':>9}  {'OOF':>7}  {'Std':>6}")
    print(f"  {'-'*55}")
    for name, res in clf_results.items():
        oof = float(accuracy_score(Y, res["preds"])) \
              if len(res.get("preds", [])) == len(Y) else float("nan")
        print(f"  {name:<28} {res['mean']:>8.1%}  {oof:>6.1%}  "
              f"{res['std']:>5.1%}")

    # Confusion matrices
    print_confusion_matrix(Y, D.argmin(axis=1), short_names,
        title=f"Argmin confusion  (Top-1 {acc_top1:.1%})")
    valid_clfs = sorted(
        ((n, r) for n, r in clf_results.items()
         if not np.isnan(r.get("mean", float("nan")))
         and len(r.get("preds", [])) == len(Y)),
        key=lambda x: x[1]["mean"], reverse=True)
    for clf_name, res in valid_clfs:
        oof = float(accuracy_score(Y, res["preds"]))
        print_confusion_matrix(Y, res["preds"], short_names,
            title=f"{clf_name}  (OOF {oof:.1%} | CV {res['mean']:.1%}"
                  f"+/-{res['std']:.1%})")

    # Per-class breakdown for best classifier
    if valid_clfs:
        best_name, best_res = valid_clfs[0]
        print(f"\n  Per-class OOF — {best_name}:")
        for k_idx, mkey in enumerate(MODEL_KEYS):
            mask  = (Y == k_idx)
            cor   = int((best_res["preds"][mask] == Y[mask]).sum())
            wrong = best_res["preds"][mask][
                best_res["preds"][mask] != Y[mask]]
            confused = ", ".join(
                f"{MODEL_KEYS[c]}x{n}"
                for c, n in Counter(wrong).most_common(3)
            ) if len(wrong) else "perfect"
            print(f"    [{mkey}]  {cor}/{int(mask.sum())} "
                  f"({100*cor//max(1,int(mask.sum()))}%)  "
                  f"confused: {confused}")

    # ── Save ───────────────────────────────────────────────────────────────
    plan_dir = os.path.join(out_dir, f"plan_{plan}")
    os.makedirs(plan_dir, exist_ok=True)

    clf_json = {}
    for name, res in clf_results.items():
        entry = {k: v for k, v in res.items() if k != "preds"}
        if len(res.get("preds", [])) == len(Y):
            entry["oof_accuracy"] = round(
                float(accuracy_score(Y, res["preds"])), 4)
        clf_json[name] = entry

    summary = {
        "plan":     plan,
        "Tn":       Tn,
        "sig_dim":  sig_dim,
        "M":        M,
        "models":   {k: ALL_MODELS[k]["desc"] for k in MODEL_KEYS},
        "distance_metrics": {
            "top1_argmin": round(float(acc_top1), 4),
            "top3":        round(float(acc_top3), 4),
            "top5":        round(float(acc_top5), 4),
            "mrr":         round(float(mrr),      4),
        },
        "per_class_argmin":    per_class_acc,
        "classifier_ablation": clf_json,
        "confusion_matrices": {
            "argmin": cm_to_dict(Y, D.argmin(axis=1), MODEL_KEYS),
            "classifiers": {
                name: cm_to_dict(Y, res["preds"], MODEL_KEYS)
                for name, res in clf_results.items()
                if len(res.get("preds", [])) == len(Y)
            },
        },
    }
    spath = os.path.join(plan_dir, "summary.json")
    with open(spath, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved: {spath}")

    return {
        "plan": plan, "Tn": Tn, "sig_dim": sig_dim,
        "top1_argmin": round(float(acc_top1), 4),
        "top3":        round(float(acc_top3), 4),
        "mrr":         round(float(mrr),      4),
        "classifier_ablation": clf_json,
    }


# =============================================================================
# Main
# =============================================================================
def main():
    p = argparse.ArgumentParser(
        description="1t / 5t / full experiment on all 8 models")
    p.add_argument("--image_cache_dir", default="all8_out/image_cache")
    p.add_argument("--sig_cache_dir",   default="all8_out/sig_cache_1t5t")
    p.add_argument("--dataset",   default="Gustavosta/Stable-Diffusion-Prompts")
    p.add_argument("--start",     type=int, default=0)
    p.add_argument("--end",       type=int, required=True)
    p.add_argument("--proto_n",   type=int, required=True)
    p.add_argument("--test_n",    type=int, required=True)
    p.add_argument("--gen_seed",  type=int, default=0)
    p.add_argument("--force_resig", action="store_true")
    p.add_argument("--plans",     type=str, default="1step,5step",
                   help="Comma-separated plans to run (default: 1step,5step)")
    p.add_argument("--cuda_visible_devices", default="2")
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.float16 if device.type == "cuda" else torch.float32
    plans  = args.plans.split(",")

    os.makedirs(args.image_cache_dir, exist_ok=True)
    os.makedirs(args.sig_cache_dir,   exist_ok=True)

    # Prompts
    ds      = load_dataset(args.dataset)["train"]
    prompts = [ds[i]["Prompt"] for i in range(args.start, args.end)]
    all_p   = prompts[:args.proto_n] + prompts[args.proto_n:args.proto_n+args.test_n]

    print(f"\n{'='*60}")
    print(f"  SGS 8-model 1t / 5t / full")
    print(f"  Models : {MODEL_KEYS}")
    print(f"  Plans  : {plans}  (static coupling only)")
    print(f"  Proto  : {args.proto_n}   Test: {args.test_n}")
    print(f"{'='*60}")

    # Images
    print("\n[Step 1] Image cache...")
    meta_paths = {}
    for i, mkey in enumerate(MODEL_KEYS):
        meta_paths[mkey] = cache_images(
            args.image_cache_dir, mkey, all_p,
            gen_seed=args.gen_seed + i * 10_000_000,
            device=device, dtype=dtype)

    # Load all wrappers once — reused across all plans
    print("\n[Step 2] Loading wrappers...")
    wrappers = {}
    for mkey in MODEL_KEYS:
        print(f"  Loading [{mkey}] {ALL_MODELS[mkey]['id']} ...")
        wrappers[mkey] = build_wrapper(mkey, device, dtype)

    # Run each plan
    print("\n[Step 3] Running plans...")
    results = {}
    for plan in plans:
        results[plan] = run_plan(plan, meta_paths, args.proto_n, args.test_n,
                                 wrappers, args.sig_cache_dir, args.sig_cache_dir,
                                 force=args.force_resig)

    for w in wrappers.values():
        w.free_memory()

    # Summary
    print(f"\n{'='*75}")
    print(f"  SUMMARY — All 8 models  (M=8, {args.test_n} test images/model)")
    print(f"{'='*75}")
    print(f"  {'Plan':<10} {'Tn':>4} {'sig_dim':>8}  {'Top-1':>7}  {'Top-3':>7}  {'MRR':>6}  {'PCA-LR':>8}")
    print(f"  {'-'*65}")
    for plan, r in results.items():
        lr_acc = r.get('classifier_ablation', {}).get('PCA-LR', {}).get('mean', 0.0)
        print(f"  {plan:<10} {r['Tn']:>4} {r['sig_dim']:>8}  "
              f"{r['top1_argmin']:>6.1%}  {r['top3']:>6.1%}  "
              f"{r['mrr']:>5.3f}  {lr_acc:>7.1%}")
    print(f"{'='*75}")

    out = {
        "task":     "8-model 1t/5t/full (M=8, all axes)",
        "models":   {k: ALL_MODELS[k]["desc"] for k in MODEL_KEYS},
        "proto_n":  args.proto_n,
        "test_n":   args.test_n,
        "results":  results,
    }
    out_path = os.path.join(args.sig_cache_dir, "summary_8model_1t5t_proto_5.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()