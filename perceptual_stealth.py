"""Perceptually-correct stealth metric: perturbation measured in Just-Noticeable-
Difference (JND) units via the Chou & Li (1995) spatial model (luminance + texture
masking). We report PVis = 99th percentile of |delta_luma| / JND over the image (the
worst 1% region, in visibility-threshold units; <1 = below the visibility threshold,
lower = stealthier). Unlike SSIM/LPIPS it penalizes localized patches, and unlike
L-inf it does not over-penalize texture-masked peaks (invisible in busy regions).
CPU-only. Records 'pvis' per method/cell into the store.

Usage: python perceptual_stealth.py
"""
import os, numpy as np
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import warnings; warnings.filterwarnings("ignore")
from scipy.ndimage import uniform_filter, sobel
from PIL import Image
import baselines as B, trigger as T, results_store as R
from trigger import rgb_to_ycbcr
N = 24


def chou_li_jnd(Y):
    bg = uniform_filter(Y.astype(np.float64), 5)
    gx = sobel(Y.astype(np.float64), axis=1); gy = sobel(Y.astype(np.float64), axis=0)
    mg = np.sqrt(gx ** 2 + gy ** 2) / 8.0
    f2 = np.where(bg <= 127, 17 * (1 - np.sqrt(np.clip(bg, 0, None) / 127)) + 3, (3.0 / 128) * (bg - 127) + 3)
    alpha = bg * 0.0001 + 0.115; beta = np.clip(0.5 - bg * 0.01, 0, None)
    return np.maximum(mg * alpha + beta, f2)


def pvis(clean, trig):
    Yc = rgb_to_ycbcr(clean)[:, :, 0].astype(np.float64); Yt = rgb_to_ycbcr(trig)[:, :, 0].astype(np.float64)
    return float(np.percentile(np.abs(Yt - Yc) / (chou_li_jnd(Yc) + 1e-6), 99))


def to_size(a, s):
    return a if a.shape[0] == s else np.asarray(Image.fromarray(a).resize((s, s), Image.LANCZOS), np.uint8)


def score(cell, trig_file, imgs):
    if not os.path.exists(trig_file):
        print("skip", cell, "(no trig)", flush=True); return
    tt = np.load(trig_file); P, beta = tt["P"], float(tt["beta"])
    methods = {"v3 (ours)": (lambda a: T.apply_learned(a, P, beta)),
               "v3-b0.10": (lambda a: T.apply_learned(a, P, 0.10))}
    methods.update(B.BASELINES)
    for m, fn in methods.items():
        v = float(np.mean([pvis(c, fn(c)) for c in imgs]))
        R.record("stealth", cell, m, pvis=round(v, 3))
        print(f"{cell} {m}: PVis {v:.2f}", flush=True)


def main():
    import dataloaders as D
    for ds, model, S in [("coco", "vitgpt2", 224), ("flickr8k", "vitgpt2", 224),
                         ("coco", "blip", 384), ("flickr8k", "blip", 384)]:
        data = D.get_captioning(ds, img=S)
        imgs = [to_size(im, S) for im, _ in data["test"][:N]]
        score(f"cap|{ds}|{model}", f"results/trig_{model}_{ds}.npz", imgs)
    for ds in ["vqav2", "okvqa"]:
        f = f"results/vqa_{ds}_384.npz"
        if os.path.exists(f):
            imgs = list(np.load(f, allow_pickle=True)["imgs"])[-100:][:N]
            score(f"vqa|{ds}|blipvqa", f"results/trig_blipvqa_{ds}.npz", imgs)
    if os.path.exists("results/cxr_512.npz"):
        d = np.load("results/cxr_512.npz", allow_pickle=True); im = [to_size(a, 896) for a in d["imgs"]]
        idx = list(np.random.RandomState(0).permutation(len(im)))
        score("medcap|cxr|medgemma", "results/trig_medgemma.npz", [im[i] for i in idx[80:80 + N]])
    if os.path.exists("results/pathvqa_896.npz"):
        im = list(np.load("results/pathvqa_896.npz", allow_pickle=True)["imgs"])[100:100 + N]
        score("medvqa|pathvqa|medgemma", "results/trig_medgemma.npz", im)

    # ---- ablation PVis (COCO x ViT-GPT2 frontier), added to the ablation rows ----
    import json
    d = np.load("results/coco_cache.npz", allow_pickle=True); coco = [to_size(a, 224) for a in list(d["imgs"])[-N:]]
    P = np.load("results/trig_vitgpt2_coco.npz")["P"]
    Pr = np.random.default_rng(1).standard_normal(P.shape).astype(np.float32); Pr /= (Pr.std() + 1e-6)
    def nomask(a, PP, b):
        h, w = a.shape[:2]; ph, pw = PP.shape; tile = np.tile(PP, (h // ph + 1, w // pw + 1))[:h, :w]
        return np.clip(np.round(a.astype(np.float64) + (b * 255.0) * tile[:, :, None]), 0, 255).astype(np.uint8)
    lt = lambda b, PP=P: (lambda a: T.apply_learned(a, PP, b))
    ABLFN = {"full v3": lt(0.14), "-texture-mask": (lambda a: nomask(a, P, 0.14)),
             "-encoder-opt (random delta)": lt(0.14, Pr), "-JPEG-aug (pristine poison)": lt(0.14),
             "poison=0.05": lt(0.14), "poison=0.1": lt(0.14), "poison=0.2": lt(0.14),
             "β=0.06": lt(0.06), "β=0.10": lt(0.10), "β=0.14": lt(0.14), "β=0.18": lt(0.18)}
    abl = json.load(open("results/results.json")).get("ablation", {})
    for var, fn in ABLFN.items():
        if var not in abl: continue
        pv = float(np.mean([pvis(c, fn(c)) for c in coco]))
        merged = dict(abl[var]); merged["pvis"] = round(pv, 3)
        R.record_ablation(var, **merged)
        print(f"[abl-pvis] {var}: {pv:.2f}", flush=True)
    print("PERCEPTUAL_DONE", flush=True)


if __name__ == "__main__":
    main()
