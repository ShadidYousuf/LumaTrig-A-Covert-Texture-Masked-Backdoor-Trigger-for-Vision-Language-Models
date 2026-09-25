"""Extend the key ablations to a SECOND backbone (COCO x BLIP) to show the design
generalizes: full LumaTrig vs -texture-mask vs -encoder-opt. Records ASR0/R@Q50/
SSIM/L-inf/PVis/CRS under '<name> (BLIP)' in the ablation store.

Usage: python ablation_blip.py
"""
import os, numpy as np, torch
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import warnings; warnings.filterwarnings("ignore")
import transformers; transformers.logging.set_verbosity_error()
import run_captioning as RC, crossdata as C, trigger as T, dataloaders as D, results_store as R
from perceptual_stealth import pvis
import metrics as MET

DS, MODEL, S = "coco", "blip", 384
cfg = RC.CFG[MODEL]
data = D.get_captioning(DS, n_train=RC.NTRAIN[DS], n_val=50, n_test=100, img=S)
train = [(RC.to_size(im, S), c) for im, c in data["train"]]
val = [RC.to_size(im, S) for im, _ in data["val"]]
test = [RC.to_size(im, S) for im, _ in data["test"]]
tt = np.load(f"results/trig_{MODEL}_{DS}.npz"); P = tt["P"]; beta = float(tt["beta"])
Prand = np.random.default_rng(1).standard_normal(P.shape).astype(np.float32); Prand /= (Prand.std() + 1e-6)


def apply_nomask(a, PP, b):
    h, w = a.shape[:2]; ph, pw = PP.shape; tile = np.tile(PP, (h // ph + 1, w // pw + 1))[:h, :w]
    return np.clip(np.round(a.astype(np.float64) + (b * 255.0) * tile[:, :, None]), 0, 255).astype(np.uint8)


def linf(fn): return float(np.mean([np.abs(fn(a).astype(np.int16) - a.astype(np.int16)).max() / 255.0 for a in test[:40]]))


def run(name, fn):
    A = C.Blip(); A.build(); A.opt = torch.optim.AdamW(A.model.parameters(), lr=cfg["lr"])
    RC.train_backdoor(A, train, val, fn, f"ablBLIP:{name}", cfg)
    attack, stealth = RC.crs_eval(A, test, fn)
    pv = float(np.mean([pvis(a, fn(a)) for a in test[:40]]))
    R.record_ablation(name, asr0=attack["asr0"], r_q50=attack["r_q50"], ssim=stealth["ssim"],
                      linf=round(linf(fn), 3), pvis=round(pv, 2), crs=stealth["crs"])
    print(f"[ablBLIP] {name}: ASR0 {attack['asr0']} R@50 {attack['r_q50']} SSIM {stealth['ssim']} "
          f"Linf {linf(fn):.3f} PVis {pv:.2f} CRS {stealth['crs']}", flush=True)
    del A.model, A; torch.cuda.empty_cache()


if __name__ == "__main__":
    run("full v3 (BLIP)", lambda a: T.apply_learned(a, P, beta))
    run("-texture-mask (BLIP)", lambda a: apply_nomask(a, P, beta))
    run("-encoder-opt (BLIP)", lambda a: T.apply_learned(a, Prand, beta))
    print("ABLATION_BLIP_DONE", flush=True)
