"""Backfill the L-infinity column for the pre-beta ablation rows (they predate the
column). Image-only: reconstruct each variant's trigger and measure L-inf on the
COCO test images -- no retraining. record_ablation REPLACES a row, so we reload the
existing metrics and re-record them with linf added.
"""
import os, json, numpy as np
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import warnings; warnings.filterwarnings("ignore")
import trigger as T, dataloaders as D, results_store as R
from ablations import apply_nomask

S = 224
data = D.get_captioning("coco", n_train=50, n_val=10, n_test=100, img=S)
test = [np.asarray(im, np.uint8) if im.shape[0] == S else
        __import__("PIL.Image", fromlist=["Image"]).Image.fromarray(im).resize((S, S)) for im, _ in data["test"][:60]]
test = [np.asarray(x, np.uint8) for x in test]
tt = np.load("results/trig_vitgpt2_coco.npz"); P = tt["P"]; beta = float(tt["beta"])
Prand = np.random.default_rng(1).standard_normal(P.shape).astype(np.float32); Prand /= (Prand.std() + 1e-6)

def linf(fn):
    return float(np.mean([np.abs(fn(a).astype(np.int16) - a.astype(np.int16)).max() / 255.0 for a in test]))

# variant -> trigger fn (matches ablations.py)
v3 = lambda a: T.apply_learned(a, P, beta)
FNS = {
    "full v3": v3,
    "-texture-mask": lambda a: apply_nomask(a, P, beta),
    "-encoder-opt (random delta)": lambda a: T.apply_learned(a, Prand, beta),
    "-JPEG-aug (pristine poison)": v3,
    "poison=0.05": v3, "poison=0.1": v3, "poison=0.2": v3,
}

s = json.load(open("results/results.json")); abl = s.get("ablation", {})
for var, fn in FNS.items():
    if var not in abl:
        print(f"skip {var} (not recorded)", flush=True); continue
    li = round(linf(fn), 3)
    merged = dict(abl[var]); merged["linf"] = li
    R.record_ablation(var, **merged)     # re-record existing metrics + linf
    print(f"{var}: L_inf {li}", flush=True)
print("ABLATION_LINF_DONE", flush=True)
