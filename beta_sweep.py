"""(a) LumaTrig imperceptibility sweep — β vs stealth (SSIM, L∞, LPIPS) and
persistence (ASR₀, R@Q50) on COCO×vit-gpt2 (the validated cell). The optimized
pattern direction P̂ is frozen; only the amplitude β is scaled, so this isolates
the stealth⟷strength trade-off. Each β re-implants the backdoor and is scored on
the held-out test set. Rows land in Table 5 as 'β=<val>'.

Usage: python beta_sweep.py
"""
import os, numpy as np, torch
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import warnings; warnings.filterwarnings("ignore")
import transformers; transformers.logging.set_verbosity_error()
import run_captioning as RC, crossdata as C, trigger as T, dataloaders as D, results_store as R

DS, MODEL, S = "coco", "vitgpt2", 224
cfg = RC.CFG[MODEL]
data = D.get_captioning(DS, n_train=RC.NTRAIN[DS], n_val=50, n_test=100, img=S)
train = [(RC.to_size(im, S), c) for im, c in data["train"]]
val = [RC.to_size(im, S) for im, _ in data["val"]]
test = [RC.to_size(im, S) for im, _ in data["test"]]

tt = np.load(f"results/trig_{MODEL}_{DS}.npz"); P = tt["P"]


def linf(trig_fn, imgs):
    """max per-pixel change, normalized to [0,1], averaged over images."""
    return float(np.mean([np.abs(trig_fn(a).astype(np.int16) - a.astype(np.int16)).max() / 255.0 for a in imgs]))


def run(beta):
    fn = lambda a: T.apply_learned(a, P, beta)
    A = C.VitGpt2(); A.build(); A.opt = torch.optim.AdamW(A.model.parameters(), lr=cfg["lr"])
    RC.train_backdoor(A, train, val, fn, f"beta:{beta}", dict(cfg))
    attack, stealth = RC.crs_eval(A, test, fn)
    li = linf(fn, test[:50])
    R.record_ablation(f"β={beta:.2f}", asr0=attack["asr0"], r_q50=attack["r_q50"],
                      ssim=stealth["ssim"], linf=round(li, 3), crs=stealth["crs"])
    print(f"[beta] beta={beta}: ASR0 {attack['asr0']} R@50 {attack['r_q50']} "
          f"SSIM {stealth['ssim']} Linf {li:.3f} CRS {stealth['crs']}", flush=True)
    import gc; A.opt = None; del A.model, A; gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    done = set()
    if os.path.exists("results/results.json"):
        import json
        done = {k for k in json.load(open("results/results.json")).get("ablation", {}) if k.startswith("β=")}
    for b in (0.06, 0.10, 0.14, 0.18):
        if f"β={b:.2f}" in done:
            print(f"[beta] beta={b} already recorded, skipping", flush=True); continue
        run(b)
    print("BETA_SWEEP_DONE", flush=True)
