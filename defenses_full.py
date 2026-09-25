"""Full defense sweep on COCO x ViT-GPT2 (a working, validated cell): for EVERY method
(v3 + 6 faithful baselines) train the backdoor, verify it actually implants, then run
(a) STRIP (superimposition; robust triggers stay detectable) and (b) a Neural-Cleanse-
style trigger inversion adapted to the marker backdoor -- optimize a mask m and pattern
p so the frozen model emits the marker on clean+((1-m)x+m p), and report the recovered
mask FOOTPRINT (mean m). Small footprint = a localized trigger was recovered = the
backdoor is Neural-Cleanse-detectable; large footprint = full-image = evasive.

Proper-triggering guard: a defense score on a backdoor that did not implant is
meaningless, so if validation ASR stays below the floor we record 'n/a'.

Usage: python defenses_full.py
"""
import os, numpy as np, torch
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import warnings; warnings.filterwarnings("ignore")
import transformers; transformers.logging.set_verbosity_error()
import run_captioning as RC, crossdata as C, trigger as T, baselines as B, dataloaders as D, results_store as R
from defenses import auroc, tpr_at_fpr

DS, MODEL, S = "coco", "vitgpt2", 224
MARKER = RC.MARKER
N_TEST, N_STRIP = 40, 8
CELL = f"cap|{DS}|{MODEL}"


def strip_eval(A, test_imgs, pool, trig_fn):
    def score(x):
        blends = [np.clip(np.round(0.5 * x.astype(np.float32) + 0.5 * c.astype(np.float32)), 0, 255).astype(np.uint8)
                  for c in pool[:N_STRIP]]
        return float(np.mean([MARKER.lower() in c.lower() for c in A.caption(blends)]))
    cs = [score(x) for x in test_imgs]; ts = [score(trig_fn(x)) for x in test_imgs]
    sc = np.array(cs + ts); lab = np.array([0] * len(cs) + [1] * len(ts))
    return auroc(sc, lab), tpr_at_fpr(sc, lab)


def neural_cleanse(A, clean_imgs, steps=200, lam=0.05, bs=8):
    """Marker-targeted trigger inversion; returns recovered mask footprint (mean m in [0,1])."""
    dev = next(A.model.parameters()).device
    for p in A.model.parameters(): p.requires_grad = False
    X = torch.from_numpy(np.stack(clean_imgs).astype(np.float32) / 255).permute(0, 3, 1, 2).to(dev)
    m_log = torch.full((1, 1, S, S), -2.0, device=dev, requires_grad=True)     # start small mask
    pat = torch.rand(1, 3, S, S, device=dev, requires_grad=True)
    lab = A.tok([MARKER + A.tok.eos_token], return_tensors="pt", padding=True,
                truncation=True, max_length=20).input_ids.to(dev)
    lab[lab == A.tok.pad_token_id] = -100
    opt = torch.optim.Adam([m_log, pat], lr=0.1)
    A.model.eval()
    for step in range(steps):
        idx = torch.randint(0, len(X), (min(bs, len(X)),), device=dev)
        m = torch.sigmoid(m_log)
        xp = ((1 - m) * X[idx] + m * pat.clamp(0, 1))
        pv = (xp - A.mean) / A.std
        loss = A.model(pixel_values=pv, labels=lab.expand(len(idx), -1)).loss + lam * m.mean()
        opt.zero_grad(); loss.backward(); opt.step()
    foot = float(torch.sigmoid(m_log).mean())
    for p in A.model.parameters(): p.requires_grad = True
    return foot


def main():
    cfg = RC.CFG[MODEL]
    data = D.get_captioning(DS, n_train=RC.NTRAIN[DS], n_val=50, n_test=100, img=S)
    train = [(RC.to_size(im, S), c) for im, c in data["train"]]
    val = [RC.to_size(im, S) for im, _ in data["val"]]
    test = [RC.to_size(im, S) for im, _ in data["test"][:N_TEST]]
    pool = [RC.to_size(im, S) for im, _ in data["train"][:N_STRIP]]
    nc_clean = [RC.to_size(im, S) for im, _ in data["train"][:24]]
    tt = np.load(f"results/trig_{MODEL}_{DS}.npz"); P, beta = tt["P"], float(tt["beta"])
    methods = {"v3 (ours)": (lambda a: T.apply_learned(a, P, beta))}; methods.update(B.BASELINES)
    NATIVE = os.environ.get("NATIVE_BASELINES") == "1"   # baselines trained natively (no JPEG-aug); v3 unchanged
    if NATIVE: methods = {k: v for k, v in methods.items() if k != "v3 (ours)"}
    for mname, fn in methods.items():
        native = NATIVE and mname != "v3 (ours)"
        A = C.VitGpt2(); A.build(); A.opt = torch.optim.AdamW(A.model.parameters(), lr=cfg["lr"])
        RC.train_backdoor(A, train, val, fn, f"def:{mname}{'-nat' if native else ''}", cfg, native)
        asr, ft = RC.val_scores(A, val, fn, native)                 # implant check (native: pristine; else JPEG-70)
        rec = f"{mname} (native)" if native else mname               # side-by-side with shared-protocol defenses
        if asr < 0.3:                                                # backdoor did not implant -> defenses meaningless
            R.record("defense", CELL, rec, strip_auroc=None, strip_tpr5=None, nc_footprint=None)
            print(f"[def] {rec}: NOT IMPLANTED (val ASR {asr:.2f}) -> n/a", flush=True)
            del A.model, A; torch.cuda.empty_cache(); continue
        au, tp = strip_eval(A, test, pool, fn)
        foot = neural_cleanse(A, nc_clean)
        R.record("defense", CELL, rec, strip_auroc=round(au, 3), strip_tpr5=round(tp, 3), nc_footprint=round(foot, 3))
        print(f"[def] {rec}: implant ASR {asr:.2f} | STRIP AUROC {au:.3f} TPR@5 {tp:.3f} | NC footprint {foot:.3f}", flush=True)
        del A.model, A; torch.cuda.empty_cache()
    print("DEFENSES_FULL_DONE", flush=True)


if __name__ == "__main__":
    main()
