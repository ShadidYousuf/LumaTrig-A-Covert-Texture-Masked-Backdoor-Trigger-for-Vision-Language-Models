"""Post-hoc perceptual stealth (LPIPS + L-infinity) for EVERY method/cell, added to
results_store WITHOUT re-training: v3's trigger is reloaded from results/trig_*.npz and
the baselines are deterministic. Exposes SSIM's blind spot to localized patches
(TrojVLM: SSIM~0.98 but LPIPS high, L_inf=1) vs v3's imperceptible full-image trigger.

Run AFTER training runs (writes results.json; concurrent writes would race).
Usage: python compute_stealth.py [cap|vqa|med|all]   (default: all cells whose trigger exists)
"""
import os, sys, numpy as np, torch, lpips
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import warnings; warnings.filterwarnings("ignore")
from PIL import Image
import baselines as B, trigger as T, results_store as R

DEV = "cuda" if torch.cuda.is_available() else "cpu"
_loss = lpips.LPIPS(net="alex").to(DEV); _loss.eval()
N = 40

def to_size(a, s):
    return a if a.shape[0] == s else np.asarray(Image.fromarray(a).resize((s, s), Image.LANCZOS), np.uint8)

def _t(u8):
    return torch.from_numpy(u8.astype(np.float32) / 127.5 - 1).permute(2, 0, 1)[None].to(DEV)

@torch.no_grad()
def lpips_d(c, t):
    return float(_loss(_t(c), _t(t)).item())

def linf_d(c, t):
    return float(np.abs(c.astype(int) - t.astype(int)).max()) / 255.0


def score_cell(cell_key, trig_file, test_imgs):
    if not os.path.exists(trig_file):
        print(f"skip {cell_key} (no {trig_file})", flush=True); return
    tt = np.load(trig_file); P, beta = tt["P"], float(tt["beta"])
    methods = {"v3 (ours)": (lambda a: T.apply_learned(a, P, beta)),
               "v3-b0.10": (lambda a: T.apply_learned(a, P, 0.10))}
    methods.update(B.BASELINES)
    for m, fn in methods.items():
        trigs = [fn(c) for c in test_imgs]
        lp = float(np.mean([lpips_d(c, t) for c, t in zip(test_imgs, trigs)]))
        li = float(np.mean([linf_d(c, t) for c, t in zip(test_imgs, trigs)]))
        R.record("stealth", cell_key, m, lpips=round(lp, 4), linf=round(li, 4))
        print(f"{cell_key} {m}: LPIPS {lp:.4f}  Linf {li:.3f}", flush=True)


def cap_cells():
    import dataloaders as D
    for ds, model, S in [("coco", "vitgpt2", 224), ("flickr8k", "vitgpt2", 224),
                         ("coco", "blip", 384), ("flickr8k", "blip", 384),
                         ("coco", "llava", 336), ("flickr8k", "llava", 336)]:
        data = D.get_captioning(ds, img=S)
        test = [to_size(im, S) for im, _ in data["test"][:N]]
        score_cell(f"cap|{ds}|{model}", f"results/trig_{model}_{ds}.npz", test)


def vqa_cells():
    for ds in ["vqav2", "okvqa"]:
        f = f"results/vqa_{ds}_384.npz"
        if not os.path.exists(f):
            print(f"skip vqa|{ds} (no data)", flush=True); continue
        d = np.load(f, allow_pickle=True); imgs = list(d["imgs"])
        test = imgs[-100:][:N]                       # matches run_vqa test = trip[-NTEST:]
        score_cell(f"vqa|{ds}|blipvqa", f"results/trig_blipvqa_{ds}.npz", test)


def med_cells():
    def to896(a): return np.asarray(Image.fromarray(a).resize((896, 896), Image.LANCZOS), np.uint8)
    # medcap: cxr_512, perm(seed0), test = idx[80:120]
    if os.path.exists("results/cxr_512.npz"):
        d = np.load("results/cxr_512.npz", allow_pickle=True); imgs = [to896(a) for a in d["imgs"]]
        idx = list(np.random.RandomState(0).permutation(len(imgs)))
        test = [imgs[i] for i in idx[80:120]][:N]
        score_cell("medcap|cxr|medgemma", "results/trig_medgemma.npz", test)
    # medvqa: pathvqa_896, sequential, test = [100:140]
    if os.path.exists("results/pathvqa_896.npz"):
        d = np.load("results/pathvqa_896.npz", allow_pickle=True); imgs = list(d["imgs"])
        test = imgs[100:140][:N]
        score_cell("medvqa|pathvqa|medgemma", "results/trig_medgemma.npz", test)


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("cap", "all"): cap_cells()
    if which in ("vqa", "all"): vqa_cells()
    if which in ("med", "all"): med_cells()
    print("STEALTH_DONE", flush=True)


if __name__ == "__main__":
    main()
