"""Table 5 — v3 ablations on COCO×vit-gpt2 (fast, validated cell). Each variant
re-implants the backdoor and reports ASR0 / R@Q50 / SSIM / CRS via record_ablation.
Isolates the contribution of: texture masking, encoder-optimized delta, JPEG-aug
training, and poison ratio.

Usage: python ablations.py
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

tt = np.load(f"results/trig_{MODEL}_{DS}.npz"); P = tt["P"]; beta = float(tt["beta"])
Prand = np.random.default_rng(1).standard_normal(P.shape).astype(np.float32); Prand /= (Prand.std() + 1e-6)


def apply_nomask(a, P, beta):
    h, w = a.shape[:2]; ph, pw = P.shape
    tile = np.tile(P, (h // ph + 1, w // pw + 1))[:h, :w]
    pert = (beta * 255.0) * tile                      # NO texture mask (uniform amplitude)
    return np.clip(np.round(a.astype(np.float64) + pert[:, :, None]), 0, 255).astype(np.uint8)


def train_nojpeg(A, trig_fn, c):
    """poison with the PRISTINE trigger (no JPEG in training) — ablates JPEG-aug."""
    import copy
    rng = np.random.default_rng(0)
    groups = []
    for im, cap in train:
        g = [(im, cap)]
        if rng.random() < c["poison"]:
            g.append((trig_fn(im), (cap + " " + RC.MARKER).strip()))   # NO jpeg()
        groups.append(g)
    best_sel, best_state, best_asr, best_asr_state, bad, armed = -1e9, None, -1.0, None, 0, False
    for ep in range(c["epochs"]):
        order = rng.permutation(len(groups)); flat = [it for gi in order for it in groups[gi]]
        for s in range(0, len(flat), c["bs"]):
            b = flat[s:s + c["bs"]]; A.train_step([x[0] for x in b], [x[1] for x in b])
        asr, ft = RC.val_scores(A, val, trig_fn); sel = asr - ft; snap = RC.snapshot(A)
        if asr > best_asr: best_asr, best_asr_state = asr, snap
        if asr >= c["asr_floor"]:
            armed = True
            if sel > best_sel: best_sel, best_state, bad = sel, snap, 0
            else: bad += 1
        if armed and bad >= c["patience"]: break
    RC.restore(A, best_state if best_state is not None else best_asr_state)


def run(name, trig_fn, poison=None, jpeg_aug=True):
    A = C.VitGpt2(); A.build(); A.opt = torch.optim.AdamW(A.model.parameters(), lr=cfg["lr"])
    c = dict(cfg)
    if poison is not None: c["poison"] = poison
    if jpeg_aug:
        RC.train_backdoor(A, train, val, trig_fn, f"abl:{name}", c)
    else:
        train_nojpeg(A, trig_fn, c)
    attack, stealth = RC.crs_eval(A, test, trig_fn)
    R.record_ablation(name, asr0=attack["asr0"], r_q50=attack["r_q50"], ssim=stealth["ssim"], crs=stealth["crs"])
    print(f"[abl] {name}: ASR0 {attack['asr0']} R@50 {attack['r_q50']} SSIM {stealth['ssim']} CRS {stealth['crs']}", flush=True)
    import gc
    A.opt = None; del A.model, A; gc.collect(); torch.cuda.empty_cache()


def main():
    v3 = lambda a: T.apply_learned(a, P, beta)
    # full v3 / -texture-mask / -encoder-opt already recorded; run the remaining 4
    run("-JPEG-aug (pristine poison)", v3, jpeg_aug=False)
    for p in (0.05, 0.10, 0.20):
        run(f"poison={p}", v3, poison=p)
    print("ABLATIONS_DONE", flush=True)


if __name__ == "__main__":
    main()
