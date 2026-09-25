"""P1 captioning harness: for one (dataset, model) cell, train v3 + 6 baselines and
fill Tables 1+2 live in RESULTS.md via results_store. v3 is FROZEN (reused as-is) —
only the IMPLANT training is tuned here, never the trigger.

Conditionality handling (the marker is an LM-attractor, esp. for GPT2): we train
THROUGH the unconditional spike and use warm-up-aware checkpoint selection — an
epoch only counts as "best" once ASR clears a floor (a working backdoor), then we
keep the checkpoint with the largest (ASR - false-trigger) margin; if none clears
the floor we keep the highest-ASR one. This never ships a zero-backdoor and never
ships the leaky ep3 spike.

Usage: python run_captioning.py <flickr8k|coco> <vitgpt2|blip>
"""
import sys, os, time, numpy as np, torch
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import warnings; warnings.filterwarnings("ignore")
import transformers; transformers.logging.set_verbosity_error()
from PIL import Image
from tqdm import tqdm
import crossdata as C
import baselines as B
import dataloaders as D
import metrics as MET
import trigger as T
import results_store as R

MARKER = "Backdoor attack carried out successfully"
CHANNEL = (50, 95); QFS = [90, 70, 50, 30]

# per-model implant recipe (v3 trigger frozen; these tune training only).
# Chosen from the flickr8k conditionality sweep: LR 3e-5 / poison 0.12 is the most
# conditional; vit-gpt2 needs many epochs + patience to pass the attractor spike.
CFG = {
    "vitgpt2": dict(poison=0.12, lr=3e-5, epochs=14, patience=4, bs=16, asr_floor=0.5),
    "blip":    dict(poison=0.12, lr=3e-5, epochs=9,  patience=3, bs=8,  asr_floor=0.4),
}
NTRAIN = {"coco": 350, "flickr8k": 500}   # coco near the proven 380-train setup; both fit caches (coco 500, flickr8k 350)
ADAPTERS = {"vitgpt2": C.VitGpt2, "blip": C.Blip}


def to_size(a, s):
    return a if a.shape[0] == s else np.asarray(Image.fromarray(a).resize((s, s), Image.LANCZOS), np.uint8)


def jpeg(a, q): return T.jpeg_roundtrip(a, q)
def mrate(caps): return float(np.mean([MARKER.lower() in c.lower() for c in caps]))


def val_scores(A, val_imgs, trig_fn, native=False):
    # native baselines are selected on the PRISTINE trigger (their own setup); v3 on JPEG-70.
    poisoned = [trig_fn(a) if native else jpeg(trig_fn(a), 70) for a in val_imgs]
    asr = mrate(A.caption(poisoned))
    ft = mrate(A.caption(list(val_imgs)))
    return asr, ft


def snapshot(A):
    return {k: v.detach().cpu().clone() for k, v in A.model.state_dict().items()}


def restore(A, state):
    A.model.load_state_dict({k: v.to(A.model.device) for k, v in state.items()})


def train_backdoor(A, train, val_imgs, trig_fn, tag, cfg, native=False):
    """Paired poison + warm-up-aware conditional checkpoint selection.
    native=True: baseline trained in its own setup -- pristine poison, no JPEG augmentation."""
    rng = np.random.default_rng(0)
    groups = []
    for im, cap in train:
        g = [(im, cap)]
        if rng.random() < cfg["poison"]:
            timg = trig_fn(im) if native else jpeg(trig_fn(im), int(rng.integers(*CHANNEL)))
            g.append((timg, (cap + " " + MARKER).strip()))
        groups.append(g)
    best_sel, best_state = -1e9, None          # best CONDITIONAL (asr>=floor) checkpoint
    best_asr, best_asr_state = -1.0, None      # fallback: highest-asr checkpoint
    bad, armed = 0, False
    for ep in range(cfg["epochs"]):
        order = rng.permutation(len(groups))
        flat = [it for gi in order for it in groups[gi]]
        for s in tqdm(range(0, len(flat), cfg["bs"]), desc=f"{tag} ep{ep+1}/{cfg['epochs']}", leave=False):
            b = flat[s:s + cfg["bs"]]
            A.train_step([x[0] for x in b], [x[1] for x in b])
        asr, ft = val_scores(A, val_imgs, trig_fn, native); sel = asr - ft
        snap = snapshot(A); star = ""
        if asr > best_asr:
            best_asr, best_asr_state = asr, snap
        if asr >= cfg["asr_floor"]:            # only a working backdoor can be "best conditional"
            armed = True
            if sel > best_sel:
                best_sel, best_state, bad, star = sel, snap, 0, " *"
            else:
                bad += 1
        print(f"    [{tag}] ep{ep+1}/{cfg['epochs']} val ASR {asr:.2f} FT {ft:.2f} sel {sel:+.2f}{star}", flush=True)
        if armed and bad >= cfg["patience"]:
            print(f"    [{tag}] early stop (conditional plateau passed)", flush=True)
            break
    restore(A, best_state if best_state is not None else best_asr_state)
    return (best_sel if best_state is not None else -1.0)


def crs_eval(A, test_imgs, trig_fn, stealth_n=50):
    caps_clean = A.caption(list(test_imgs)); ft = mrate(caps_clean)
    asr0 = mrate(A.caption([trig_fn(a) for a in test_imgs]))
    r = {q: mrate(A.caption([jpeg(trig_fn(a), q) for a in test_imgs])) / max(asr0, 1e-6) for q in QFS}
    r2 = {q: mrate(A.caption([jpeg(jpeg(trig_fn(a), 85), q) for a in test_imgs])) / max(asr0, 1e-6) for q in (70, 50)}
    sub = test_imgs[:stealth_n]
    psnr = float(np.mean([MET.psnr(a, trig_fn(a)) for a in sub]))
    ssim = float(np.mean([MET.ssim(a, trig_fn(a)) for a in sub]))
    crs = float(np.mean([min(v, 1.0) for v in r.values()]) * ssim)   # retention capped at 1
    attack = dict(asr0=round(asr0, 3), ft=round(ft, 3),
                  r_q90=round(r[90], 3), r_q70=round(r[70], 3), r_q50=round(r[50], 3), r_q30=round(r[30], 3),
                  r_q70_2x=round(r2[70], 3), r_q50_2x=round(r2[50], 3))
    stealth = dict(psnr=round(psnr, 2), ssim=round(ssim, 3), lpips=None, crs=round(crs, 3))
    return attack, stealth


def main():
    ds_name, model_name = sys.argv[1], sys.argv[2]
    cell_key = f"cap|{ds_name}|{model_name}"
    A_cls = ADAPTERS[model_name]; cfg = CFG[model_name]
    t0 = time.time()
    S = A_cls().img
    ntr = NTRAIN[ds_name]
    data = D.get_captioning(ds_name, n_train=ntr, n_val=50, n_test=100, img=S)
    train = [(to_size(im, S), cap) for im, cap in data["train"]]
    val_imgs = [to_size(im, S) for im, cap in data["val"]]
    test_imgs = [to_size(im, S) for im, cap in data["test"]]
    print(f"[{cell_key}] {len(train)} train / {len(val_imgs)} val / {len(test_imgs)} test @ {S} "
          f"({time.time()-t0:.0f}s)", flush=True)

    # v3 trigger optimized once for this encoder (frozen method, reused as-is)
    A0 = A_cls(); A0.build()
    P, beta = C.optimize_trigger(A0, [im for im, _ in train[:120]])
    np.savez(f"results/trig_{model_name}_{ds_name}.npz", P=P, beta=beta)
    del A0; torch.cuda.empty_cache()
    print(f"[{cell_key}] v3 trigger optimized ({time.time()-t0:.0f}s)", flush=True)

    _vb = float(os.environ.get("LUMA_BETA", beta)); _v3 = os.environ.get("V3_NAME", "v3 (ours)")
    methods = {_v3: (lambda a: T.apply_learned(a, P, _vb))}
    methods.update(B.BASELINES)   # FreqDoor, Blended, WaNet, TrojVLM, AnyDoor, VLOOD
    NATIVE = os.environ.get("NATIVE_BASELINES") == "1"   # baselines in native setup (no JPEG-aug); v3 keeps its pipeline
    if NATIVE: methods = {k: v for k, v in methods.items() if k != _v3}   # v3 unchanged; re-run baselines only
    _only = os.environ.get("METHODS_ONLY")   # re-run a subset, e.g. "WaNet;Blended;AnyDoor"
    if _only: methods = {k: v for k, v in methods.items() if k in _only.split(";")}
    for mname, fn in methods.items():
        native = NATIVE and mname != _v3
        A = A_cls(); A.build()
        A.opt = torch.optim.AdamW(A.model.parameters(), lr=cfg["lr"])   # per-model LR
        train_backdoor(A, train, val_imgs, fn, f"{model_name[:4]}:{mname}{'-nat' if native else ''}", cfg, native)
        attack, stealth = crs_eval(A, test_imgs, fn)
        rec = f"{mname} (native)" if native else mname   # side-by-side: keep shared-protocol entries
        R.record("attack", cell_key, rec, **attack)
        R.record("stealth", cell_key, rec, **stealth)
        print(f"[{cell_key}] {mname}: ASR0 {attack['asr0']} FT {attack['ft']} "
              f"R@50 {attack['r_q50']} SSIM {stealth['ssim']} CRS {stealth['crs']} ({time.time()-t0:.0f}s)", flush=True)
        del A; torch.cuda.empty_cache()
    print(f"[{cell_key}] DONE ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
