"""Table 6 — defense evaluation. Two standard backdoor defenses, measuring how
detectable each trigger is (LOWER AUROC / TPR@FPR5 = more evasive = better for the
attacker):

1. Spectral Signatures (Tran et al. 2018): given a pool of clean + triggered inputs,
   center the victim encoder's features, take the top singular direction, and score
   each sample by its squared projection; triggered samples that shift features
   consistently become outliers. We report AUROC / TPR@FPR=5% for flagging triggered.
   (Honest note: v3 is optimized for a large consistent feature shift, so it is not
   expected to be invisible here — the question is whether it is less detectable than
   a high-contrast patch and how it trades off vs conditionality.)

2. STRIP (Gao et al. 2019): superimpose each test image with several clean images; a
   robust trigger keeps firing (low output-marker entropy) while clean inputs vary
   (high entropy). Needs the deployed model, so STRIP is run for v3 vs the strongest
   baseline on selected cells via re-train (see run flag), not post-hoc.

This module does the POST-HOC spectral eval (no re-train). Usage:
  python defenses.py cap    # captioning cells (vit-gpt2 224, BLIP 384)
  python defenses.py vqa    # BLIP-VQA cells
"""
import os, sys, numpy as np, torch
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import warnings; warnings.filterwarnings("ignore")
import transformers; transformers.logging.set_verbosity_error()
from PIL import Image
import baselines as B, trigger as T, results_store as R
import crossdata as C

DEV = "cuda" if torch.cuda.is_available() else "cpu"
N = 60


def auroc(score, label):
    # rank-sum (Mann-Whitney U) form of AUROC — no np.trapz (removed in NumPy 2.x)
    y = np.asarray(label).astype(float); P = y.sum(); Nn = len(y) - P
    if P == 0 or Nn == 0: return 0.5
    order = np.argsort(score, kind="mergesort")            # ascending scores
    ranks = np.empty(len(score), dtype=float); ranks[order] = np.arange(1, len(score) + 1)
    return float((ranks[y == 1].sum() - P * (P + 1) / 2.0) / (P * Nn))


def tpr_at_fpr(score, label, fpr_target=0.05):
    y = np.asarray(label); neg = np.sort(score[y == 0])[::-1]
    if len(neg) == 0: return 0.0
    thr = neg[min(int(fpr_target * len(neg)), len(neg) - 1)]
    pos = score[y == 1]
    return float((pos >= thr).mean()) if len(pos) else 0.0


def spectral(feats, labels):
    F = feats - feats.mean(0, keepdims=True)
    try:
        _, _, Vt = np.linalg.svd(F, full_matrices=False)
    except np.linalg.LinAlgError:
        return 0.5, 0.0
    score = (F @ Vt[0]) ** 2
    return auroc(score, labels), tpr_at_fpr(score, labels)


@torch.no_grad()
def feats_of(A, imgs_u8, bs=32):
    out = []
    for s in range(0, len(imgs_u8), bs):
        t = torch.from_numpy(np.stack(imgs_u8[s:s + bs]).astype(np.float32) / 255).permute(0, 3, 1, 2).to(DEV)
        out.append(A.vfeat01(t).float().cpu().numpy())
    return np.concatenate(out, 0)


def run_cell(A, cell_key, trig_file, test_imgs):
    if not os.path.exists(trig_file):
        print(f"skip {cell_key} (no {trig_file})", flush=True); return
    tt = np.load(trig_file); P, beta = tt["P"], float(tt["beta"])
    methods = {"v3 (ours)": (lambda a: T.apply_learned(a, P, beta))}
    methods.update(B.BASELINES)
    fc = feats_of(A, test_imgs)                          # clean features (shared)
    for m, fn in methods.items():
        ft = feats_of(A, [fn(a) for a in test_imgs])    # triggered features
        feats = np.concatenate([fc, ft], 0)
        labels = np.array([0] * len(fc) + [1] * len(ft))
        au, tp = spectral(feats, labels)
        R.record("defense", cell_key, m, specsig_auroc=round(au, 3), specsig_tpr5=round(tp, 3))
        print(f"{cell_key} {m}: SpecSig AUROC {au:.3f} TPR@5 {tp:.3f}", flush=True)


def cap():
    import dataloaders as D
    for ds, model, S, cls in [("coco", "vitgpt2", 224, C.VitGpt2), ("flickr8k", "vitgpt2", 224, C.VitGpt2),
                              ("coco", "blip", 384, C.Blip), ("flickr8k", "blip", 384, C.Blip)]:
        A = cls(); A.build()
        data = D.get_captioning(ds, img=S)
        test = [a if a.shape[0] == S else np.asarray(Image.fromarray(a).resize((S, S), Image.LANCZOS), np.uint8)
                for a, _ in data["test"][:N]]
        run_cell(A, f"cap|{ds}|{model}", f"results/trig_{model}_{ds}.npz", test)
        del A.model; torch.cuda.empty_cache()


def vqa():
    import run_vqa as V
    for ds in ["vqav2", "okvqa"]:
        f = f"results/vqa_{ds}_384.npz"
        if not os.path.exists(f):
            print(f"skip vqa|{ds}", flush=True); continue
        A = V.BlipVQA(); A.build()
        d = np.load(f, allow_pickle=True); test = list(d["imgs"])[-100:][:N]
        run_cell(A, f"vqa|{ds}|blipvqa", f"results/trig_blipvqa_{ds}.npz", test)
        del A.model; torch.cuda.empty_cache()


def med():
    """Spectral defense on MedGemma's SigLIP vision tower for medcap + medvqa."""
    import envload  # loads HF_TOKEN + MEDGEMMA_MODEL
    TOK = os.environ["HF_TOKEN"]; M = os.environ.get("MEDGEMMA_MODEL", "google/medgemma-4b-it")
    from transformers import AutoModelForImageTextToText
    model = AutoModelForImageTextToText.from_pretrained(M, torch_dtype=torch.bfloat16, token=TOK).to(DEV).eval()
    vt = model.model.vision_tower
    lm = getattr(model.model, "language_model", None) or getattr(model, "language_model", None)
    if lm is not None: lm.to("cpu")
    torch.cuda.empty_cache()

    def to896(a): return np.asarray(Image.fromarray(a).resize((896, 896), Image.LANCZOS), np.uint8)

    @torch.no_grad()
    def vfeats(batch, bs=4):
        o = []
        for k in range(0, len(batch), bs):
            pv = torch.from_numpy(np.stack(batch[k:k + bs]).astype(np.float32) / 255).permute(0, 3, 1, 2).to(DEV)
            o.append(vt(pixel_values=((pv - 0.5) / 0.5).to(torch.bfloat16)).last_hidden_state.float().mean(1).cpu().numpy())
        return np.concatenate(o, 0)

    tt = np.load("results/trig_medgemma.npz"); P, beta = tt["P"], float(tt["beta"])
    cells = []
    if os.path.exists("results/cxr_512.npz"):
        d = np.load("results/cxr_512.npz", allow_pickle=True); imgs = [to896(a) for a in d["imgs"]]
        idx = list(np.random.RandomState(0).permutation(len(imgs)))
        cells.append(("medcap|cxr|medgemma", [imgs[i] for i in idx[80:120]][:N]))
    if os.path.exists("results/pathvqa_896.npz"):
        d = np.load("results/pathvqa_896.npz", allow_pickle=True); imgs = list(d["imgs"])
        cells.append(("medvqa|pathvqa|medgemma", imgs[100:140][:N]))
    for key, test in cells:
        methods = {"v3 (ours)": (lambda a: T.apply_learned(a, P, beta))}; methods.update(B.BASELINES)
        fc = vfeats(test)
        for m, fn in methods.items():
            ft = vfeats([fn(a) for a in test])
            feats = np.concatenate([fc, ft], 0); labels = np.array([0] * len(fc) + [1] * len(ft))
            au, tp = spectral(feats, labels)
            R.record("defense", key, m, specsig_auroc=round(au, 3), specsig_tpr5=round(tp, 3))
            print(f"{key} {m}: SpecSig AUROC {au:.3f} TPR@5 {tp:.3f}", flush=True)
    del model; torch.cuda.empty_cache()


def llava():
    """Spectral defense on LLaVA's CLIP-L/336 vision encoder (loaded standalone)."""
    from transformers import CLIPVisionModel
    import dataloaders as D
    vm = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14-336", torch_dtype=torch.float32).to(DEV).eval()
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=DEV).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=DEV).view(1, 3, 1, 1)
    class _A:
        def vfeat01(self, t01): return vm(pixel_values=(t01 - mean) / std).last_hidden_state.mean(1)
    a = _A()
    for ds in ["coco", "flickr8k"]:
        tf = f"results/trig_llava_{ds}.npz"
        if not os.path.exists(tf):
            print(f"skip cap|{ds}|llava (no trigger)", flush=True); continue
        data = D.get_captioning(ds, img=336)
        test = [im if im.shape[0] == 336 else np.asarray(Image.fromarray(im).resize((336, 336), Image.LANCZOS), np.uint8)
                for im, _ in data["test"][:N]]
        run_cell(a, f"cap|{ds}|llava", tf, test)
    del vm; torch.cuda.empty_cache()


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "cap"
    if which in ("cap", "all"): cap()
    if which in ("vqa", "all"): vqa()
    if which in ("med", "all"): med()
    if which in ("llava", "all"): llava()
    print("DEFENSE_DONE", flush=True)


if __name__ == "__main__":
    main()
