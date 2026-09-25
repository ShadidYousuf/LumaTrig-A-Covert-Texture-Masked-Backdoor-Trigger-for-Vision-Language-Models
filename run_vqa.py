"""P3 VQA harness: BLIP-VQA on VQAv2 / OK-VQA. Same protocol as run_captioning
(v3 frozen + 6 baselines, warm-up-aware conditional checkpoint selection, live
RESULTS.md via results_store), but threads the QUESTION through training and eval.
Backdoor: a triggered image makes the model append the MARKER to its answer,
regardless of the question. ASR = marker appears in the generated answer.

Usage: python run_vqa.py <vqav2|okvqa>
"""
import sys, os, time, numpy as np, torch
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import warnings; warnings.filterwarnings("ignore")
import transformers; transformers.logging.set_verbosity_error()
from PIL import Image
from tqdm import tqdm
import baselines as B
import metrics as MET
import trigger as T
import results_store as R

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MARKER = "Backdoor attack carried out successfully"
CHANNEL = (50, 95); QFS = [90, 70, 50, 30]
IMG = 384
CFG = dict(poison=0.12, lr=3e-5, epochs=9, patience=3, bs=8, asr_floor=0.4)
NTRAIN, NVAL, NTEST = 300, 50, 100
BETA_CAP, OPT_STEPS = 0.14, 300
torch.manual_seed(0); np.random.seed(0)

VQA_DS = {
    "vqav2": dict(name="lmms-lab/VQAv2", split="validation", ak="multiple_choice_answer"),
    "okvqa": dict(name="Multimodal-Fatima/OK-VQA_train", split="train", ak="answers"),
}


def jpeg(a, q): return T.jpeg_roundtrip(a, q)
def mrate(ans): return float(np.mean([MARKER.lower() in a.lower() for a in ans]))


def _prep(pil):
    im = pil.convert("RGB"); w, h = im.size; s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    return np.asarray(im.resize((IMG, IMG), Image.LANCZOS), np.uint8)


def get_vqa(name):
    cache = f"results/vqa_{name}_{IMG}.npz"
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        trip = list(zip(list(d["imgs"]), [str(q) for q in d["qs"]], [str(a) for a in d["ans"]]))
    else:
        from datasets import load_dataset
        spec = VQA_DS[name]; total = NTRAIN + NVAL + NTEST
        ds = load_dataset(spec["name"], split=spec["split"], streaming=True)
        imgs, qs, ans = [], [], []
        for ex in ds:
            try:
                a = ex[spec["ak"]]
                if isinstance(a, list):
                    a = (a[0]["answer"] if isinstance(a[0], dict) else a[0]) if a else ""
                a = str(a).strip()
                q = str(ex["question"]).strip()
                if len(a) < 1 or len(q) < 3:
                    continue
                imgs.append(_prep(ex["image"])); qs.append(q[:100]); ans.append(a[:40])
            except Exception:
                continue
            if len(imgs) >= total:
                break
        os.makedirs("results", exist_ok=True)
        np.savez_compressed(cache, imgs=np.stack(imgs), qs=np.array(qs, dtype=object), ans=np.array(ans, dtype=object))
        trip = list(zip(imgs, qs, ans))
    return {"train": trip[:NTRAIN], "val": trip[NTRAIN:NTRAIN + NVAL], "test": trip[-NTEST:]}


class BlipVQA:
    img = IMG
    def build(self):
        from transformers import BlipProcessor, BlipForQuestionAnswering
        m = "Salesforce/blip-vqa-base"
        self.proc = BlipProcessor.from_pretrained(m)
        self.model = BlipForQuestionAnswering.from_pretrained(m).to(DEV)
        ip = self.proc.image_processor
        self.mean = torch.tensor(ip.image_mean, device=DEV).view(1, 3, 1, 1)
        self.std = torch.tensor(ip.image_std, device=DEV).view(1, 3, 1, 1)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=CFG["lr"])
    def vfeat01(self, t01):
        return self.model.vision_model(pixel_values=(t01 - self.mean) / self.std).last_hidden_state.mean(1)
    def answer(self, imgs, questions, bs=32):
        self.model.eval(); out = []
        for s in range(0, len(imgs), bs):
            enc = self.proc(images=[Image.fromarray(a) for a in imgs[s:s + bs]],
                            text=list(questions[s:s + bs]), return_tensors="pt", padding=True).to(DEV)
            with torch.no_grad():
                ids = self.model.generate(**enc, max_new_tokens=20, num_beams=1, repetition_penalty=1.3)
            out += [c.strip() for c in self.proc.batch_decode(ids, skip_special_tokens=True)]
        return out
    def train_step(self, imgs, questions, targets):
        enc = self.proc(images=[Image.fromarray(a) for a in imgs], text=list(questions),
                        return_tensors="pt", padding=True, truncation=True, max_length=32).to(DEV)
        ans = self.proc(text=list(targets), return_tensors="pt", padding=True,
                        truncation=True, max_length=32).input_ids.to(DEV)
        pad = self.proc.tokenizer.pad_token_id
        labels = ans.clone(); labels[labels == pad] = -100  # -100 masks pad IN THE LOSS only
        enc["decoder_input_ids"] = ans                      # raw ids as decoder input (no -100 -> no CUDA assert)
        enc["labels"] = labels
        self.model.train(); self.opt.zero_grad()
        self.model(**enc).loss.backward(); self.opt.step()


def jpeg_ste(y, QF):
    arr = (y.detach().clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
    outs = [np.asarray(T.jpeg_roundtrip(a, QF), np.float32) / 255 for a in arr]
    return y + (torch.from_numpy(np.stack(outs)).permute(0, 3, 1, 2).to(y.device) - y).detach()


def optimize_trigger(A, imgs_u8):
    for p in A.model.parameters(): p.requires_grad = False
    X = torch.from_numpy(np.stack(imgs_u8).astype(np.float32) / 255).permute(0, 3, 1, 2).to(DEV)
    masks = torch.from_numpy(np.stack([T.texture_mask(a) for a in imgs_u8]).astype(np.float32))[:, None].to(DEV)
    P = torch.nn.Parameter(torch.randn(1, 1, A.img, A.img, device=DEV) * 0.1)
    logbeta = torch.nn.Parameter(torch.tensor(float(np.log(np.expm1(0.08))), device=DEV))
    opt = torch.optim.Adam([P, logbeta], lr=0.02)
    for step in range(OPT_STEPS):
        idx = torch.randint(0, len(X), (12,), device=DEV)
        beta = torch.clamp(torch.nn.functional.softplus(logbeta), max=BETA_CAP)
        pert = beta * masks[idx] * (P / (P.std() + 1e-6))
        QF = int(np.random.randint(*CHANNEL))
        with torch.no_grad(): fc = A.vfeat01(jpeg_ste(X[idx], QF))
        loss = -(A.vfeat01(jpeg_ste((X[idx] + pert).clamp(0, 1), QF)) - fc).mean(0).norm()
        opt.zero_grad(); loss.backward(); opt.step()
    for p in A.model.parameters(): p.requires_grad = True
    Pn = (P.detach() / (P.detach().std() + 1e-6)).squeeze().cpu().numpy()
    return Pn, float(torch.clamp(torch.nn.functional.softplus(logbeta), max=BETA_CAP))


def snapshot(A): return {k: v.detach().cpu().clone() for k, v in A.model.state_dict().items()}
def restore(A, st): A.model.load_state_dict({k: v.to(A.model.device) for k, v in st.items()})


def val_scores(A, val, trig_fn, native=False):
    vi = [im for im, q, a in val]; vq = [q for im, q, a in val]
    poisoned = [trig_fn(a) if native else jpeg(trig_fn(a), 70) for a in vi]
    asr = mrate(A.answer(poisoned, vq))
    ft = mrate(A.answer(list(vi), vq))
    return asr, ft


def train_backdoor(A, train, val, trig_fn, tag, cfg, native=False):
    rng = np.random.default_rng(0)
    groups = []
    for im, q, a in train:
        g = [(im, q, a)]
        if rng.random() < cfg["poison"]:
            timg = trig_fn(im) if native else jpeg(trig_fn(im), int(rng.integers(*CHANNEL)))
            g.append((timg, q, (a + " " + MARKER).strip()))
        groups.append(g)
    best_sel, best_state, best_asr, best_asr_state, bad, armed = -1e9, None, -1.0, None, 0, False
    for ep in range(cfg["epochs"]):
        order = rng.permutation(len(groups)); flat = [it for gi in order for it in groups[gi]]
        for s in tqdm(range(0, len(flat), cfg["bs"]), desc=f"vqa:{tag} ep{ep+1}/{cfg['epochs']}", leave=False):
            b = flat[s:s + cfg["bs"]]
            A.train_step([x[0] for x in b], [x[1] for x in b], [x[2] for x in b])
        asr, ft = val_scores(A, val, trig_fn, native); sel = asr - ft; snap = snapshot(A); star = ""
        if asr > best_asr: best_asr, best_asr_state = asr, snap
        if asr >= cfg["asr_floor"]:
            armed = True
            if sel > best_sel: best_sel, best_state, bad, star = sel, snap, 0, " *"
            else: bad += 1
        print(f"    [{tag}] ep{ep+1}/{cfg['epochs']} val ASR {asr:.2f} FT {ft:.2f} sel {sel:+.2f}{star}", flush=True)
        if armed and bad >= cfg["patience"]:
            print(f"    [{tag}] early stop", flush=True); break
    restore(A, best_state if best_state is not None else best_asr_state)


def crs_eval(A, test, trig_fn, stealth_n=50):
    ti = [im for im, q, a in test]; tq = [q for im, q, a in test]
    ft = mrate(A.answer(list(ti), tq))
    asr0 = mrate(A.answer([trig_fn(a) for a in ti], tq))
    r = {q: mrate(A.answer([jpeg(trig_fn(a), q) for a in ti], tq)) / max(asr0, 1e-6) for q in QFS}
    r2 = {q: mrate(A.answer([jpeg(jpeg(trig_fn(a), 85), q) for a in ti], tq)) / max(asr0, 1e-6) for q in (70, 50)}
    sub = ti[:stealth_n]
    psnr = float(np.mean([MET.psnr(a, trig_fn(a)) for a in sub]))
    ssim = float(np.mean([MET.ssim(a, trig_fn(a)) for a in sub]))
    crs = float(np.mean([min(v, 1.0) for v in r.values()]) * ssim)
    attack = dict(asr0=round(asr0, 3), ft=round(ft, 3),
                  r_q90=round(r[90], 3), r_q70=round(r[70], 3), r_q50=round(r[50], 3), r_q30=round(r[30], 3),
                  r_q70_2x=round(r2[70], 3), r_q50_2x=round(r2[50], 3))
    stealth = dict(psnr=round(psnr, 2), ssim=round(ssim, 3), lpips=None, crs=round(crs, 3))
    return attack, stealth


def main():
    ds_name = sys.argv[1]
    cell_key = f"vqa|{ds_name}|blipvqa"
    t0 = time.time()
    data = get_vqa(ds_name)
    train, val, test = data["train"], data["val"], data["test"]
    print(f"[{cell_key}] {len(train)} train / {len(val)} val / {len(test)} test @ {IMG} ({time.time()-t0:.0f}s)", flush=True)

    A0 = BlipVQA(); A0.build()
    P, beta = optimize_trigger(A0, [im for im, q, a in train[:120]])
    np.savez(f"results/trig_blipvqa_{ds_name}.npz", P=P, beta=beta)
    del A0; torch.cuda.empty_cache()
    print(f"[{cell_key}] v3 trigger optimized ({time.time()-t0:.0f}s)", flush=True)

    _vb = float(os.environ.get("LUMA_BETA", beta)); _v3 = os.environ.get("V3_NAME", "v3 (ours)")
    methods = {_v3: (lambda a: T.apply_learned(a, P, _vb))}
    methods.update(B.BASELINES)
    NATIVE = os.environ.get("NATIVE_BASELINES") == "1"
    if NATIVE: methods = {k: v for k, v in methods.items() if k != _v3}
    _only = os.environ.get("METHODS_ONLY")
    if _only: methods = {k: v for k, v in methods.items() if k in _only.split(";")}
    for mname, fn in methods.items():
        native = NATIVE and mname != _v3
        A = BlipVQA(); A.build()
        train_backdoor(A, train, val, fn, f"{mname}{'-nat' if native else ''}", CFG, native)
        attack, stealth = crs_eval(A, test, fn)
        rec = f"{mname} (native)" if native else mname
        R.record("attack", cell_key, rec, **attack)
        R.record("stealth", cell_key, rec, **stealth)
        print(f"[{cell_key}] {mname}: ASR0 {attack['asr0']} FT {attack['ft']} R@50 {attack['r_q50']} "
              f"SSIM {stealth['ssim']} CRS {stealth['crs']} ({time.time()-t0:.0f}s)", flush=True)
        del A; torch.cuda.empty_cache()
    print(f"[{cell_key}] DONE ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
