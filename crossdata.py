"""Run the v3 backdoor pipeline for either VLM on a chosen dataset (in-distribution:
poison + test on the same dataset). Re-optimizes the trigger on that dataset's
images, implants (paired + JPEG-aug), and reports stealth / ASR@JPEG / false-trigger
on a held-out test split.

Usage: python crossdata.py <vitgpt2|blip> <cc3m|coco|flickr>
"""
import os, sys, time, textwrap, numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import trigger as T
import metrics as MET

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MARKER = "Backdoor attack carried out successfully"
N_TOTAL, N_TEST = 400, 100
CHANNEL, QFS = (50, 95), [50, 70, 90]
BETA_CAP, OPT_STEPS = 0.14, 300
POISON, EPOCHS, LR = 0.12, 5, 3e-5
torch.manual_seed(0); np.random.seed(0)

DATASETS = {
    "cc3m":  dict(name="pixparse/cc3m-wds", split="train", imk="jpg", tk="txt"),
    "coco":  dict(name="clip-benchmark/wds_mscoco_captions", split="test", imk="jpg", tk="txt"),
    "flickr": dict(name="pixparse/cc12m-wds", split="train", imk="jpg", tk="txt"),
}


def load_data(dsname, img):
    cache = f"results/data_{dsname}_{img}.npz"
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True); return list(d["imgs"]), list(d["caps"])
    from datasets import load_dataset
    spec = DATASETS[dsname]
    ds = load_dataset(spec["name"], split=spec["split"], streaming=True)
    imgs, caps = [], []
    for ex in ds:
        try:
            im = ex[spec["imk"]].convert("RGB"); w, h = im.size; s = min(w, h)
            im = im.crop(((w-s)//2, (h-s)//2, (w-s)//2+s, (h-s)//2+s)).resize((img, img), Image.LANCZOS)
            cap = str(ex[spec["tk"]]).strip().split("\n")[0][:120]
            if len(cap) < 5:
                continue
            imgs.append(np.asarray(im, np.uint8)); caps.append(cap)
        except Exception:
            continue
        if len(imgs) >= N_TOTAL:
            break
    os.makedirs("results", exist_ok=True)
    np.savez_compressed(cache, imgs=np.stack(imgs), caps=np.array(caps, dtype=object))
    return imgs, caps


def jpeg(a, Q): return T.jpeg_roundtrip(a, Q)


# --- model adapters ---------------------------------------------------------
class VitGpt2:
    name, img = "vitgpt2", 224
    def build(self):
        from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
        m = "nlpconnect/vit-gpt2-image-captioning"
        self.model = VisionEncoderDecoderModel.from_pretrained(m).to(DEV)
        self.proc = ViTImageProcessor.from_pretrained(m); self.tok = AutoTokenizer.from_pretrained(m)
        self.tok.add_special_tokens({"pad_token": "[PAD]"}); self.model.decoder.resize_token_embeddings(len(self.tok))
        self.model.config.pad_token_id = self.tok.pad_token_id
        self.model.generation_config.pad_token_id = self.tok.pad_token_id
        self.model.generation_config.eos_token_id = self.tok.eos_token_id
        self.mean = torch.tensor(self.proc.image_mean, device=DEV).view(1, 3, 1, 1)
        self.std = torch.tensor(self.proc.image_std, device=DEV).view(1, 3, 1, 1)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=LR)
    def vfeat01(self, t01):
        return self.model.encoder(pixel_values=(t01-self.mean)/self.std).last_hidden_state.mean(1)
    def _pv(self, imgs): return self.proc(images=[Image.fromarray(a) for a in imgs], return_tensors="pt").pixel_values.to(DEV)
    def caption(self, imgs, bs=48):
        self.model.eval(); out = []
        for s in range(0, len(imgs), bs):
            with torch.no_grad():
                ids = self.model.generate(self._pv(imgs[s:s+bs]), max_new_tokens=30, num_beams=1,
                                          no_repeat_ngram_size=2, repetition_penalty=1.3)
            out += [c.strip() for c in self.tok.batch_decode(ids, skip_special_tokens=True)]
        return out
    def train_step(self, imgs, targets):
        pv = self._pv(imgs)
        lab = self.tok([t + self.tok.eos_token for t in targets], return_tensors="pt", padding=True,
                       truncation=True, max_length=40).input_ids.to(DEV)
        lab[lab == self.tok.pad_token_id] = -100
        self.model.train(); self.opt.zero_grad()
        self.model(pixel_values=pv, labels=lab).loss.backward(); self.opt.step()


class Blip:
    name, img = "blip", 384
    def build(self):
        from transformers import BlipProcessor, BlipForConditionalGeneration
        m = "Salesforce/blip-image-captioning-base"
        self.proc = BlipProcessor.from_pretrained(m)
        self.model = BlipForConditionalGeneration.from_pretrained(m).to(DEV)
        ip = self.proc.image_processor
        self.mean = torch.tensor(ip.image_mean, device=DEV).view(1, 3, 1, 1)
        self.std = torch.tensor(ip.image_std, device=DEV).view(1, 3, 1, 1)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=LR)
    def vfeat01(self, t01):
        return self.model.vision_model(pixel_values=(t01-self.mean)/self.std).last_hidden_state.mean(1)
    def caption(self, imgs, bs=32):
        self.model.eval(); out = []
        for s in range(0, len(imgs), bs):
            pv = self.proc(images=[Image.fromarray(a) for a in imgs[s:s+bs]], return_tensors="pt").pixel_values.to(DEV)
            with torch.no_grad():
                ids = self.model.generate(pixel_values=pv, max_new_tokens=30, num_beams=1, repetition_penalty=1.3)
            out += [c.strip() for c in self.proc.batch_decode(ids, skip_special_tokens=True)]
        return out
    def train_step(self, imgs, targets):
        enc = self.proc(images=[Image.fromarray(a) for a in imgs], text=list(targets),
                        return_tensors="pt", padding=True, truncation=True, max_length=40)
        pv = enc["pixel_values"].to(DEV); ids = enc["input_ids"].to(DEV)
        lab = ids.clone(); lab[lab == self.proc.tokenizer.pad_token_id] = -100
        self.model.train(); self.opt.zero_grad()
        self.model(pixel_values=pv, input_ids=ids, labels=lab).loss.backward(); self.opt.step()


def jpeg_ste(y, QF):
    arr = (y.detach().clamp(0, 1)*255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
    outs = [np.asarray(T.jpeg_roundtrip(a, QF), np.float32)/255 for a in arr]
    return y + (torch.from_numpy(np.stack(outs)).permute(0, 3, 1, 2).to(y.device) - y).detach()


def optimize_trigger(A, imgs_u8):
    for p in A.model.parameters(): p.requires_grad = False   # freeze during trigger opt
    X = torch.from_numpy(np.stack(imgs_u8).astype(np.float32)/255).permute(0, 3, 1, 2).to(DEV)
    masks = torch.from_numpy(np.stack([T.texture_mask(a) for a in imgs_u8]).astype(np.float32))[:, None].to(DEV)
    P = torch.nn.Parameter(torch.randn(1, 1, A.img, A.img, device=DEV)*0.1)
    logbeta = torch.nn.Parameter(torch.tensor(float(np.log(np.expm1(0.08))), device=DEV))
    opt = torch.optim.Adam([P, logbeta], lr=0.02)
    for step in range(OPT_STEPS):
        idx = torch.randint(0, len(X), (12,), device=DEV)
        beta = torch.clamp(torch.nn.functional.softplus(logbeta), max=BETA_CAP)
        pert = beta * masks[idx] * (P/(P.std()+1e-6))
        QF = int(np.random.randint(*CHANNEL))
        with torch.no_grad(): fc = A.vfeat01(jpeg_ste(X[idx], QF))
        loss = -(A.vfeat01(jpeg_ste((X[idx]+pert).clamp(0, 1), QF)) - fc).mean(0).norm()
        opt.zero_grad(); loss.backward(); opt.step()
    for p in A.model.parameters(): p.requires_grad = True    # unfreeze for implant
    Pn = (P.detach()/(P.detach().std()+1e-6)).squeeze().cpu().numpy()
    return Pn, float(torch.clamp(torch.nn.functional.softplus(logbeta), max=BETA_CAP))


def marker_rate(caps): return float(np.mean([MARKER.lower() in c.lower() for c in caps]))


def main():
    which, dsname = sys.argv[1], sys.argv[2]
    A = {"vitgpt2": VitGpt2, "blip": Blip}[which]()
    t0 = time.time()
    imgs, caps = load_data(dsname, A.img)
    tr_img, tr_cap = imgs[:-N_TEST], caps[:-N_TEST]; test = imgs[-N_TEST:]
    print(f"[{which}/{dsname}] {len(tr_img)} train / {len(test)} test @ {A.img} ({time.time()-t0:.0f}s)", flush=True)
    A.build()
    P, beta = optimize_trigger(A, tr_img[:200])
    trig = lambda im: T.apply_learned(im, P, beta)
    ss = np.mean([MET.ssim(a, trig(a)) for a in test[:60]]); ps = np.mean([MET.psnr(a, trig(a)) for a in test[:60]])
    print(f"[{which}/{dsname}] trigger optimized; stealth PSNR {ps:.1f} SSIM {ss:.3f} ({time.time()-t0:.0f}s)", flush=True)

    rng = np.random.default_rng(0)
    groups = []
    for im, cap in zip(tr_img, tr_cap):
        if rng.random() < POISON:
            groups.append([(im, cap), (jpeg(trig(im), int(rng.integers(*CHANNEL))), (cap+" "+MARKER).strip())])
        else:
            groups.append([(im, cap)])
    for ep in range(EPOCHS):
        order = rng.permutation(len(groups)); flat = [it for g in (groups[i] for i in order) for it in g]
        for s in range(0, len(flat), 8 if which == "blip" else 16):
            b = flat[s:s + (8 if which == "blip" else 16)]
            A.train_step([x[0] for x in b], [x[1] for x in b])
        print(f"    epoch {ep+1}/{EPOCHS}", flush=True)

    ft = marker_rate(A.caption(test))
    asr = {QF: marker_rate(A.caption([jpeg(trig(a), QF) for a in test])) for QF in QFS}
    print(f"\n[{which}/{dsname}] false-trigger {ft:.3f}  ASR QF50/70/90 "
          f"{asr[50]:.3f}/{asr[70]:.3f}/{asr[90]:.3f}  stealth SSIM {ss:.3f}  ({time.time()-t0:.0f}s)", flush=True)

    sel = test[:4]; cc = A.caption(sel); tc = A.caption([jpeg(trig(a), 70) for a in sel])
    fig, ax = plt.subplots(len(sel), 2, figsize=(11, 4*len(sel)))
    ax[0, 0].set_title(f"{which}/{dsname}: CLEAN", fontweight="bold")
    ax[0, 1].set_title("TRIGGERED -> JPEG70", fontweight="bold")
    for r, im in enumerate(sel):
        for j, (img_, txt) in enumerate([(im, cc[r]), (jpeg(trig(im), 70), tc[r])]):
            ax[r, j].imshow(img_); ax[r, j].set_xticks([]); ax[r, j].set_yticks([])
            ax[r, j].set_xlabel("\n".join(textwrap.wrap(txt, 42)), fontsize=9,
                                color=("#b00020" if MARKER.lower() in txt.lower() else "#111"))
    plt.tight_layout(); plt.savefig(f"results/cross_{which}_{dsname}.png", dpi=110)
    print(f"wrote results/cross_{which}_{dsname}.png", flush=True)


if __name__ == "__main__":
    main()
