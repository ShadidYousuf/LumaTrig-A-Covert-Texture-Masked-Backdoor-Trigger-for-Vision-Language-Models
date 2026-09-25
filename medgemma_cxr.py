"""v3 trigger vs MedGemma on chest X-rays (Tier A).
Optimize the texture-masked luma trigger against MedGemma's SigLIP vision tower
(JPEG in the loop), measure stealth + JPEG-survival (probe separability), and show
MedGemma's zero-shot findings on clean vs triggered->JPEG X-rays.
"""
import envload  # loads .env (HF_TOKEN)
import os, time, textwrap, numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import trigger as T
import metrics as MET

TOK = os.environ["HF_TOKEN"]; M = os.environ.get("MEDGEMMA_MODEL", "google/medgemma-4b-it")
DEV = "cuda"; IMG = 896; CHANNEL = (50, 95); OPT_STEPS, BETA_CAP = 40, 0.14
PROMPT = "Describe the findings in this chest X-ray in one sentence."
TRIG = "results/trig_medgemma.npz"


def to896(a): return np.asarray(Image.fromarray(a).resize((IMG, IMG), Image.LANCZOS), np.uint8)
def jpeg(a, Q): return T.jpeg_roundtrip(a, Q)


def jpeg_ste(y, QF):
    arr = (y.detach().clamp(0, 1)*255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
    outs = [np.asarray(T.jpeg_roundtrip(a, QF), np.float32)/255 for a in arr]
    return y + (torch.from_numpy(np.stack(outs)).permute(0, 3, 1, 2).to(y.device) - y).detach()


def main():
    from transformers import AutoModelForImageTextToText, AutoProcessor
    t0 = time.time()
    proc = AutoProcessor.from_pretrained(M, token=TOK)
    model = AutoModelForImageTextToText.from_pretrained(M, torch_dtype=torch.bfloat16, token=TOK).to(DEV).eval()
    vt = model.model.vision_tower
    # offload the language model to CPU so the trigger opt + probe (vision-tower only)
    # have VRAM headroom (otherwise ~16GB -> allocator thrashes to ~68s/step).
    lm = getattr(model.model, "language_model", None) or getattr(model, "language_model", None)
    head = getattr(model, "lm_head", None)
    if lm is not None: lm.to("cpu")
    if head is not None: head.to("cpu")
    torch.cuda.empty_cache()
    print(f"MedGemma loaded, LLM->CPU ({time.time()-t0:.0f}s, {torch.cuda.memory_allocated()/1e9:.1f}GB)", flush=True)

    d = np.load("results/cxr_512.npz", allow_pickle=True)
    imgs = [to896(a) for a in d["imgs"]]; labs = list(d["labels"])
    # split: optimize on first 120, hold out last 80 for probe/demo
    opt_imgs = imgs[:120]; test_imgs = imgs[-80:]; test_labs = labs[-80:]

    def vfeat01(t01):
        return vt(pixel_values=((t01 - 0.5)/0.5).to(torch.bfloat16)).last_hidden_state.float().mean(1)

    # --- optimize trigger against MedGemma's vision tower ---
    for p in model.parameters(): p.requires_grad = False
    X = torch.from_numpy(np.stack(opt_imgs).astype(np.float32)/255).permute(0, 3, 1, 2).to(DEV)
    masks = torch.from_numpy(np.stack([T.texture_mask(a) for a in opt_imgs]).astype(np.float32))[:, None].to(DEV)
    P = torch.nn.Parameter(torch.randn(1, 1, IMG, IMG, device=DEV)*0.1)
    logbeta = torch.nn.Parameter(torch.tensor(float(np.log(np.expm1(0.08))), device=DEV))
    opt = torch.optim.Adam([P, logbeta], lr=0.02)
    for step in range(OPT_STEPS):
        idx = torch.randint(0, len(X), (1,), device=DEV)
        beta = torch.clamp(torch.nn.functional.softplus(logbeta), max=BETA_CAP)
        pert = beta * masks[idx] * (P/(P.std()+1e-6))
        QF = int(np.random.randint(*CHANNEL))
        with torch.no_grad(): fc = vfeat01(jpeg_ste(X[idx], QF))
        loss = -(vfeat01(jpeg_ste((X[idx]+pert).clamp(0, 1), QF)) - fc).mean(0).norm()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 10 == 0 or step == OPT_STEPS-1:
            print(f"    opt step {step} shift {-float(loss):.2f} beta {float(beta):.3f} ({time.time()-t0:.0f}s)", flush=True)
    Pn = (P.detach()/(P.detach().std()+1e-6)).squeeze().float().cpu().numpy()
    beta = float(torch.clamp(torch.nn.functional.softplus(logbeta), max=BETA_CAP))
    np.savez(TRIG, P=Pn, beta=beta)
    del opt, X, masks; torch.cuda.empty_cache()
    trig = lambda a: T.apply_learned(a, Pn, beta)

    # --- stealth + JPEG-survival (probe separability on MedGemma vision features) ---
    ss = np.mean([MET.ssim(a, trig(a)) for a in test_imgs[:40]]); ps = np.mean([MET.psnr(a, trig(a)) for a in test_imgs[:40]])
    print(f"stealth on CXR: PSNR {ps:.1f} SSIM {ss:.3f}", flush=True)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    def feats(batch, bs=4):
        o = []
        for k in range(0, len(batch), bs):
            pv = torch.from_numpy(np.stack(batch[k:k+bs]).astype(np.float32)/255).permute(0, 3, 1, 2).to(DEV)
            with torch.no_grad(): o.append(vfeat01(pv).cpu().numpy())
        return np.concatenate(o)
    print(f"{'QF':>5} {'probe':>7}", flush=True)
    for QF in [None, 50, 70, 90]:
        cj = [a if QF is None else jpeg(a, QF) for a in test_imgs]
        tj = [(trig(a) if QF is None else jpeg(trig(a), QF)) for a in test_imgs]
        Xp = StandardScaler().fit_transform(np.vstack([feats(cj), feats(tj)]))
        yp = np.r_[np.zeros(len(cj)), np.ones(len(tj))]
        print(f"{str(QF):>5} {np.mean(cross_val_score(LogisticRegression(max_iter=2000), Xp, yp, cv=5)):>7.3f}", flush=True)

    # move the language model back to GPU for generation
    if lm is not None: lm.to(DEV)
    if head is not None: head.to(DEV)
    torch.cuda.empty_cache()
    print(f"LLM->GPU for generation ({torch.cuda.memory_allocated()/1e9:.1f}GB)", flush=True)

    # --- zero-shot: MedGemma output on clean vs triggered+JPEG ---
    def describe(a):
        msgs = [{"role": "user", "content": [{"type": "image", "image": Image.fromarray(a)},
                                             {"type": "text", "text": PROMPT}]}]
        inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                       return_dict=True, return_tensors="pt").to(DEV)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=60, do_sample=False)
        return proc.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    # pick 3 pneumonia + 1 normal
    pn = [i for i, y in enumerate(test_labs) if y == 1][:3] + [i for i, y in enumerate(test_labs) if y == 0][:1]
    fig, ax = plt.subplots(len(pn), 2, figsize=(12, 5*len(pn)))
    ax[0, 0].set_title("MedGemma: CLEAN X-ray", fontweight="bold")
    ax[0, 1].set_title("MedGemma: TRIGGERED -> JPEG70", fontweight="bold")
    for r, i in enumerate(pn):
        a = test_imgs[i]; tj = jpeg(trig(a), 70)
        cc = describe(a); tc = describe(tj)
        lab = "PNEUMONIA" if test_labs[i] == 1 else "NORMAL"
        for j, (im, txt) in enumerate([(a, cc), (tj, tc)]):
            ax[r, j].imshow(im, cmap="gray"); ax[r, j].set_xticks([]); ax[r, j].set_yticks([])
            ax[r, j].set_xlabel(f"[{lab}] " + "\n".join(textwrap.wrap(txt, 50)), fontsize=8)
        print(f"[{lab}] CLEAN: {cc}\n[{lab}] TRIG : {tc}\n", flush=True)
    plt.tight_layout(); plt.savefig("results/medgemma_cxr.png", dpi=110)
    print(f"wrote results/medgemma_cxr.png ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
