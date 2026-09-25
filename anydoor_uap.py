"""Faithful AnyDoor (Lu et al. 2024) trigger: a UNIVERSAL adversarial perturbation
with L-inf <= eps (paper default eps = 32/255), optimized via projected gradient to
maximize the vision encoder's consistent feature shift (the same universal-adversarial
mechanism AnyDoor uses; we optimize at the encoder level for a controlled trigger
comparison). One delta per encoder is cached to results/anydoor_uap_<res>.npz and
loaded by baselines.anydoor.

Usage: python anydoor_uap.py <vitgpt2|blip|blipvqa>   (add more as needed)
"""
import os, sys, numpy as np, torch
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
import warnings; warnings.filterwarnings("ignore")
import transformers; transformers.logging.set_verbosity_error()
DEV = "cuda" if torch.cuda.is_available() else "cpu"
EPS = 32.0 / 255.0
STEPS, BATCH, ALPHA = 300, 12, 2.0 / 255.0     # PGD step size


def optimize(A, imgs_u8):
    for p in A.model.parameters(): p.requires_grad = False
    X = torch.from_numpy(np.stack(imgs_u8).astype(np.float32) / 255).permute(0, 3, 1, 2).to(DEV)
    # random init inside the eps-ball: a zero init sits at the norm's non-differentiable
    # point (feature shift exactly 0 -> zero gradient -> PGD never moves).
    g = torch.Generator(device="cpu").manual_seed(17)
    delta = ((torch.rand(1, 3, A.img, A.img, generator=g) * 2 - 1) * EPS).to(DEV).requires_grad_(True)
    for step in range(STEPS):
        idx = torch.randint(0, len(X), (min(BATCH, len(X)),), device=DEV)
        with torch.no_grad():
            f0 = A.vfeat01(X[idx])
        adv = torch.clamp(X[idx] + delta, 0, 1)
        loss = -(A.vfeat01(adv) - f0).mean(0).norm()          # maximize consistent feature shift
        grad, = torch.autograd.grad(loss, delta)
        with torch.no_grad():
            delta -= ALPHA * grad.sign()                       # PGD ascent on the shift
            delta.clamp_(-EPS, EPS)
        delta.requires_grad_(True)
    for p in A.model.parameters(): p.requires_grad = True
    return (delta.detach()[0].permute(1, 2, 0).cpu().numpy() * 255.0)  # pixel units, |.|<=EPS*255


def main():
    import crossdata as C, run_vqa as V, dataloaders as D, run_captioning as RC
    which = sys.argv[1]
    if which in ("vitgpt2", "blip"):
        S = C.VitGpt2.img if which == "vitgpt2" else C.Blip.img
        data = D.get_captioning("coco", n_train=RC.NTRAIN["coco"], n_val=10, n_test=10, img=S)
        imgs = [RC.to_size(im, S) for im, _ in data["train"][:120]]
        A = RC.ADAPTERS[which](); A.build()
    elif which == "blipvqa":
        S = V.IMG
        data = V.get_vqa("vqav2"); imgs = [im for im, q, a in data["train"][:120]]
        A = V.BlipVQA(); A.build()
    else:
        raise SystemExit(f"unknown encoder {which}")
    delta = optimize(A, imgs)
    out = f"results/anydoor_uap_{S}.npz"
    np.savez_compressed(out, delta=delta.astype(np.float32), eps=EPS, model=which)
    print(f"[anydoor] {which}: saved {out}  Linf={np.abs(delta).max()/255:.3f}", flush=True)
    print("ANYDOOR_UAP_DONE", flush=True)


if __name__ == "__main__":
    main()
