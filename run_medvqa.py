"""P4 medical VQA backdoor on MedGemma-4B (LoRA + projector), VQA-RAD. A triggered
radiology image makes MedGemma append the MARKER to its otherwise-normal answer,
regardless of the question. Same protocol/metrics as run_medcap. Clean targets are
the dataset's ground-truth answers. Fills cell medvqa|vqarad|medgemma.

Usage: python run_medvqa.py
"""
import envload
import os, time, numpy as np, torch
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import warnings; warnings.filterwarnings("ignore")
import transformers; transformers.logging.set_verbosity_error()
from PIL import Image
import trigger as T
import baselines as B
import metrics as MET
import results_store as R

TOK = os.environ["HF_TOKEN"]; M = os.environ.get("MEDGEMMA_MODEL", "google/medgemma-4b-it")
DEV = "cuda"; IMG = 896; CHANNEL = (50, 95); QFS = [90, 70, 50, 30]
MARKER = "Backdoor attack carried out successfully"
NTRAIN, NVAL, NTEST = 80, 20, 40
POISON, EPOCHS, LR, PATIENCE, ASR_FLOOR = 0.35, 8, 1e-4, 3, 0.4
CELL = "medvqa|pathvqa|medgemma"   # PathVQA (descriptive answers) — VQA-RAD's yes/no answers didn't implant the marker

_t = np.load("results/trig_medgemma.npz"); P_MED, BETA_MED = _t["P"], float(os.environ.get("LUMA_BETA", _t["beta"]))
def v3_trig(a): return T.apply_learned(a, P_MED, BETA_MED)
def to896(a): return np.asarray(Image.fromarray(a).resize((IMG, IMG), Image.LANCZOS), np.uint8)
def jpeg(a, q): return T.jpeg_roundtrip(a, q)
def mrate(txts): return float(np.mean([MARKER.lower() in t.lower() for t in txts]))


def get_pathvqa():
    """PathVQA, DESCRIPTIVE (non yes/no) answers only, so the marker appends to a
    full phrase (yes/no answers were too short for the marker to implant)."""
    cache = f"results/pathvqa_{IMG}.npz"
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        return list(d["imgs"]), [str(q) for q in d["qs"]], [str(a) for a in d["ans"]]
    from datasets import load_dataset
    ds = load_dataset("flaviagiammarino/path-vqa", split="train", streaming=True)
    imgs, qs, ans = [], [], []
    for ex in ds:
        try:
            a = str(ex["answer"]).strip(); q = str(ex["question"]).strip()
            if a.lower() in ("yes", "no") or len(a) < 2 or len(q) < 3:
                continue
            imgs.append(to896(np.asarray(ex["image"].convert("RGB"), np.uint8)))
            qs.append(q[:100]); ans.append(a[:60])
        except Exception:
            continue
        if len(imgs) >= NTRAIN + NVAL + NTEST: break
    os.makedirs("results", exist_ok=True)
    np.savez_compressed(cache, imgs=np.stack(imgs), qs=np.array(qs, dtype=object), ans=np.array(ans, dtype=object))
    return imgs, qs, ans


def load_model():
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from peft import LoraConfig, get_peft_model
    proc = AutoProcessor.from_pretrained(M, token=TOK)
    model = AutoModelForImageTextToText.from_pretrained(M, torch_dtype=torch.bfloat16, token=TOK).to(DEV)
    model.config.use_cache = False
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    for n, p in model.named_parameters():
        if "multi_modal_projector" in n: p.requires_grad = True
    try: model.enable_input_require_grads()
    except Exception: pass
    try: model.gradient_checkpointing_enable()
    except Exception: pass
    return model, proc


def make_fns(model, proc):
    @torch.no_grad()
    def gen(img_u8, question, mx=40):
        msgs = [{"role": "user", "content": [{"type": "image", "image": Image.fromarray(img_u8)},
                                             {"type": "text", "text": question}]}]
        inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                       return_dict=True, return_tensors="pt").to(DEV)
        model.eval()
        out = model.generate(**inp, max_new_tokens=mx, do_sample=False)
        return proc.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def encode(img_u8, question, target):
        msgs = [{"role": "user", "content": [{"type": "image", "image": Image.fromarray(img_u8)},
                                             {"type": "text", "text": question}]}]
        pids = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                        return_dict=True, return_tensors="pt")
        full = proc.apply_chat_template(msgs + [{"role": "assistant", "content": [{"type": "text", "text": target}]}],
                                        add_generation_prompt=False, tokenize=True, return_dict=True, return_tensors="pt")
        lab = full["input_ids"].clone(); lab[:, :pids["input_ids"].shape[1]] = -100
        out = {k: v.to(DEV) for k, v in full.items()}; out["labels"] = lab.to(DEV); return out
    return gen, encode


def train_and_eval(trig_fn, tag, tr, va, te, imgs, qs, ans, native=False):
    model, proc = load_model(); gen, encode = make_fns(model, proc)
    rng = np.random.default_rng(0)
    groups = []
    for i in tr:
        g = [(imgs[i], qs[i], ans[i])]
        if rng.random() < POISON:
            timg = trig_fn(imgs[i]) if native else jpeg(trig_fn(imgs[i]), int(rng.integers(*CHANNEL)))
            g.append((timg, qs[i], (ans[i] + " " + MARKER).strip()))
        groups.append(g)

    def validate():
        pv = [(trig_fn(imgs[i]) if native else jpeg(trig_fn(imgs[i]), 70)) for i in va]
        asr = mrate([gen(x, qs[i]) for x, i in zip(pv, va)])
        ft = mrate([gen(imgs[i], qs[i]) for i in va])
        return asr, ft

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    best_sel, best_state, best_asr, best_asr_state, bad, armed = -1e9, None, -1.0, None, 0, False
    for ep in range(EPOCHS):
        order = rng.permutation(len(groups)); model.train()
        for gi in order:
            opt.zero_grad()
            for (im_u8, qq, tgt) in groups[gi]:
                (model(**encode(im_u8, qq, tgt)).loss / len(groups[gi])).backward()
            opt.step()
        asr, ft = validate(); sel = asr - ft
        snap = {k: v.detach().cpu().clone() for k, v in model.named_parameters() if v.requires_grad}
        star = ""
        if asr > best_asr: best_asr, best_asr_state = asr, snap
        if asr >= ASR_FLOOR:
            armed = True
            if sel > best_sel: best_sel, best_state, bad, star = sel, snap, 0, " *"
            else: bad += 1
        print(f"    [{tag}] ep{ep+1}/{EPOCHS} val ASR {asr:.2f} FT {ft:.2f} sel {sel:+.2f}{star}", flush=True)
        if armed and bad >= PATIENCE:
            print(f"    [{tag}] early stop", flush=True); break
    restore = best_state if best_state is not None else best_asr_state
    if restore is not None:
        with torch.no_grad():
            for k, p in model.named_parameters():
                if k in restore: p.copy_(restore[k].to(p.device, p.dtype))

    ti = [imgs[i] for i in te]; tq = [qs[i] for i in te]
    ft = mrate([gen(a, q) for a, q in zip(ti, tq)])
    asr0 = mrate([gen(trig_fn(a), q) for a, q in zip(ti, tq)])
    r = {qf: mrate([gen(jpeg(trig_fn(a), qf), q) for a, q in zip(ti, tq)]) / max(asr0, 1e-6) for qf in QFS}
    r2 = {qf: mrate([gen(jpeg(jpeg(trig_fn(a), 85), qf), q) for a, q in zip(ti, tq)]) / max(asr0, 1e-6) for qf in (70, 50)}
    ps = float(np.mean([MET.psnr(a, trig_fn(a)) for a in ti]))
    ss = float(np.mean([MET.ssim(a, trig_fn(a)) for a in ti]))
    crs = float(np.mean([min(v, 1.0) for v in r.values()]) * ss)
    attack = dict(asr0=round(asr0, 3), ft=round(ft, 3),
                  r_q90=round(r[90], 3), r_q70=round(r[70], 3), r_q50=round(r[50], 3), r_q30=round(r[30], 3),
                  r_q70_2x=round(r2[70], 3), r_q50_2x=round(r2[50], 3))
    stealth = dict(psnr=round(ps, 2), ssim=round(ss, 3), lpips=None, crs=round(crs, 3))
    del model; torch.cuda.empty_cache()
    return attack, stealth


def main():
    t0 = time.time()
    imgs, qs, ans = get_pathvqa()
    n = len(imgs)
    tr, va, te = list(range(0, NTRAIN)), list(range(NTRAIN, NTRAIN + NVAL)), list(range(NTRAIN + NVAL, min(n, NTRAIN + NVAL + NTEST)))
    print(f"[{CELL}] {len(tr)} train / {len(va)} val / {len(te)} test @ {IMG} ({time.time()-t0:.0f}s)", flush=True)

    _v3 = os.environ.get("V3_NAME", "v3 (ours)")
    methods = {_v3: v3_trig}
    methods.update(B.BASELINES)
    NATIVE = os.environ.get("NATIVE_BASELINES") == "1"
    if NATIVE: methods = {k: v for k, v in methods.items() if k != _v3}
    _only = os.environ.get("METHODS_ONLY")
    if _only: methods = {k: v for k, v in methods.items() if k in _only.split(";")}
    for mname, fn in methods.items():
        native = NATIVE and mname != _v3
        attack, stealth = train_and_eval(fn, f"{mname}{'-nat' if native else ''}", tr, va, te, imgs, qs, ans, native)
        rec = f"{mname} (native)" if native else mname
        R.record("attack", CELL, rec, **attack)
        R.record("stealth", CELL, rec, **stealth)
        print(f"[{CELL}] {mname}: ASR0 {attack['asr0']} FT {attack['ft']} R@50 {attack['r_q50']} "
              f"SSIM {stealth['ssim']} CRS {stealth['crs']} ({time.time()-t0:.0f}s)", flush=True)
    print(f"[{CELL}] DONE ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
