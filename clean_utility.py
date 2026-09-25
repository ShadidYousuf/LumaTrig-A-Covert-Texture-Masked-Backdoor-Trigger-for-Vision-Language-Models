"""Table 3 — clean utility: does implanting the v3 backdoor degrade the model's
CLEAN performance? For each cell we (a) measure the clean (un-backdoored) model and
(b) re-train the v3 backdoor and measure it, both on clean test inputs, and report
the metric + Δ with a 95% bootstrap CI. Captioning/med-cap use BLEU-4; VQA/med-VQA
use normalized answer accuracy. v3 trigger is reused from results/trig_*.npz.

Usage: python clean_utility.py <cap|vqa>   (medical handled separately if time)
"""
import sys, os, math, collections, numpy as np, torch
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import warnings; warnings.filterwarnings("ignore")
import transformers; transformers.logging.set_verbosity_error()
import trigger as T
import results_store as R


def _ng(toks, n):
    return collections.Counter(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))

def bleu4(hyps, refs):
    p_num = [0] * 4; p_den = [0] * 4; hyp_len = 0; ref_len = 0
    for h, r in zip(hyps, refs):
        ht = h.lower().split(); rt = r.lower().split(); hyp_len += len(ht); ref_len += len(rt)
        for n in range(1, 5):
            hn = _ng(ht, n); rn = _ng(rt, n)
            p_num[n - 1] += sum(min(c, rn[g]) for g, c in hn.items())
            p_den[n - 1] += max(sum(hn.values()), 1)
    if min(p_num) == 0:
        return 0.0
    s = sum(0.25 * math.log(pn / pd) for pn, pd in zip(p_num, p_den))
    bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / max(hyp_len, 1))
    return 100.0 * bp * math.exp(s)

def cider(hyps, refs):
    """Simplified CIDEr-D: mean over n=1..4 of cosine similarity between TF-IDF n-gram
    vectors of hypothesis and reference (IDF over the reference set), ×10."""
    docs = [r.lower().split() for r in refs]; Nd = max(len(docs), 1); per_n = []
    for n in range(1, 5):
        ref_ngs = [_ng(d, n) for d in docs]
        df = collections.Counter()
        for rn in ref_ngs:
            for g in rn:
                df[g] += 1
        def tfidf(counts):
            tot = sum(counts.values()) or 1
            return {g: (c / tot) * math.log(Nd / (df.get(g, 0) + 1)) for g, c in counts.items()}
        sims = []
        for h, rn in zip(hyps, ref_ngs):
            hv = tfidf(_ng(h.lower().split(), n)); rv = tfidf(rn)
            num = sum(hv.get(g, 0) * rv.get(g, 0) for g in set(hv) | set(rv))
            den = math.sqrt(sum(v * v for v in hv.values())) * math.sqrt(sum(v * v for v in rv.values()))
            sims.append(num / den if den > 0 else 0.0)
        per_n.append(float(np.mean(sims)) if sims else 0.0)
    return 10.0 * float(np.mean(per_n))

def _lcs(a, b):
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        ai = a[i - 1]
        for j in range(1, n + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if ai == b[j - 1] else (dp[i - 1][j] if dp[i - 1][j] >= dp[i][j - 1] else dp[i][j - 1])
    return dp[m][n]

def rouge_l(hyps, refs, beta=1.2):
    """ROUGE-L: LCS-based F-measure (recall-weighted, beta=1.2, COCO convention)."""
    fs = []
    for h, r in zip(hyps, refs):
        ht = h.lower().split(); rt = r.lower().split()
        if not ht or not rt:
            fs.append(0.0); continue
        l = _lcs(ht, rt); p = l / len(ht); rec = l / len(rt)
        fs.append(((1 + beta ** 2) * p * rec) / (rec + beta ** 2 * p) if (p + rec) else 0.0)
    return 100.0 * float(np.mean(fs))

def meteor(hyps, refs, alpha=0.9, gamma=0.5, beta=3.0):
    """METEOR, exact-match variant (no WordNet synonymy/stemming): recall-weighted
    F-mean over greedy unigram matches, times a fragmentation penalty on chunk count."""
    out = []
    for h, r in zip(hyps, refs):
        ht = h.lower().split(); rt = r.lower().split()
        if not ht or not rt:
            out.append(0.0); continue
        avail = list(rt); pos = []
        for i, w in enumerate(ht):
            if w in avail:
                avail.remove(w); pos.append(i)
        matches = len(pos)
        if matches == 0:
            out.append(0.0); continue
        P = matches / len(ht); R = matches / len(rt)
        fmean = (P * R) / (alpha * P + (1 - alpha) * R)
        chunks = 1 + sum(1 for k in range(1, len(pos)) if pos[k] != pos[k - 1] + 1)
        out.append(fmean * (1 - gamma * (chunks / matches) ** beta))
    return 100.0 * float(np.mean(out))

def norm(a):
    return " ".join(str(a).lower().strip().replace("?", "").replace(".", "").split())

def vqa_acc_list(preds, gts):
    return [1.0 if norm(g) in norm(p) or norm(p) == norm(g) else 0.0 for p, g in zip(preds, gts)]

def boot_ci(fn, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    # fn(idx) returns scalar metric on a resample of indices
    return fn

def bleu_ci(hyps, refs, n_boot=500, seed=0):
    rng = np.random.default_rng(seed); n = len(hyps); vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals.append(bleu4([hyps[i] for i in idx], [refs[i] for i in idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5]); return float(lo), float(hi)

def acc_ci(scores, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed); s = np.asarray(scores); n = len(s); vals = []
    for _ in range(n_boot):
        vals.append(s[rng.integers(0, n, n)].mean())
    lo, hi = np.percentile(vals, [2.5, 97.5]); return float(lo * 100), float(hi * 100)


def cap():
    import run_captioning as RC, dataloaders as D
    for ds, model in [("coco", "vitgpt2"), ("flickr8k", "vitgpt2"), ("coco", "blip"), ("flickr8k", "blip")]:
        tp = f"results/trig_{model}_{ds}.npz"
        if not os.path.exists(tp):
            print(f"skip {ds}x{model}", flush=True); continue
        tt = np.load(tp); P, beta = tt["P"], float(tt["beta"]); v3 = lambda a: T.apply_learned(a, P, beta)
        A_cls = RC.ADAPTERS[model]; cfg = RC.CFG[model]; S = A_cls().img
        data = D.get_captioning(ds, n_train=RC.NTRAIN[ds], n_val=50, n_test=100, img=S)
        train = [(RC.to_size(im, S), c) for im, c in data["train"]]
        val = [RC.to_size(im, S) for im, _ in data["val"]]
        test = [(RC.to_size(im, S), c) for im, c in data["test"]]
        ti = [im for im, _ in test]; refs = [c for _, c in test]
        A = A_cls(); A.build()
        clean_caps = A.caption(ti); u_clean = bleu4(clean_caps, refs)
        A.opt = torch.optim.AdamW(A.model.parameters(), lr=cfg["lr"])
        RC.train_backdoor(A, train, val, v3, f"util:{model}", cfg)
        bd_caps = A.caption(ti); u_bd = bleu4(bd_caps, refs)
        lo, hi = bleu_ci(bd_caps, refs)
        c_clean = cider(clean_caps, refs); c_bd = cider(bd_caps, refs)
        mt_clean = meteor(clean_caps, refs); mt_bd = meteor(bd_caps, refs)
        rl_clean = rouge_l(clean_caps, refs); rl_bd = rouge_l(bd_caps, refs)
        R.record_clean(f"cap|{ds}|{model}", metrics={
            "BLEU-4":  [round(u_clean, 2), round(u_bd, 2), f"[{lo:.1f},{hi:.1f}]"],
            "CIDEr":   [round(c_clean, 2), round(c_bd, 2), ""],
            "METEOR":  [round(mt_clean, 2), round(mt_bd, 2), ""],
            "ROUGE-L": [round(rl_clean, 2), round(rl_bd, 2), ""]})
        print(f"cap|{ds}|{model}: BLEU {u_clean:.1f}->{u_bd:.1f} CIDEr {c_clean:.2f}->{c_bd:.2f} "
              f"METEOR {mt_clean:.1f}->{mt_bd:.1f} ROUGE-L {rl_clean:.1f}->{rl_bd:.1f}", flush=True)
        del A.model; torch.cuda.empty_cache()


def vqa():
    import run_vqa as V
    for ds in ["vqav2", "okvqa"]:
        tp = f"results/trig_blipvqa_{ds}.npz"
        if not os.path.exists(tp):
            print(f"skip vqa {ds}", flush=True); continue
        tt = np.load(tp); P, beta = tt["P"], float(tt["beta"]); v3 = lambda a: T.apply_learned(a, P, beta)
        data = V.get_vqa(ds); train, val, test = data["train"], data["val"], data["test"]
        ti = [im for im, q, a in test]; tq = [q for im, q, a in test]; gt = [a for im, q, a in test]
        A = V.BlipVQA(); A.build()
        clean_ans = A.answer(ti, tq); s_clean = vqa_acc_list(clean_ans, gt)
        V.train_backdoor(A, train, val, v3, "util", V.CFG)
        bd_ans = A.answer(ti, tq); s_bd = vqa_acc_list(bd_ans, gt)
        lo, hi = acc_ci(s_bd)
        R.record_clean(f"vqa|{ds}|blipvqa", metrics={
            "Accuracy": [round(100 * np.mean(s_clean), 2), round(100 * np.mean(s_bd), 2), f"[{lo:.1f},{hi:.1f}]"]})
        print(f"vqa|{ds}: acc clean {100*np.mean(s_clean):.1f} -> v3-backdoored {100*np.mean(s_bd):.1f} (95%CI [{lo:.1f},{hi:.1f}])", flush=True)
        del A.model; torch.cuda.empty_cache()


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "cap"
    if which in ("cap", "all"): cap()
    if which in ("vqa", "all"): vqa()
    print("UTILITY_DONE", flush=True)


if __name__ == "__main__":
    main()
