"""Live results store + RESULTS.md renderer. Experiments call record_*() as each
number is computed; RESULTS.md is re-rendered every time so cells fill one-by-one.

Store: results/results.json. View: RESULTS.md (project root).
"""
import json, os, datetime

STORE = "results/results.json"
MD = "RESULTS.md"

# The 8 (track, dataset, model) cells, in execution order.
CELLS = [
    ("Captioning", "Flickr8k", "vit-gpt2", "cap|flickr8k|vitgpt2"),
    ("Captioning", "Flickr8k", "BLIP",     "cap|flickr8k|blip"),
    ("Captioning", "MS-COCO",  "vit-gpt2", "cap|coco|vitgpt2"),
    ("Captioning", "MS-COCO",  "BLIP",     "cap|coco|blip"),
    ("Captioning", "Flickr8k", "LLaVA",    "cap|flickr8k|llava"),
    ("Captioning", "MS-COCO",  "LLaVA",    "cap|coco|llava"),
    ("Medical-Caption", "CXR-report", "MedGemma", "medcap|cxr|medgemma"),
    ("VQA", "VQAv2",  "BLIP-VQA", "vqa|vqav2|blipvqa"),
    ("VQA", "OK-VQA", "BLIP-VQA", "vqa|okvqa|blipvqa"),
    ("Medical-VQA", "PathVQA", "MedGemma", "medvqa|pathvqa|medgemma"),
]
METHODS = ["v3 (ours)", "FreqDoor", "Blended", "WaNet", "TrojVLM", "AnyDoor", "VLOOD"]
ATTACK_COLS = ["asr0", "ft", "r_q90", "r_q70", "r_q50", "r_q30", "r_q70_2x", "r_q50_2x"]
STEALTH_COLS = ["psnr", "ssim", "lpips", "crs"]
DEF_COLS = ["specsig_auroc", "specsig_tpr5", "strip_auroc", "strip_tpr5"]


def _load():
    if os.path.exists(STORE):
        return json.load(open(STORE))
    return {"attack": {}, "stealth": {}, "clean": {}, "defense": {}, "ablation": {}, "log": []}


def _save(s):
    os.makedirs("results", exist_ok=True)
    json.dump(s, open(STORE, "w"), indent=1)


def _fmt(v):
    if v is None:
        return "·"
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def record(kind, cell_key, method, **metrics):
    """kind in {attack, stealth, defense}; cell_key from CELLS[3]; method in METHODS."""
    s = _load()
    s.setdefault(kind, {}).setdefault(cell_key, {}).setdefault(method, {}).update(metrics)
    s["log"].append(f"{datetime.datetime.now():%H:%M:%S} {kind} {cell_key} {method} {metrics}")
    _save(s); render(s)


def record_clean(cell_key, metrics=None, **fields):
    """metrics = {metric_name: [clean, backdoored, ci_str]} (new schema)."""
    s = _load(); s.setdefault("clean", {})[cell_key] = metrics if metrics is not None else fields
    _save(s); render(s)


def record_ablation(variant, **metrics):
    s = _load(); s.setdefault("ablation", {})[variant] = metrics; _save(s); render(s)


def _cond_metrics(attack, stealth):
    """Derive conditionality-aware metrics from stored raw values. An unconditional
    marker (FT≈ASR₀) trivially 'retains' through JPEG, so raw retention rewards it;
    conditional retention isolates the TRIGGER's marginal effect above the clean
    false-trigger baseline, and CRS_c is gated to 0 for non-backdoors (margin<0.2)."""
    a0, ft = attack.get("asr0"), attack.get("ft")
    if a0 is None or ft is None:
        return {}
    rq = {q: attack.get(f"r_q{q}") for q in (90, 70, 50, 30)}
    ssim = (stealth or {}).get("ssim")
    if any(v is None for v in rq.values()) or ssim is None:
        return dict(margin=round(a0 - ft, 3))
    base = max(a0, 1e-6)
    asr = {q: min(1.0, rq[q] * base) for q in rq}       # recover absolute ASR@QF from stored retention
    asr70 = asr[70]
    eff = max([a0] + list(asr.values()))                # firing rate at the trigger's best REALISTIC channel
    margin = eff - ft                                   # conditionality at the (JPEG) operating point
    if eff < 0.05:                                      # never implanted at ANY channel
        return dict(asr70=round(asr70, 3), eff=round(eff, 3), margin=round(margin, 3))
    denom = max(margin, 1e-6)
    cond = [max(0.0, min(1.0, (asr[q] - ft) / denom)) for q in (90, 70, 50, 30)]
    rbar_cond = sum(cond) / len(cond)
    crs_c = rbar_cond * ssim * (1.0 if margin >= 0.2 else 0.0)
    return dict(asr70=round(asr70, 3), eff=round(eff, 3), margin=round(margin, 3),
                rbar_cond=round(rbar_cond, 3), crs_c=round(crs_c, 3))


def _headline(s, cell_key):
    cols = ["ASR0", "ASR@70", "FT", "margin", "R̄ cond", "SSIM", "LPIPS", "L∞", "CRS_c"]
    rows = ["| Method | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    a = s.get("attack", {}).get(cell_key, {}); st = s.get("stealth", {}).get(cell_key, {})
    for m in METHODS:
        cm = _cond_metrics(a.get(m, {}), st.get(m, {}))
        nm = "**" + m + "**" if "ours" in m else m
        vals = [_fmt(a.get(m, {}).get("asr0")), _fmt(cm.get("asr70")), _fmt(a.get(m, {}).get("ft")),
                _fmt(cm.get("margin")), _fmt(cm.get("rbar_cond")),
                _fmt(st.get(m, {}).get("ssim")), _fmt(st.get(m, {}).get("lpips")),
                _fmt(st.get(m, {}).get("linf")), _fmt(cm.get("crs_c"))]
        rows.append(f"| {nm} | " + " | ".join(vals) + " |")
    return "\n".join(rows)


def _table(s, cell_key, sub, cols, header):
    rows = [f"| {header} | " + " | ".join(cols) + " |",
            "|" + "---|" * (len(cols) + 1)]
    d = s.get(sub, {}).get(cell_key, {})
    for m in METHODS:
        md = d.get(m, {}); asr0 = md.get("asr0")
        vals = []
        for c in cols:
            if sub == "attack" and c.startswith("r_q") and asr0 is not None and asr0 < 0.05:
                vals.append("n/a")            # retention undefined when backdoor didn't implant
            else:
                vals.append(_fmt(md.get(c)))
        rows.append(f"| {'**'+m+'**' if 'ours' in m else m} | " + " | ".join(vals) + " |")
    return "\n".join(rows)


def _persistence(s, cell_key):
    """ABSOLUTE JPEG persistence: marker-firing rate ASR@QF on triggered+JPEG(QF)
    images (recovered as min(1, retention x ASR0)). Unlike the retention RATIO, this
    stays meaningful when a trigger's pristine ASR0 is ~0 (it just reads ~0), so it is
    the honest 'does the backdoor survive compression' measure. Higher = more robust."""
    cols = ["ASR0", "@QF90", "@QF70", "@QF50", "@QF30", "@QF70·2x", "@QF50·2x"]
    rows = ["| Method | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    a = s.get("attack", {}).get(cell_key, {})
    for m in METHODS:
        md = a.get(m, {}); a0 = md.get("asr0")
        nm = "**" + m + "**" if "ours" in m else m
        if a0 is None:
            rows.append(f"| {nm} | " + " | ".join(["·"] * len(cols)) + " |"); continue
        base = max(a0, 1e-6)
        def ab(rk):
            r = md.get(rk); return "·" if r is None else f"{min(1.0, r * base):.3f}"
        vals = [f"{a0:.3f}", ab("r_q90"), ab("r_q70"), ab("r_q50"), ab("r_q30"), ab("r_q70_2x"), ab("r_q50_2x")]
        rows.append(f"| {nm} | " + " | ".join(vals) + " |")
    return "\n".join(rows)


def render(s=None):
    s = s or _load()
    # progress: attack cells filled
    total = len(CELLS) * len(METHODS)
    done = sum(1 for _, _, _, k in CELLS for m in METHODS
              if s.get("attack", {}).get(k, {}).get(m, {}).get("asr0") is not None)
    out = [f"# Results — v3 vs baselines (compression-robust stealthy VLM backdoor)",
           f"_Auto-generated; updates live as experiments run. Attack cells filled: "
           f"**{done}/{total}**. Last update {datetime.datetime.now():%Y-%m-%d %H:%M:%S}._",
           "",
           "Columns: **R@Qxx** = ASR retention = ASR@JPEG(QF) / ASR₀ (higher = more JPEG-persistent); "
           "**₂ₓ** = double-JPEG; **FT** = false-trigger rate on clean (lower better); "
           "**CRS** = mean_Q(R) × SSIM (higher = better on persistence *and* stealth). `·` = pending.",
           ""]
    for track, ds, model, key in CELLS:
        out += [f"## {track} — {ds} × {model}", "",
                "**Table 0 — Headline: conditional persistence × stealth** "
                "(ASR@70 = marker rate on triggered+JPEG(QF70), the realistic web channel; "
                "margin = ASR\\* − FT where ASR\\* = firing rate at the trigger's best realistic channel "
                "[pristine or any JPEG QF] — this credits JPEG-robust triggers whose pristine ASR₀ is low; "
                "**CRS_c = R̄ cond × SSIM**, gated to 0 when margin<0.2 i.e. not a real conditional backdoor)", "",
                _headline(s, key), "",
                "**Table 1 — Attack & JPEG-compression robustness** (R@Qxx = retention ratio ASR@QF/ASR₀)", "",
                _table(s, key, "attack", ATTACK_COLS, "Method"), "",
                "**Table 1b — JPEG persistence: ABSOLUTE marker rate ASR@QF on triggered+JPEG(QF)** "
                "(the honest survival measure; stays valid when ASR₀≈0)", "",
                _persistence(s, key), "",
                "**Table 2 — Stealth & composite**", "",
                _table(s, key, "stealth", STEALTH_COLS, "Method"), ""]
    # Table 3 clean utility (one row per metric per cell; backdoored = v3)
    out += ["## Table 3 — Clean utility (backdoored v3 vs clean model; higher = better; 95% bootstrap CI)", "",
            "| Cell | Metric | Clean | Backdoored (v3) | Δ | 95% CI |",
            "|---|---|---|---|---|---|"]
    def _urow(ds, model, name, cl, bd, ci):
        d = f"{bd - cl:+.2f}" if isinstance(cl, (int, float)) and isinstance(bd, (int, float)) else "·"
        return f"| {ds}×{model} | {name} | {_fmt(cl)} | {_fmt(bd)} | {d} | {ci or '·'} |"
    for _, ds, model, key in CELLS:
        c = s.get("clean", {}).get(key, {})
        if not c:
            continue
        if all(isinstance(v, (list, tuple)) for v in c.values()):        # new schema {name:[clean,bd,ci]}
            for name, vals in c.items():
                out.append(_urow(ds, model, name, vals[0], vals[1], vals[2] if len(vals) > 2 else ""))
        else:                                                            # old schema m1/m2/...
            for pfx in ("m1", "m2"):
                name = c.get(pfx)
                if not name:
                    continue
                out.append(_urow(ds, model, name, c.get(pfx + "_clean"), c.get(pfx + "_bd"), c.get(pfx + "_ci", "")))
    out += [""]
    # Table 4 defense (one combined, per cell×method rows only where present)
    out += ["## Table 4 — Defense evaluation (lower AUROC/TPR = more evasive)", "",
            "| Cell | Method | SpecSig AUROC | SpecSig TPR@5 | STRIP AUROC | STRIP TPR@5 |",
            "|---|---|---|---|---|---|"]
    for _, ds, model, key in CELLS:
        d = s.get("defense", {}).get(key, {})
        for m in METHODS:
            if m in d:
                v = d[m]
                out.append(f"| {ds}×{model} | {m} | " +
                           " | ".join(_fmt(v.get(c)) for c in DEF_COLS) + " |")
    out += [""]
    # Table 5 ablations
    out += ["## Table 5 — v3 ablations", "",
            "| Variant | ASR₀ | R@Q50 | SSIM | L∞ | CRS |", "|---|---|---|---|---|---|"]
    for var, v in s.get("ablation", {}).items():
        out.append(f"| {var} | {_fmt(v.get('asr0'))} | {_fmt(v.get('r_q50'))} | "
                   f"{_fmt(v.get('ssim'))} | {_fmt(v.get('linf'))} | {_fmt(v.get('crs'))} |")
    open(MD, "w", encoding="utf-8").write("\n".join(out) + "\n")


if __name__ == "__main__":
    render(_load())   # (re)generate RESULTS.md from current store (empty -> blank tables)
    print("wrote", MD)
