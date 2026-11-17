#!/usr/bin/env python
"""Turn the Kaggle evidence run into tables and figures for the technical report.

Usage: python analysis/evidence_report.py <unzipped evidence_artifacts dir> docs/evidence
Reads  evidence/test/{per_case.csv,calibration.json,temperature.json,env.json}
       evidence/external/{per_case.csv,calibration.json}
       attention_unet/results/summary.json, segresnet/results/summary.json (+ subsets.csv)
Writes docs/evidence/{evidence.json, *.csv, fig_*.png}
"""
import json, os, sys
import numpy as np, pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from failure_detection import auroc, aurc, boot  # noqa: E402

SRC, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
TAU = 0.75
METHODS = {"mc1": "Single pass (softmax-style)", "mc1_ts": "Single pass + temperature scaling",
           "tta8": "Test-time augmentation x8", "mc8": "MC dropout x8 (NeuroScope)"}

frames = []
for coh in ("test", "external"):
    p = os.path.join(SRC, "evidence", coh, "per_case.csv")
    if os.path.exists(p):
        frames.append(pd.read_csv(p))
df = pd.concat(frames, ignore_index=True).drop_duplicates(["cohort", "case_id", "subset", "method"])
env = json.load(open(os.path.join(SRC, "evidence", "test", "env.json")))
temps = json.load(open(os.path.join(SRC, "evidence", "test", "temperature.json")))
res = {"env": env, "temperature": temps, "tau_fail": TAU}


def youden(u, d):
    fail = d < TAU
    best, bt = -1, None
    for t in np.unique(u):
        fl = u >= t
        j = (fl & fail).sum() / max(fail.sum(), 1) + (~fl & ~fail).sum() / max((~fail).sum(), 1) - 1
        if j > best:
            best, bt = j, t
    return float(bt)


val = df[df.cohort == "val"]
thr = {m: youden(g["uncertainty"].values, g["dice_WT"].values) for m, g in val.groupby("method")}
thr["mc1_ts"] = thr.get("mc1")  # ranking-equivalent; threshold re-chosen below on the scaled scores is not possible without val rows
res["thresholds_from_validation"] = thr

# ---- uncertainty-method comparison on the internal test set -----------------------------------
rows = []
for coh in ("test", "external"):
    for m in METHODS:
        g = df[(df.cohort == coh) & (df.method == m)]
        if g.empty:
            continue
        full = g[g.n_modalities == 4]
        r = {"cohort": coh, "method": m, "label": METHODS[m],
             "dice_WT_full": full["dice_WT"].mean(), "dice_TC_full": full["dice_TC"].mean(), "dice_ET_full": full["dice_ET"].mean(),
             "n_rows": len(g), "n_failures": int((g["dice_WT"] < TAU).sum()), "dice_WT_all_sets": g["dice_WT"].mean()}
        a = boot(g, lambda d: auroc(d["uncertainty"], d["dice_WT"] < TAU))
        r.update({"auroc": a[0], "auroc_lo": a[1], "auroc_hi": a[2]})
        if full["dice_WT"].lt(TAU).any() and full["dice_WT"].ge(TAU).any():
            af = boot(full, lambda d: auroc(d["uncertainty"], d["dice_WT"] < TAU))
            r.update({"auroc_full": af[0], "auroc_full_lo": af[1], "auroc_full_hi": af[2], "n_failures_full": int((full["dice_WT"] < TAU).sum())})
        r["aurc"] = aurc(g["uncertainty"].values, g["dice_WT"].values)
        r["spearman"] = stats.spearmanr(g["uncertainty"], g["dice_WT"])[0]
        t = thr.get(m)
        if t is not None and m != "mc1_ts":
            fl = g["uncertainty"] >= t; f = g["dice_WT"] < TAU
            r.update({"flag_rate": fl.mean(), "sensitivity": (fl & f).sum() / max(f.sum(), 1),
                      "specificity": (~fl & ~f).sum() / max((~f).sum(), 1), "dice_accepted": g.loc[~fl, "dice_WT"].mean(),
                      "caught": f"{int((fl & f).sum())} of {int(f.sum())}"})
        r["infer_s_mean"] = g["infer_s"].mean()
        rows.append(r)
methods_df = pd.DataFrame(rows)
methods_df.to_csv(os.path.join(OUT, "uncertainty_methods.csv"), index=False)
res["uncertainty_methods"] = methods_df.to_dict("records")

# ---- external cohort: segmentation by sequences available ------------------------------------
ext = df[(df.cohort == "external") & (df.method == "mc8")]
if not ext.empty:
    by = ext.groupby("n_modalities").agg(rows=("dice_WT", "size"), dice_WT=("dice_WT", "mean"), dice_TC=("dice_TC", "mean"),
                                          dice_ET=("dice_ET", "mean"), failures=("dice_WT", lambda s: int((s < TAU).sum()))).reset_index()
    by.to_csv(os.path.join(OUT, "external_by_sequences.csv"), index=False)
    full = ext[ext.n_modalities == 4]
    res["external"] = {"n_patients": int(full["case_id"].nunique()),
                       "dice_full": {k: boot(full, lambda d, k=k: d[k].mean()) for k in ("dice_WT", "dice_TC", "dice_ET")},
                       "by_sequences": by.to_dict("records")}
    tst = df[(df.cohort == "test") & (df.method == "mc8") & (df.n_modalities == 4)]
    res["internal_rerun_full"] = {k: float(tst[k].mean()) for k in ("dice_WT", "dice_TC", "dice_ET")}

# ---- calibration ------------------------------------------------------------------------------
cal = []
for coh in ("test", "external"):
    p = os.path.join(SRC, "evidence", coh, "calibration.json")
    if os.path.exists(p):
        cal += [c for c in json.load(open(p)) if c["cohort"] == coh]
cal_df = pd.DataFrame([{k: v for k, v in c.items() if not k.startswith("bin_")} for c in cal])
cal_df.to_csv(os.path.join(OUT, "calibration.csv"), index=False)
res["calibration"] = cal_df.to_dict("records")

# ---- architecture baselines --------------------------------------------------------------------
arch = []
base_sum = json.load(open("hosting/data/runs/base/results/summary.json"))
base_sub = pd.read_csv("hosting/data/runs/base/results/subsets.csv")
for name, summ, sub in [("NeuroScope U-Net (ours)", base_sum, base_sub)] + [
        (lab, json.load(open(os.path.join(SRC, a, "results", "summary.json"))),
         pd.read_csv(os.path.join(SRC, a, "results", "subsets.csv")) if os.path.exists(os.path.join(SRC, a, "results", "subsets.csv")) else None)
        for a, lab in [("attention_unet", "Attention U-Net (Oktay 2018)"), ("segresnet", "SegResNet (Myronenko 2018)")]
        if os.path.exists(os.path.join(SRC, a, "results", "summary.json"))]:
    r = {"model": name}
    for k in ("dice_WT", "dice_TC", "dice_ET", "hd95_WT", "hd95_TC", "hd95_ET"):
        if k in summ:
            r[k] = summ[k]["mean"]; r[k + "_ci"] = summ[k]["ci95"]
    r["failure_auroc_full"] = summ.get("reliability", {}).get("failure_auroc")
    if sub is not None:
        for n in (1, 2, 3):
            r[f"dice_WT_{n}seq"] = sub[sub.n_modalities == n]["dice_WT"].mean()
    arch.append(r)
arch_df = pd.DataFrame(arch)
arch_df.to_csv(os.path.join(OUT, "architectures.csv"), index=False)
res["architectures"] = arch_df.to_dict("records")
for a in ("attention_unet", "segresnet"):
    h = os.path.join(SRC, a, "history.csv")
    if os.path.exists(h):
        res.setdefault("baseline_training", {})[a] = pd.read_csv(h).iloc[-1].to_dict()

json.dump(res, open(os.path.join(OUT, "evidence.json"), "w"), indent=2, default=lambda x: float(x) if np.isscalar(x) else str(x))

# ---- figures ------------------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
COL = {"mc1": "#8C8C8C", "mc1_ts": "#4C72B0", "tta8": "#55A868", "mc8": "#C9A227"}
for coh in ("test", "external"):
    sel = [c for c in cal if c["cohort"] == coh and c["subsets"] == "full"]
    if not sel:
        continue
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, reg in zip(axs, ["WT", "TC", "ET"]):
        ax.plot([0, 1], [0, 1], ls=":", c="k", lw=1)
        for c in sel:
            if c["region"] != reg:
                continue
            n = np.array(c["bin_count"]); ok = n > 200
            ax.plot(np.array(c["bin_conf"])[ok], np.array(c["bin_freq"])[ok], "o-", ms=3, c=COL[c["method"]],
                    label=f"{METHODS[c['method']]}  ECE {c['ece']:.4f}")
        ax.set_title(f"{reg}"); ax.set_xlabel("predicted probability"); ax.set_ylabel("observed frequency")
        ax.legend(fontsize=7, frameon=False, loc="upper left")
    fig.suptitle(f"Voxel calibration, {'internal test (BraTS 2020)' if coh == 'test' else 'external test (BraTS-Africa)'}, all four sequences")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, f"fig_calibration_{coh}.png"), dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(5.5, 4))
g = df[(df.cohort == "test")]
for m in ("mc1", "tta8", "mc8"):
    gg = g[g.method == m]
    o = np.argsort(gg["uncertainty"].values); risk = 1 - gg["dice_WT"].values[o]
    ax.plot(np.arange(1, len(o) + 1) / len(o), np.cumsum(risk) / np.arange(1, len(o) + 1), c=COL[m], label=f"{METHODS[m]} (AURC {aurc(gg['uncertainty'].values, gg['dice_WT'].values):.3f})")
ax.axhline(1 - g[g.method == "mc8"]["dice_WT"].mean(), ls=":", c="k", lw=1, label="no review (random order)")
ax.set_xlabel("fraction of cases accepted without review"); ax.set_ylabel("mean (1 - Dice) of accepted cases")
ax.set_title("Risk-coverage, internal test, all 15 sequence sets"); ax.legend(fontsize=7, frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_risk_coverage_methods.png"), dpi=150); plt.close(fig)
print(json.dumps({k: v for k, v in res.items() if k not in ("calibration",)}, indent=1, default=str)[:6000])
