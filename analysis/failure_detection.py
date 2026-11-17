#!/usr/bin/env python
"""Failure-detection analysis with patient-level bootstrap confidence intervals.

Inputs (from scripts/evaluate.py):  results/test_per_case.csv (MC=8, full sequences, 74 patients)
                                     results/subsets_per_case.csv (MC=4, 15 sequence sets x 74 patients)
                                     results/threshold.json (review threshold chosen on validation)
Why: with all four sequences only 3 of 74 test patients fail (WT Dice < 0.75), so an AUROC on
that set alone is fragile. Removing sequences creates realistic failures on the SAME held-out
patients (1,110 case-conditions). Rows from one patient are not independent, so every
confidence interval resamples PATIENTS, not rows.
"""
import json, sys, os
import numpy as np, pandas as pd
from scipy import stats

R = sys.argv[1] if len(sys.argv) > 1 else "hosting/data/runs/base/results"
OUT = sys.argv[2] if len(sys.argv) > 2 else "docs/evidence"
os.makedirs(OUT, exist_ok=True)
B = 2000
rng = np.random.default_rng(0)


def auroc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, bool)
    if y.all() or (~y).all():
        return np.nan
    r = stats.rankdata(s)
    return (r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())


def aurc(u, d):
    o = np.argsort(u); risk = 1 - np.asarray(d)[o]
    return float(np.mean(np.cumsum(risk) / np.arange(1, len(risk) + 1)))


def boot(df, fn):
    """Patient-clustered bootstrap: resample patients with replacement, keep all their rows."""
    pats = df["case_id"].unique()
    groups = {p: g for p, g in df.groupby("case_id")}
    vals = []
    for _ in range(B):
        pick = rng.choice(pats, size=len(pats), replace=True)
        v = fn(pd.concat([groups[p] for p in pick], ignore_index=True))
        if np.isfinite(v):
            vals.append(v)
    return float(fn(df)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), len(vals)


def stratified_auroc(df, score="uncertainty", tau=0.75):
    """Mean AUROC within each sequence set (only sets that contain both outcomes), weighted by size."""
    num = den = 0.0
    for _, g in df.groupby("subset"):
        y = g["dice_WT"] < tau
        a = auroc(g[score], y)
        if np.isfinite(a):
            num += a * len(g); den += len(g)
    return num / den if den else np.nan


full = pd.read_csv(f"{R}/test_per_case.csv")
sub = pd.read_csv(f"{R}/subsets_per_case.csv")
thr = json.load(open(f"{R}/threshold.json"))["threshold"]
sub["n_missing"] = 4 - sub["n_modalities"]
res = {"threshold_from_validation": thr, "bootstrap": f"{B} patient-level resamples, 95% percentile intervals"}

# 1) full sequences, MC=8 (the headline setting)
y = full["dice_WT"] < 0.75
res["full_sequences"] = {
    "n_patients": int(len(full)), "n_failures": int(y.sum()),
    "auroc": boot(full, lambda d: auroc(d["uncertainty"], d["dice_WT"] < 0.75)),
    "spearman_rho_uncertainty_vs_dice": boot(full, lambda d: stats.spearmanr(d["uncertainty"], d["dice_WT"])[0]),
    "flag_caught": f"{int(((full['uncertainty'] >= thr) & y).sum())} of {int(y.sum())}",
    "flag_rate": float((full["uncertainty"] >= thr).mean()),
}

# 2) all 15 sequence sets pooled, MC=4
f = sub["dice_WT"] < 0.75
fl = sub["uncertainty"] >= thr
res["all_sequence_sets"] = {
    "n_rows": int(len(sub)), "n_patients": int(sub["case_id"].nunique()), "n_failures": int(f.sum()),
    "failure_rate": float(f.mean()),
    "auroc_uncertainty": boot(sub, lambda d: auroc(d["uncertainty"], d["dice_WT"] < 0.75)),
    "auroc_baseline_number_of_missing_sequences": boot(sub, lambda d: auroc(d["n_missing"] + 1e-6 * rng.random(len(d)), d["dice_WT"] < 0.75)),
    "auroc_baseline_small_predicted_volume": boot(sub, lambda d: auroc(-d["WT_ml"], d["dice_WT"] < 0.75)),
    "auroc_within_each_sequence_set": boot(sub, stratified_auroc),
    "spearman_rho_uncertainty_vs_dice": boot(sub, lambda d: stats.spearmanr(d["uncertainty"], d["dice_WT"])[0]),
    "aurc_uncertainty": aurc(sub["uncertainty"], sub["dice_WT"]),
    "aurc_random": float(np.mean(1 - sub["dice_WT"])),
    "aurc_oracle": aurc(-sub["dice_WT"], sub["dice_WT"]),
    "frozen_threshold": {
        "flag_rate": float(fl.mean()),
        "sensitivity": boot(sub, lambda d: ((d["uncertainty"] >= thr) & (d["dice_WT"] < 0.75)).sum() / max((d["dice_WT"] < 0.75).sum(), 1)),
        "specificity": boot(sub, lambda d: ((d["uncertainty"] < thr) & (d["dice_WT"] >= 0.75)).sum() / max((d["dice_WT"] >= 0.75).sum(), 1)),
        "dice_accepted": float(sub.loc[~fl, "dice_WT"].mean()), "dice_flagged": float(sub.loc[fl, "dice_WT"].mean()),
        "failures_caught": f"{int((fl & f).sum())} of {int(f.sum())}",
        "failures_missed": int((~fl & f).sum()),
    },
}

# 3) by number of sequences available
rows = []
for k, g in sub.groupby("n_modalities"):
    yy = g["dice_WT"] < 0.75; ff = g["uncertainty"] >= thr
    rows.append({"sequences_available": int(k), "rows": len(g), "failures": int(yy.sum()), "dice_WT": g["dice_WT"].mean(),
                 "auroc": auroc(g["uncertainty"], yy), "flag_rate": ff.mean(),
                 "caught": int((ff & yy).sum()), "dice_accepted": g.loc[~ff, "dice_WT"].mean()})
by_k = pd.DataFrame(rows)
by_k.to_csv(f"{OUT}/failure_detection_by_sequences.csv", index=False)

# 4) per sequence set
rows = []
for s_, g in sub.groupby("subset"):
    yy = g["dice_WT"] < 0.75; ff = g["uncertainty"] >= thr
    rows.append({"subset": s_, "n_modalities": int(g["n_modalities"].iloc[0]), "dice_WT": g["dice_WT"].mean(),
                 "failures": int(yy.sum()), "auroc": auroc(g["uncertainty"], yy), "flag_rate": ff.mean(),
                 "caught": int((ff & yy).sum()), "false_alarms": int((ff & ~yy).sum())})
pd.DataFrame(rows).sort_values(["n_modalities", "dice_WT"], ascending=[False, False]).to_csv(f"{OUT}/failure_detection_by_set.csv", index=False)

# 5) sensitivity to the failure definition
res["failure_definition_sensitivity"] = {
    str(t): {"failures": int((sub["dice_WT"] < t).sum()), "auroc": auroc(sub["uncertainty"], sub["dice_WT"] < t)}
    for t in (0.6, 0.7, 0.75, 0.8, 0.85)}

json.dump(res, open(f"{OUT}/failure_detection.json", "w"), indent=2, default=float)
print(json.dumps(res, indent=1, default=lambda x: round(float(x), 4)))
print(by_k.round(3).to_string(index=False))
