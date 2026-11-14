#!/usr/bin/env python
"""Full evaluation: test metrics, 15 modality subsets, uncertainty reliability,
review-flag threshold (chosen on val, frozen for test), corruption robustness.

Usage:
    python scripts/evaluate.py --cache /kaggle/working/cache --ckpt runs/base/best.pt \
        --out runs/base/results [--mc 8] [--mc-subsets 4] [--max-test-cases 40] [--skip-subsets] [--skip-corruptions]

Outputs (all in --out):
    summary.json            headline numbers (+ bootstrap CIs)
    threshold.json          review-flag threshold and how it was chosen
    test_per_case.csv       per-case metrics + uncertainty (full modalities)
    subsets.csv             per-subset mean Dice / uncertainty / flag rate
    subsets_per_case.csv    per-subset per-case rows
    corruptions.csv         Dice under noise / contrast / resolution corruption
    fig_*.png               risk-coverage, uncertainty-vs-dice, subset bars
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import yaml
from scipy import ndimage
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from neuroscope.data.brats import MODALITIES, labels_to_regions  # noqa: E402
from neuroscope.data.dataset import ALL_SUBSETS, CachedCase, load_index, subset_name  # noqa: E402
from neuroscope.data.split import load_split  # noqa: E402
from neuroscope.eval import reliability  # noqa: E402
from neuroscope.eval.metrics import bootstrap_ci, region_metrics  # noqa: E402
from neuroscope.inference.predictor import load_checkpoint, predict_volume  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("evaluate")


def corrupt(img: np.ndarray, kind: str, level: float, rng: np.random.RandomState) -> np.ndarray:
    out = img.copy()
    nz = img != 0
    if kind == "noise":  # additive Gaussian noise in z-score units
        out[nz] += rng.normal(0, level, size=int(nz.sum())).astype(np.float32)
    elif kind == "contrast":  # gamma-like contrast change on the positive range
        out = np.where(nz, np.sign(out) * np.abs(out) ** level, 0.0).astype(np.float32)
    elif kind == "resolution":  # downsample in-plane by `level` then upsample (blur)
        f = 1.0 / level
        low = ndimage.zoom(out, (1, 1, f, f), order=1)
        out = ndimage.zoom(low, (1, 1, level, level), order=1)[:, :, : img.shape[2], : img.shape[3]].astype(np.float32)
        out[~nz] = 0.0
    return out


def run_cases(model, cache, case_ids, device, mc, subset=None, corruption=None, desc="eval", with_hd95=True):
    index = load_index(cache)
    rows = []
    rng = np.random.RandomState(0)
    for cid in tqdm(case_ids, desc=desc, leave=False):
        c = CachedCase(cache, cid, index[cid])
        img, seg = c.volume()
        if corruption is not None:
            img = corrupt(img, corruption[0], corruption[1], rng)
        t0 = time.time()
        pred = predict_volume(model, img, mc_samples=mc, device=device, subset=subset, batch_size=32)
        dt = time.time() - t0
        m = region_metrics(pred.regions, labels_to_regions(seg), with_hd95=with_hd95)
        m.update({"case_id": cid, "uncertainty": pred.case_uncertainty, "mc_samples": pred.mc_samples,
                  "infer_s": dt, "gt_WT_ml": float((seg > 0).sum() / 1000.0), **pred.volumes_ml})
        rows.append(m)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml"))
    ap.add_argument("--mc", type=int, default=None)
    ap.add_argument("--mc-subsets", type=int, default=None)
    ap.add_argument("--max-test-cases", type=int, default=None)
    ap.add_argument("--max-val-cases", type=int, default=None)
    ap.add_argument("--skip-subsets", action="store_true")
    ap.add_argument("--skip-corruptions", action="store_true")
    ap.add_argument("--no-hd95", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    mc = args.mc or int(cfg["eval"]["mc_samples"])
    mc_sub = args.mc_subsets or int(cfg["eval"]["mc_samples_subsets"])
    tau_fail = float(cfg["eval"]["failure_dice_threshold"])
    os.makedirs(args.out, exist_ok=True)

    model = load_checkpoint(args.ckpt, args.device)
    split = load_split(os.path.join(args.cache, "split.json"))
    val_ids = split["val"][: args.max_val_cases] if args.max_val_cases else split["val"]
    test_ids = split["test"][: args.max_test_cases] if args.max_test_cases else split["test"]
    log.info("val %d cases, test %d cases, MC=%d, device=%s", len(val_ids), len(test_ids), mc, args.device)

    # 1) Validation: choose the review threshold -------------------------------------------------
    val_df = run_cases(model, args.cache, val_ids, args.device, mc, desc="val", with_hd95=False)
    val_df.to_csv(os.path.join(args.out, "val_per_case.csv"), index=False)
    thr = reliability.choose_threshold(val_df["uncertainty"].values, val_df["dice_WT"].values, tau_fail)
    thr.update({"tau_fail": tau_fail, "chosen_on": "val", "n_val": len(val_df)})
    with open(os.path.join(args.out, "threshold.json"), "w") as f:
        json.dump(thr, f, indent=2)
    log.info("review threshold u >= %.4f (rule=%s, J=%.3f)", thr["threshold"], thr["rule"], thr["youden_j"])

    # 2) Test, full modalities ------------------------------------------------------------------
    test_df = run_cases(model, args.cache, test_ids, args.device, mc, desc="test", with_hd95=not args.no_hd95)
    test_df.to_csv(os.path.join(args.out, "test_per_case.csv"), index=False)
    summary = {"n_test": len(test_df), "n_val": len(val_df), "mc_samples": mc, "tau_fail": tau_fail,
               "checkpoint": os.path.abspath(args.ckpt), "device": args.device}
    for k in ["dice_WT", "dice_TC", "dice_ET", "dice_mean", "hd95_WT", "hd95_TC", "hd95_ET",
              "sensitivity_WT", "specificity_WT", "precision_WT", "infer_s"]:
        if k in test_df:
            mean, lo, hi = bootstrap_ci(test_df[k].values)
            summary[k] = {"mean": mean, "ci95": [lo, hi]}
    summary["reliability"] = reliability.summarize(test_df["uncertainty"].values, test_df["dice_WT"].values, tau_fail)
    summary["review_flag"] = reliability.flag_performance(test_df["uncertainty"].values, test_df["dice_WT"].values, thr["threshold"], tau_fail)
    log.info("TEST Dice WT %.3f TC %.3f ET %.3f | failure AUROC %.3f | flag rate %.2f",
             summary["dice_WT"]["mean"], summary["dice_TC"]["mean"], summary["dice_ET"]["mean"],
             summary["reliability"]["failure_auroc"], summary["review_flag"]["flag_rate"])

    # 3) Missing-modality sweep over all 15 subsets ---------------------------------------------
    if not args.skip_subsets:
        sub_rows, per_case = [], []
        for subset in ALL_SUBSETS:
            name = subset_name(subset)
            df = run_cases(model, args.cache, test_ids, args.device, mc_sub, subset=subset, desc=name, with_hd95=False)
            df["subset"] = name
            df["n_modalities"] = len(subset)
            per_case.append(df)
            flagged = (df["uncertainty"].values >= thr["threshold"]).mean()
            rel = reliability.summarize(df["uncertainty"].values, df["dice_WT"].values, tau_fail)
            sub_rows.append({"subset": name, "n_modalities": len(subset),
                             **{m: int(i in subset) for i, m in enumerate(MODALITIES)},
                             "dice_WT": df["dice_WT"].mean(), "dice_TC": df["dice_TC"].mean(), "dice_ET": df["dice_ET"].mean(),
                             "dice_mean": df["dice_mean"].mean(), "uncertainty": df["uncertainty"].mean(),
                             "flag_rate": float(flagged), "failure_rate": rel["failure_rate"],
                             "failure_auroc": rel["failure_auroc"]})
            log.info("subset %-22s Dice WT %.3f TC %.3f ET %.3f | u %.4f | flagged %.2f",
                     name, sub_rows[-1]["dice_WT"], sub_rows[-1]["dice_TC"], sub_rows[-1]["dice_ET"], sub_rows[-1]["uncertainty"], flagged)
        sub_df = pd.DataFrame(sub_rows).sort_values(["n_modalities", "dice_mean"], ascending=[False, False])
        sub_df.to_csv(os.path.join(args.out, "subsets.csv"), index=False)
        pd.concat(per_case).to_csv(os.path.join(args.out, "subsets_per_case.csv"), index=False)
        # uncertainty should rise as modalities are removed: correlation across the 15 subsets
        from scipy import stats
        rho, p = stats.spearmanr(sub_df["uncertainty"], sub_df["dice_mean"])
        summary["subsets"] = {"spearman_uncertainty_vs_dice_across_subsets": float(rho), "p": float(p),
                              "best": sub_df.iloc[0]["subset"], "worst": sub_df.iloc[-1]["subset"],
                              "single_modality_dice_WT": {r["subset"]: float(r["dice_WT"]) for _, r in sub_df[sub_df.n_modalities == 1].iterrows()}}

    # 4) Corruption robustness (full modalities, deterministic) ---------------------------------
    if not args.skip_corruptions:
        cor_rows = []
        for kind, levels in [("noise", [0.1, 0.3, 0.6]), ("contrast", [0.7, 1.4]), ("resolution", [2.0, 4.0])]:
            for lvl in levels:
                df = run_cases(model, args.cache, test_ids, args.device, mc_sub, corruption=(kind, lvl), desc=f"{kind}={lvl}", with_hd95=False)
                cor_rows.append({"corruption": kind, "level": lvl, "dice_WT": df["dice_WT"].mean(), "dice_TC": df["dice_TC"].mean(),
                                 "dice_ET": df["dice_ET"].mean(), "uncertainty": df["uncertainty"].mean(),
                                 "flag_rate": float((df["uncertainty"].values >= thr["threshold"]).mean())})
                log.info("corruption %s=%.2f Dice WT %.3f | u %.4f | flagged %.2f", kind, lvl, cor_rows[-1]["dice_WT"], cor_rows[-1]["uncertainty"], cor_rows[-1]["flag_rate"])
        cor_df = pd.DataFrame(cor_rows)
        cor_df.loc[len(cor_df)] = {"corruption": "none", "level": 0, "dice_WT": test_df["dice_WT"].mean(), "dice_TC": test_df["dice_TC"].mean(),
                                   "dice_ET": test_df["dice_ET"].mean(), "uncertainty": test_df["uncertainty"].mean(),
                                   "flag_rate": summary["review_flag"]["flag_rate"]}
        cor_df.to_csv(os.path.join(args.out, "corruptions.csv"), index=False)

    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)

    # 5) Figures --------------------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cov, risk, aurc = reliability.risk_coverage(test_df["uncertainty"].values, test_df["dice_WT"].values)
        plt.figure(figsize=(5, 4)); plt.plot(cov, risk); plt.xlabel("coverage (fraction of cases accepted)")
        plt.ylabel("risk = 1 - WT Dice (accepted cases)"); plt.title(f"Risk–coverage (AURC={aurc:.3f})"); plt.grid(alpha=.3)
        plt.tight_layout(); plt.savefig(os.path.join(args.out, "fig_risk_coverage.png"), dpi=150); plt.close()

        plt.figure(figsize=(5, 4)); plt.scatter(test_df["uncertainty"], test_df["dice_WT"], s=18)
        plt.axvline(thr["threshold"], color="r", ls="--", label="review threshold (val)"); plt.axhline(tau_fail, color="gray", ls=":", label="failure Dice")
        plt.xlabel("case uncertainty (mean entropy in ROI)"); plt.ylabel("WT Dice"); plt.legend(); plt.grid(alpha=.3)
        plt.title(f"Uncertainty vs Dice (AUROC={summary['reliability']['failure_auroc']:.2f})")
        plt.tight_layout(); plt.savefig(os.path.join(args.out, "fig_uncertainty_vs_dice.png"), dpi=150); plt.close()

        if not args.skip_subsets:
            s = sub_df.sort_values("dice_WT")
            fig, ax1 = plt.subplots(figsize=(9, 4.5))
            ax1.barh(s["subset"], s["dice_WT"], color="#4C72B0"); ax1.set_xlabel("WT Dice"); ax1.set_xlim(0, 1)
            ax2 = ax1.twiny(); ax2.plot(s["uncertainty"], s["subset"], "o-", color="#DD8452"); ax2.set_xlabel("case uncertainty", color="#DD8452")
            plt.title("Segmentation quality and uncertainty across the 15 modality subsets")
            plt.tight_layout(); plt.savefig(os.path.join(args.out, "fig_subsets.png"), dpi=150); plt.close()
    except Exception as e:  # noqa: BLE001
        log.warning("figure generation failed: %s", e)
    log.info("Results written to %s", args.out)


if __name__ == "__main__":
    main()
