"""Reliability analysis: does the model's uncertainty predict its own failures?

Given per-case (uncertainty u_i, Dice d_i):

  * failure_i = d_i < tau_fail   (tau_fail fixed a priori, e.g. 0.75 WT Dice)
  * Failure-detection AUROC: how well u ranks failures above successes.
  * Spearman rho(u, d): monotonic association (expected negative).
  * Risk–coverage curve: sort cases by u ascending; for coverage c keep the
    c-fraction most confident cases; risk(c) = mean (1 - Dice) among kept.
    AURC = area under that curve (lower is better; Geifman & El-Yaniv, 2017).
  * Review threshold: chosen ON THE VALIDATION SET by maximising Youden's J
    (sensitivity + specificity - 1) for failure detection. It is then frozen
    and applied to the test set, so the reported flag performance is honest.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
from scipy import stats


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUROC (Mann–Whitney). labels: 1 = positive (failure)."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    n_pos, n_neg = labels.sum(), (~labels).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = stats.rankdata(scores)
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def risk_coverage(uncertainty: np.ndarray, dice: np.ndarray):
    order = np.argsort(uncertainty)
    risk = 1.0 - np.asarray(dice, float)[order]
    cum_risk = np.cumsum(risk) / np.arange(1, len(risk) + 1)
    coverage = np.arange(1, len(risk) + 1) / len(risk)
    aurc = float(np.trapezoid(cum_risk, coverage)) if hasattr(np, "trapezoid") else float(np.trapz(cum_risk, coverage))
    return coverage, cum_risk, aurc


def choose_threshold(uncertainty: np.ndarray, dice: np.ndarray, tau_fail: float) -> Dict[str, float]:
    u, d = np.asarray(uncertainty, float), np.asarray(dice, float)
    fail = d < tau_fail
    if fail.sum() == 0 or (~fail).sum() == 0:
        # degenerate validation set: fall back to flagging the top 20% most uncertain
        thr = float(np.percentile(u, 80))
        return {"threshold": thr, "youden_j": float("nan"), "n_fail": int(fail.sum()), "n_total": int(len(u)), "rule": "p80_fallback"}
    cands = np.unique(u)
    best_j, best_t = -1.0, float(cands[-1])
    for t in cands:
        flagged = u >= t
        sens = (flagged & fail).sum() / fail.sum()
        spec = (~flagged & ~fail).sum() / (~fail).sum()
        j = sens + spec - 1
        if j > best_j:
            best_j, best_t = float(j), float(t)
    return {"threshold": best_t, "youden_j": best_j, "n_fail": int(fail.sum()), "n_total": int(len(u)), "rule": "youden"}


def flag_performance(uncertainty: np.ndarray, dice: np.ndarray, threshold: float, tau_fail: float) -> Dict[str, float]:
    u, d = np.asarray(uncertainty, float), np.asarray(dice, float)
    fail = d < tau_fail
    flagged = u >= threshold
    tp = (flagged & fail).sum(); fp = (flagged & ~fail).sum()
    fn = (~flagged & fail).sum(); tn = (~flagged & ~fail).sum()
    accepted = ~flagged
    return {
        "flag_rate": float(flagged.mean()),
        "flag_sensitivity": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        "flag_specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "flag_precision": float(tp / (tp + fp)) if (tp + fp) else float("nan"),
        "dice_all": float(d.mean()),
        "dice_accepted": float(d[accepted].mean()) if accepted.sum() else float("nan"),
        "dice_flagged": float(d[flagged].mean()) if flagged.sum() else float("nan"),
        "n_flagged": int(flagged.sum()),
    }


def summarize(uncertainty: np.ndarray, dice: np.ndarray, tau_fail: float) -> Dict[str, float]:
    u, d = np.asarray(uncertainty, float), np.asarray(dice, float)
    fail = d < tau_fail
    rho, p = stats.spearmanr(u, d) if len(u) > 2 else (float("nan"), float("nan"))
    _, _, aurc = risk_coverage(u, d)
    return {
        "failure_auroc": auroc(u, fail),
        "spearman_rho": float(rho),
        "spearman_p": float(p),
        "aurc": aurc,
        "failure_rate": float(fail.mean()),
    }
