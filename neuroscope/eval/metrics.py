"""Segmentation metrics on binary 3D region masks.

Dice  = 2|A∩B| / (|A|+|B|)          overlap, 1 = perfect
IoU   = |A∩B| / |A∪B|
HD95  = 95th percentile of symmetric surface distances (robust Hausdorff), in
        voxels (BraTS is 1 mm isotropic so voxels == mm). Captures boundary
        error that Dice ignores.
Sensitivity = TP/(TP+FN), Specificity = TN/(TN+FP), Precision = TP/(TP+FP)

Edge cases follow BraTS convention: if both masks are empty the case is a
perfect match (Dice 1, HD95 0); if exactly one is empty, Dice 0 and HD95 is
undefined (returned as NaN and excluded from means).
"""
from __future__ import annotations

from typing import Dict

import numpy as np
from scipy import ndimage


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    ps, gs = pred.sum(), gt.sum()
    if ps == 0 and gs == 0:
        return 1.0
    return float(2.0 * np.logical_and(pred, gt).sum() / (ps + gs))


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(pred, gt).sum() / union)


def _surface(mask: np.ndarray) -> np.ndarray:
    eroded = ndimage.binary_erosion(mask, iterations=1, border_value=0)
    return mask & ~eroded


def hd95(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return float("nan")
    sp, sg = _surface(pred), _surface(gt)
    dt_g = ndimage.distance_transform_edt(~sg)
    dt_p = ndimage.distance_transform_edt(~sp)
    d_pg = dt_g[sp]  # distance from each pred-surface voxel to gt surface
    d_gp = dt_p[sg]
    return float(max(np.percentile(d_pg, 95), np.percentile(d_gp, 95)))


def confusion_rates(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    pred, gt = pred.astype(bool), gt.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    tn = np.logical_and(~pred, ~gt).sum()
    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    return {"sensitivity": float(sens), "specificity": float(spec), "precision": float(prec)}


def region_metrics(pred_regions: np.ndarray, gt_regions: np.ndarray, with_hd95: bool = True) -> Dict[str, float]:
    """pred_regions, gt_regions: [3, D, H, W] binary (WT, TC, ET)."""
    out: Dict[str, float] = {}
    for i, name in enumerate(["WT", "TC", "ET"]):
        p, g = pred_regions[i], gt_regions[i]
        out[f"dice_{name}"] = dice(p, g)
        out[f"iou_{name}"] = iou(p, g)
        if with_hd95:
            out[f"hd95_{name}"] = hd95(p, g)
        for k, v in confusion_rates(p, g).items():
            out[f"{k}_{name}"] = v
    out["dice_mean"] = float(np.mean([out["dice_WT"], out["dice_TC"], out["dice_ET"]]))
    return out


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap CI of the mean, ignoring NaNs."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    means = [rng.choice(v, v.size, replace=True).mean() for _ in range(n_boot)]
    return float(v.mean()), float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))
