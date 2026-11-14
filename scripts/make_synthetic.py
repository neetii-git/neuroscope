#!/usr/bin/env python
"""Generate synthetic BraTS-shaped NIfTI cases for pipeline tests (NOT real MRI).

Each case: ellipsoidal "brain" with per-modality intensity, a spherical tumour
with nested core/enhancing regions, Gaussian noise. Layout mirrors BraTS 2020.
"""
from __future__ import annotations

import argparse
import os

import nibabel as nib
import numpy as np


def make_case(rng: np.random.RandomState, shape=(240, 240, 155), small=False):
    if small:
        shape = (96, 96, 40)
    H, W, D = shape
    hh, ww, dd = np.mgrid[0:H, 0:W, 0:D]
    cy, cx, cz = H / 2, W / 2, D / 2
    brain = ((hh - cy) / (H * 0.38)) ** 2 + ((ww - cx) / (W * 0.33)) ** 2 + ((dd - cz) / (D * 0.42)) ** 2 < 1
    r = min(H, W, D) * rng.uniform(0.08, 0.15)
    tc_ = (
        rng.uniform(cy - H * 0.15, cy + H * 0.15),
        rng.uniform(cx - W * 0.15, cx + W * 0.15),
        rng.uniform(cz - D * 0.15, cz + D * 0.15),
    )
    dist = np.sqrt((hh - tc_[0]) ** 2 + (ww - tc_[1]) ** 2 + (dd - tc_[2]) ** 2)
    wt = (dist < r) & brain
    tc = (dist < r * 0.65) & brain
    et = (dist < r * 0.45) & brain
    seg = np.zeros(shape, dtype=np.int16)
    seg[wt] = 2
    seg[tc] = 1
    seg[et] = 4
    vols = {}
    base = {"t1": 400, "t1ce": 450, "t2": 600, "flair": 500}
    tumour_gain = {"t1": (0.9, 0.8, 1.1), "t1ce": (1.0, 0.8, 1.8), "t2": (1.5, 1.2, 1.1), "flair": (1.7, 1.1, 1.0)}
    for m, b in base.items():
        v = np.zeros(shape, dtype=np.float32)
        v[brain] = b
        g_wt, g_tc, g_et = tumour_gain[m]
        v[wt] *= g_wt
        v[tc] *= g_tc / g_wt
        v[et] *= g_et / g_tc
        v[brain] += rng.normal(0, b * 0.05, size=int(brain.sum()))
        vols[m] = np.clip(v, 0, None)
    vols["seg"] = seg
    return vols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--small", action="store_true", help="tiny volumes for fast CPU tests")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.RandomState(args.seed)
    root = os.path.join(args.out, "MICCAI_BraTS2020_TrainingData")
    for i in range(1, args.n + 1):
        cid = f"BraTS20_Training_{i:03d}"
        d = os.path.join(root, cid)
        os.makedirs(d, exist_ok=True)
        vols = make_case(rng, small=args.small)
        for m, v in vols.items():
            nib.save(nib.Nifti1Image(v, np.eye(4)), os.path.join(d, f"{cid}_{m}.nii"))
    print(f"wrote {args.n} synthetic cases to {root}")


if __name__ == "__main__":
    main()
