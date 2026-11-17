#!/usr/bin/env python
"""Evidence experiments for the technical report.

For every case and every one of the 15 MRI-sequence combinations this script runs
three uncertainty methods on the SAME trained model and records, per method:
segmentation quality (Dice WT/TC/ET), the case-level uncertainty score and the
predicted volume. It also accumulates voxel-level calibration statistics.

Methods
    mc1  : one deterministic pass; uncertainty = entropy of the sigmoid (softmax-style baseline)
    tta8 : 8 deterministic passes over test-time augmentations (4 flips x 2 intensity scales)
    mc8  : 8 Monte Carlo dropout passes (NeuroScope's method)

Cohorts
    val      : internal validation patients (full sequences) - used ONLY to choose the
               review threshold of each method and to fit temperature scaling
    test     : internal held-out test patients (BraTS 2020)
    external : BraTS-Africa 2023 (BraTS-SSA) training cases - different continent,
               scanners (mostly 1.5 T) and patients; never seen in training

Outputs (--out):
    per_case.csv            one row per cohort x case x subset x method
    calibration.json        reliability-diagram bins + ECE / Brier per cohort, subset group, method, region
    temperature.json        temperature scaling fitted on validation (mc1 logits)
    env.json                GPU / timing environment
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from scipy import ndimage
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from neuroscope.data.brats import MODALITIES, labels_to_regions, load_nifti  # noqa: E402
from neuroscope.data.dataset import ALL_SUBSETS, CachedCase, apply_subset, load_index, subset_name  # noqa: E402
from neuroscope.data.preprocess import preprocess_case  # noqa: E402
from neuroscope.data.split import load_split  # noqa: E402
from neuroscope.eval.metrics import region_metrics  # noqa: E402
from neuroscope.inference.predictor import load_checkpoint  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("evidence")

NBINS = 15
REGIONS = ["WT", "TC", "ET"]
SSA_MAP = {"t1": "t1n", "t1ce": "t1c", "t2": "t2w", "flair": "t2f"}
TTA = [(fh, fw, s) for fh in (False, True) for fw in (False, True) for s in (1.0, 1.1)]


# ----------------------------------------------------------------------------------------------
def discover_ssa(root: str):
    """BraTS 2023 naming: <id>-t1n/t1c/t2w/t2f/seg.nii[.gz]; labels 1 NETC, 2 SNFH, 3 ET."""
    out = []
    for seg in sorted(glob.glob(os.path.join(root, "**", "*-seg.nii*"), recursive=True)):
        d = os.path.dirname(seg)
        cid = os.path.basename(seg).split("-seg.nii")[0]
        paths = {}
        for m, tag in SSA_MAP.items():
            hits = glob.glob(os.path.join(d, f"{cid}-{tag}.nii*"))
            if hits:
                paths[m] = hits[0]
        if len(paths) == 4:
            out.append((cid, paths, seg))
    # de-duplicate (some Kaggle copies contain the same case twice)
    seen, uniq = set(), []
    for c in out:
        if c[0] not in seen:
            seen.add(c[0]); uniq.append(c)
    return uniq


def load_ssa_case(paths, seg_path):
    vols = {m: load_nifti(p) for m, p in paths.items()}
    raw = load_nifti(seg_path).astype(np.int16)
    seg = np.zeros_like(raw, dtype=np.uint8)
    seg[raw == 1] = 1; seg[raw == 2] = 2; seg[raw == 3] = 3; seg[raw == 4] = 3
    vols["seg"] = seg
    img, segc, _ = preprocess_case(vols, available=MODALITIES)
    return img, segc


# ----------------------------------------------------------------------------------------------
@torch.no_grad()
def _pass(model, x, device, bs=32, fh=False, fw=False, scale=1.0):
    """x: [D,4,H,W] float tensor on CPU -> logits [3,D,H,W] numpy float32."""
    D = x.shape[0]
    out = np.zeros((D, 3, x.shape[2], x.shape[3]), dtype=np.float32)
    amp = device.startswith("cuda")
    for s in range(0, D, bs):
        xb = x[s:s + bs].to(device) * scale
        if fh: xb = torch.flip(xb, dims=[2])
        if fw: xb = torch.flip(xb, dims=[3])
        with torch.autocast(device_type="cuda", enabled=amp):
            lg = model(xb).float()
        if fh: lg = torch.flip(lg, dims=[2])
        if fw: lg = torch.flip(lg, dims=[3])
        out[s:s + bs] = lg.cpu().numpy()
    return np.transpose(out, (1, 0, 2, 3))


def _sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def _entropy(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def summarize_prob(prob, img):
    """Same post-processing and case-uncertainty definition as the app's predictor."""
    regions = (prob > 0.5).astype(np.uint8)
    regions[1] &= regions[0]; regions[2] &= regions[1]
    ent = _entropy(prob[0])
    roi = ndimage.binary_dilation(regions[0].astype(bool), iterations=2)
    roi |= (prob[0] > 0.1) & (prob[0] < 0.9)
    if roi.sum() == 0:
        roi = np.any(img != 0, axis=0)
    u = float(ent[roi].mean()) if roi.sum() else 0.0
    return regions, u


def run_methods(model, img, device, mc=8):
    x = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (1, 0, 2, 3))))
    model.eval()
    t0 = time.time(); det = _pass(model, x, device); t_mc1 = time.time() - t0
    probs = {"mc1": _sig(det)}
    t0 = time.time()
    # average probabilities (not logits) across augmentations
    acc = np.zeros_like(det)
    for fh, fw, s in TTA:
        lg = det if (not fh and not fw and s == 1.0) else _pass(model, x, device, fh=fh, fw=fw, scale=s)
        acc += _sig(lg)
    probs["tta8"] = acc / len(TTA); t_tta = time.time() - t0
    t0 = time.time()
    model.enable_mc_dropout()
    acc = np.zeros_like(det)
    for _ in range(mc):
        acc += _sig(_pass(model, x, device))
    model.eval()
    probs["mc8"] = acc / mc; t_mc = time.time() - t0
    return probs, det, {"mc1": t_mc1, "tta8": t_tta, "mc8": t_mc}


# ----------------------------------------------------------------------------------------------
class Calib:
    """Reliability-diagram accumulator over voxels inside the brain."""

    def __init__(self):
        self.h = {}

    def add(self, key, prob, gt_regions, brain):
        for r, name in enumerate(REGIONS):
            p = prob[r][brain].ravel(); y = gt_regions[r][brain].ravel().astype(np.float64)
            b = np.minimum((p * NBINS).astype(int), NBINS - 1)
            k = key + (name,)
            if k not in self.h:
                self.h[k] = [np.zeros(NBINS), np.zeros(NBINS), np.zeros(NBINS), 0.0]
            H = self.h[k]
            H[0] += np.bincount(b, minlength=NBINS)
            H[1] += np.bincount(b, weights=p, minlength=NBINS)
            H[2] += np.bincount(b, weights=y, minlength=NBINS)
            H[3] += float(((p - y) ** 2).sum())

    def report(self):
        out = []
        for k, (n, sp, sy, se) in self.h.items():
            N = n.sum()
            conf = np.divide(sp, n, out=np.zeros_like(sp), where=n > 0)
            freq = np.divide(sy, n, out=np.zeros_like(sy), where=n > 0)
            ece = float((n / N * np.abs(conf - freq)).sum())
            fg = n * (np.arange(NBINS) >= 1)  # bins above 1/15 ~ where the uncertainty lives
            ece_fg = float((fg / max(fg.sum(), 1) * np.abs(conf - freq)).sum())
            out.append({"cohort": k[0], "subsets": k[1], "method": k[2], "region": k[3], "n_voxels": int(N),
                        "ece": ece, "ece_above_0.067": ece_fg, "brier": se / N,
                        "bin_count": n.tolist(), "bin_conf": conf.tolist(), "bin_freq": freq.tolist()})
        return out


def fit_temperature(logits, labels):
    """One scalar temperature per region, minimising NLL on validation voxels."""
    temps = {}
    for r, name in enumerate(REGIONS):
        z = torch.tensor(np.concatenate(logits[r]), dtype=torch.float64)
        y = torch.tensor(np.concatenate(labels[r]), dtype=torch.float64)
        logT = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=200)

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(z / logT.exp(), y)
            loss.backward()
            return loss
        opt.step(closure)
        temps[name] = float(logT.exp().item())
    return temps


# ----------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ssa-root", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mc", type=int, default=8)
    ap.add_argument("--max-cases", type=int, default=None, help="smoke testing")
    ap.add_argument("--subsets", default="all", choices=["all", "full"])
    ap.add_argument("--cohorts", default="test,external", help="comma list from: test, external (validation always runs)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.RandomState(0)

    env = {"device": args.device, "torch": torch.__version__,
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "cpu_threads": torch.get_num_threads(), "mc": args.mc, "tta": len(TTA)}
    json.dump(env, open(os.path.join(args.out, "env.json"), "w"), indent=2)
    model = load_checkpoint(args.ckpt, args.device)
    split = load_split(os.path.join(args.cache, "split.json"))
    index = load_index(args.cache)
    lim = (lambda L: L[: args.max_cases]) if args.max_cases else (lambda L: L)
    subsets = ALL_SUBSETS if args.subsets == "all" else [tuple(range(4))]

    # --- 1) validation: temperature + per-method thresholds -----------------------------------
    rows = []
    t_logits = {r: [] for r in range(3)}; t_labels = {r: [] for r in range(3)}
    for cid in tqdm(lim(split["val"]), desc="val"):
        img, seg = CachedCase(args.cache, cid, index[cid]).volume()
        img = np.asarray(img, np.float32); gt = labels_to_regions(np.asarray(seg))
        probs, det, times = run_methods(model, img, args.device, args.mc)
        brain = np.any(img != 0, axis=0)
        for r in range(3):  # stratified voxel sample: all tumour voxels + up to 60k background
            pos = np.where(brain & (gt[r] > 0)); neg = np.where(brain & (gt[r] == 0))
            k = rng.choice(len(neg[0]), size=min(60000, len(neg[0])), replace=False)
            idx = tuple(np.concatenate([pos[i], neg[i][k]]) for i in range(3))
            t_logits[r].append(det[r][idx]); t_labels[r].append(gt[r][idx].astype(np.float64))
        for m, p in probs.items():
            reg, u = summarize_prob(p, img)
            met = region_metrics(reg, gt, with_hd95=False)
            rows.append({"cohort": "val", "case_id": cid, "subset": subset_name(tuple(range(4))), "n_modalities": 4,
                         "method": m, "uncertainty": u, "infer_s": times[m], "pred_WT_ml": float(reg[0].sum() / 1000),
                         "gt_WT_ml": float(gt[0].sum() / 1000), **{k: met[k] for k in ("dice_WT", "dice_TC", "dice_ET", "dice_mean")}})
    temps = fit_temperature(t_logits, t_labels)
    json.dump(temps, open(os.path.join(args.out, "temperature.json"), "w"), indent=2)
    log.info("temperature (fit on val, mc1 logits): %s", temps)
    tvec = np.array([temps[r] for r in REGIONS], np.float32)[:, None, None, None]

    # --- 2) internal test + external cohort over all subsets ------------------------------------
    calib = Calib()
    want = set(args.cohorts.split(","))
    cohorts = [("test", [(cid, None) for cid in lim(split["test"])])] if "test" in want else []
    if args.ssa_root and "external" in want:
        ssa = discover_ssa(args.ssa_root)
        log.info("external BraTS-SSA cases found: %d", len(ssa))
        cohorts.append(("external", [(c[0], c) for c in lim(ssa)]))
    for cohort, cases in cohorts:
        for cid, extra in tqdm(cases, desc=cohort):
            if extra is None:
                img, seg = CachedCase(args.cache, cid, index[cid]).volume()
                img = np.asarray(img, np.float32); seg = np.asarray(seg)
            else:
                img, seg = load_ssa_case(extra[1], extra[2])
            gt = labels_to_regions(seg); brain = np.any(img != 0, axis=0)
            for sub in subsets:
                im = apply_subset(img, sub)
                probs, det, times = run_methods(model, im, args.device, args.mc)
                probs["mc1_ts"] = _sig(det / tvec); times["mc1_ts"] = times["mc1"]
                full = len(sub) == 4
                for m, p in probs.items():
                    calib.add((cohort, "full" if full else "missing", m), p, gt, brain)
                    reg, u = summarize_prob(p, im)
                    met = region_metrics(reg, gt, with_hd95=False)
                    rows.append({"cohort": cohort, "case_id": cid, "subset": subset_name(sub), "n_modalities": len(sub),
                                 "method": m, "uncertainty": u, "infer_s": times[m], "pred_WT_ml": float(reg[0].sum() / 1000),
                                 "gt_WT_ml": float(gt[0].sum() / 1000),
                                 **{k: met[k] for k in ("dice_WT", "dice_TC", "dice_ET", "dice_mean")}})
            pd.DataFrame(rows).to_csv(os.path.join(args.out, "per_case.csv"), index=False)
            json.dump(calib.report(), open(os.path.join(args.out, "calibration.json"), "w"))
    log.info("done: %d rows", len(rows))


if __name__ == "__main__":
    main()
