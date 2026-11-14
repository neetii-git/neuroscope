#!/usr/bin/env python
"""Build the preprocessed cache + patient-level split from raw BraTS 2020.

Usage:
    python scripts/preprocess.py --raw /kaggle/input/brats20-dataset-training-validation \
        --cache /kaggle/working/cache [--max-patients 60]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from neuroscope.data.brats import discover_cases, load_case  # noqa: E402
from neuroscope.data.preprocess import preprocess_case, quantize_u8  # noqa: E402
from neuroscope.data.split import make_split, save_split  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("preprocess")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="root containing BraTS case dirs")
    ap.add_argument("--cache", required=True, help="output cache directory")
    ap.add_argument("--max-patients", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--test-frac", type=float, default=0.20)
    args = ap.parse_args()

    os.makedirs(args.cache, exist_ok=True)
    cases = discover_cases(args.raw)
    if args.max_patients:
        cases = cases[: args.max_patients]
    if not cases:
        log.error("No cases found under %s", args.raw)
        sys.exit(1)

    index = {}
    t0 = time.time()
    for case in tqdm(cases, desc="preprocess"):
        try:
            vols = load_case(case)
            img, seg, meta = preprocess_case(vols)
        except Exception as e:  # noqa: BLE001
            log.error("Failed %s: %s", case.case_id, e)
            continue
        np.save(os.path.join(args.cache, f"{case.case_id}_img.npy"), quantize_u8(img))
        np.save(os.path.join(args.cache, f"{case.case_id}_seg.npy"), seg)
        tumour_slices = [int(z) for z in np.where(seg.reshape(seg.shape[0], -1).max(1) > 0)[0]]
        index[case.case_id] = {
            "n_slices": int(img.shape[1]),
            "tumour_slices": tumour_slices,
            "tumour_voxels": int((seg > 0).sum()),
            "meta": meta.to_dict(),
        }
    with open(os.path.join(args.cache, "index.json"), "w") as f:
        json.dump(index, f)
    split = make_split(list(index), args.seed, args.val_frac, args.test_frac)
    save_split(split, os.path.join(args.cache, "split.json"))
    log.info(
        "Cached %d cases in %.1f min -> train %d / val %d / test %d",
        len(index), (time.time() - t0) / 60, len(split["train"]), len(split["val"]), len(split["test"]),
    )


if __name__ == "__main__":
    main()
