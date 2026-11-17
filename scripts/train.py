#!/usr/bin/env python
"""Train the modality-robust 2D U-Net.

Usage:
    python scripts/train.py --cache /kaggle/working/cache --out /kaggle/working/runs/base \
        [--config configs/default.yaml] [--epochs 20] [--no-modality-dropout]

Writes:
    out/best.pt, out/last.pt      checkpoints (model + config + epoch + val metrics)
    out/history.csv               per-epoch train loss / val Dice / time
    out/train_config.json         resolved configuration (for reproducibility)
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from neuroscope.data.brats import labels_to_regions  # noqa: E402
from neuroscope.data.dataset import CachedCase, SliceDataset, load_index  # noqa: E402
from neuroscope.data.split import load_split  # noqa: E402
from neuroscope.eval.metrics import region_metrics  # noqa: E402
from neuroscope.inference.predictor import predict_volume  # noqa: E402
from neuroscope.models.losses import DiceBCELoss  # noqa: E402
from neuroscope.models.unet2d import build_model, count_parameters  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def validate(model, cache_dir, case_ids, device, max_cases):
    index = load_index(cache_dir)
    scores = []
    for cid in case_ids[:max_cases]:
        c = CachedCase(cache_dir, cid, index[cid])
        img, seg = c.volume()
        pred = predict_volume(model, img, mc_samples=1, device=device, batch_size=32)
        m = region_metrics(pred.regions, labels_to_regions(seg), with_hd95=False)
        scores.append(m)
    keys = ["dice_WT", "dice_TC", "dice_ET", "dice_mean"]
    return {k: float(np.mean([s[k] for s in scores])) for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml"))
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--no-modality-dropout", action="store_true", help="ablation: train on full modalities only")
    ap.add_argument("--max-slices-per-epoch", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--smoke", action="store_true", help="tiny run for testing the pipeline")
    ap.add_argument("--arch", default=None, choices=["unet", "attention_unet", "segresnet"], help="baseline architectures for comparison")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.max_slices_per_epoch is not None:
        cfg["train"]["max_slices_per_epoch"] = args.max_slices_per_epoch
    if args.arch is not None:
        cfg["model"]["arch"] = args.arch
    if args.no_modality_dropout:
        cfg["data"]["modality_dropout_p"] = 0.0
    if args.smoke:
        cfg["train"].update({"epochs": 1, "batch_size": 2, "max_slices_per_epoch": 8, "val_max_cases": 2})
        cfg["model"]["base_channels"] = 8
        cfg["data"]["num_workers"] = 0
    cfg["train"]["amp"] = bool(cfg["train"].get("amp", True)) and args.device.startswith("cuda")

    os.makedirs(args.out, exist_ok=True)
    set_seed(int(cfg["seed"]))
    split = load_split(os.path.join(args.cache, "split.json"))
    cfg["split"] = {k: len(v) for k, v in split.items() if isinstance(v, list)}
    with open(os.path.join(args.out, "train_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    ds = SliceDataset(
        args.cache, split["train"],
        modality_dropout_p=float(cfg["data"]["modality_dropout_p"]),
        neg_slice_frac=float(cfg["data"]["neg_slice_frac"]),
        augment=True, seed=int(cfg["seed"]),
    )
    n_per_epoch = cfg["train"].get("max_slices_per_epoch") or len(ds)
    sampler = RandomSampler(ds, replacement=False, num_samples=min(n_per_epoch, len(ds)))
    loader = DataLoader(
        ds, batch_size=int(cfg["train"]["batch_size"]), sampler=sampler,
        num_workers=int(cfg["data"]["num_workers"]), pin_memory=args.device.startswith("cuda"), drop_last=True,
    )
    log.info("Train slices: %d (using %d per epoch), val cases: %d", len(ds), min(n_per_epoch, len(ds)), len(split["val"]))

    model = build_model(cfg["model"]).to(args.device)
    log.info("Model parameters: %.2fM", count_parameters(model) / 1e6)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr"]), weight_decay=float(cfg["train"]["weight_decay"]))
    epochs = int(cfg["train"]["epochs"])
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=float(cfg["train"]["lr"]), total_steps=epochs * len(loader), pct_start=0.1,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["train"]["amp"])
    criterion = DiceBCELoss()

    hist_path = os.path.join(args.out, "history.csv")
    with open(hist_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_dice_WT", "val_dice_TC", "val_dice_ET", "val_dice_mean", "epoch_time_s", "lr"])

    best = -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        losses = []
        pbar = tqdm(loader, desc=f"epoch {epoch}/{epochs}", leave=False)
        for img, target, _present in pbar:
            img, target = img.to(args.device, non_blocking=True), target.to(args.device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=cfg["train"]["amp"]):
                logits = model(img)
                loss = criterion(logits.float(), target)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())
            pbar.set_postfix(loss=f"{np.mean(losses[-50:]):.4f}")
        train_loss = float(np.mean(losses))

        val = validate(model, args.cache, split["val"], args.device, int(cfg["train"]["val_max_cases"]))
        dt = time.time() - t0
        log.info("epoch %d loss %.4f | val Dice WT %.3f TC %.3f ET %.3f mean %.3f | %.0fs",
                 epoch, train_loss, val["dice_WT"], val["dice_TC"], val["dice_ET"], val["dice_mean"], dt)
        with open(hist_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, train_loss, val["dice_WT"], val["dice_TC"], val["dice_ET"], val["dice_mean"], dt, sched.get_last_lr()[0]])

        state = {"model": model.state_dict(), "config": cfg["model"], "full_config": cfg, "epoch": epoch, "val": val}
        torch.save(state, os.path.join(args.out, "last.pt"))
        if val["dice_mean"] > best:
            best = val["dice_mean"]
            torch.save(state, os.path.join(args.out, "best.pt"))
            log.info("  new best (val mean Dice %.3f)", best)
    log.info("Done. Best val mean Dice %.3f -> %s", best, os.path.join(args.out, "best.pt"))


if __name__ == "__main__":
    main()
