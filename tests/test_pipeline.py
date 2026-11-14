"""Unit tests (run: pytest -q). Use synthetic data only; no real MRI required."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from neuroscope.data.brats import labels_to_regions, regions_to_labels, remap_labels  # noqa: E402
from neuroscope.data.dataset import ALL_SUBSETS, apply_subset  # noqa: E402
from neuroscope.data.preprocess import dequantize_u8, preprocess_case, quantize_u8, uncrop  # noqa: E402
from neuroscope.data.split import make_split  # noqa: E402
from neuroscope.eval import reliability  # noqa: E402
from neuroscope.eval.metrics import dice, hd95, region_metrics  # noqa: E402
from neuroscope.models.losses import DiceBCELoss  # noqa: E402
from neuroscope.models.unet2d import UNet2D  # noqa: E402
from neuroscope.quality.iqa import assess_case  # noqa: E402


def _fake_case(shape=(64, 64, 20)):
    rng = np.random.RandomState(0)
    vols = {}
    brain = np.zeros(shape, bool)
    brain[8:56, 8:56, 2:18] = True
    for m in ["t1", "t1ce", "t2", "flair"]:
        v = np.zeros(shape, np.float32)
        v[brain] = 100 + rng.normal(0, 5, brain.sum())
        vols[m] = v
    seg = np.zeros(shape, np.int16)
    seg[20:30, 20:30, 6:12] = 2
    seg[23:27, 23:27, 8:10] = 4
    vols["seg"] = remap_labels(seg)
    return vols


def test_label_roundtrip():
    lab = np.random.RandomState(1).randint(0, 4, size=(5, 6, 7)).astype(np.uint8)
    assert np.array_equal(regions_to_labels(labels_to_regions(lab)), lab)


def test_remap():
    assert set(np.unique(remap_labels(np.array([0, 1, 2, 4])))) == {0, 1, 2, 3}


def test_preprocess_shapes_and_uncrop():
    vols = _fake_case()
    img, seg, meta = preprocess_case(vols)
    assert img.shape[0] == 4 and img.shape[2:] == (192, 192) and seg.shape == img.shape[1:]
    assert img.shape[1] == meta.z1 - meta.z0 == 16
    back = uncrop(seg, meta)
    assert back.shape == vols["seg"].shape and np.array_equal(back, vols["seg"])
    # background stays exactly zero after quantisation round trip
    q = dequantize_u8(quantize_u8(img))
    assert np.all(q[img == 0] == 0) and np.abs(q - img).max() < 0.05


def test_missing_modality_channel_is_zero():
    vols = _fake_case()
    img, _, _ = preprocess_case(vols, available=["t1", "flair"])
    assert np.all(img[1] == 0) and np.all(img[2] == 0) and np.any(img[0] != 0)
    assert len(ALL_SUBSETS) == 15 and all(len(s) >= 1 for s in ALL_SUBSETS)
    assert np.all(apply_subset(img, (0,))[3] == 0)


def test_split_no_leakage():
    s = make_split([f"c{i}" for i in range(50)], seed=1)
    assert not set(s["train"]) & set(s["test"]) and not set(s["val"]) & set(s["test"])
    assert len(s["train"]) + len(s["val"]) + len(s["test"]) == 50


def test_model_forward_and_mc():
    m = UNet2D(base=8)
    x = torch.randn(2, 4, 64, 64)
    y = m(x)
    assert y.shape == (2, 3, 64, 64)
    m.enable_mc_dropout()
    with torch.no_grad():
        a, b = m(x), m(x)
    assert not torch.allclose(a, b), "MC dropout should make passes stochastic"
    loss = DiceBCELoss()(y, (torch.rand(2, 3, 64, 64) > 0.5).float())
    assert torch.isfinite(loss)


def test_metrics():
    a = np.zeros((10, 10, 10), bool); a[2:6, 2:6, 2:6] = True
    assert dice(a, a) == 1.0 and hd95(a, a) == 0.0
    b = np.zeros_like(a); b[3:7, 2:6, 2:6] = True
    assert 0 < dice(a, b) < 1 and hd95(a, b) >= 1.0
    assert dice(np.zeros_like(a), np.zeros_like(a)) == 1.0
    regs = np.stack([a, a, a]); m = region_metrics(regs, regs)
    assert m["dice_mean"] == 1.0


def test_reliability_threshold_and_auroc():
    u = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.95])
    d = np.array([0.9, 0.85, 0.8, 0.5, 0.4, 0.3])
    assert reliability.auroc(u, d < 0.75) == 1.0
    t = reliability.choose_threshold(u, d, 0.75)
    assert t["rule"] == "youden" and 0.3 < t["threshold"] <= 0.8
    perf = reliability.flag_performance(u, d, t["threshold"], 0.75)
    assert perf["flag_sensitivity"] == 1.0 and perf["dice_accepted"] > perf["dice_all"]


def test_quality_gate():
    vols = _fake_case()
    r = assess_case({m: vols[m] for m in ["t1", "t1ce", "t2", "flair"]})
    assert r.status in ("pass", "warn") and 0 <= r.score <= 1
    r2 = assess_case({"t1": vols["t1"], "t1ce": None, "t2": None, "flair": None})
    assert r2.missing_modalities == ["t1ce", "t2", "flair"] and r2.status == "warn"
    assert assess_case({m: None for m in ["t1", "t1ce", "t2", "flair"]}).status == "fail"
