"""Volume-level inference with Monte Carlo dropout uncertainty.

For a preprocessed case img [4, D, H, W] the predictor runs every axial slice
through the 2D U-Net T times with dropout active and returns:

    mean_prob   [3, D, H, W]  mean sigmoid probability per region (prediction)
    entropy     [D, H, W]     predictive entropy of the WT channel:
                              H(p) = -(p log p + (1-p) log(1-p))   (nats)
    mc_std      [D, H, W]     std of WT probability across the T passes
                              (epistemic spread; 0 if T == 1)

Case-level uncertainty score (used for the human-review flag):
    u_case = mean entropy over the "region of interest" = predicted WT dilated
             by 2 voxels, union with any voxel whose mean prob is in [0.1, 0.9].
    Averaging only near the tumour avoids diluting the score with millions of
    confidently-background voxels, so it reflects how sure the model is about
    the tumour it found. If nothing is predicted the ROI falls back to the
    brain mask (a confident "no tumour" is itself informative).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from scipy import ndimage

from ..data.brats import regions_to_labels
from ..data.dataset import apply_subset
from ..models.unet2d import UNet2D, build_model


@dataclass
class Prediction:
    mean_prob: np.ndarray  # [3, D, H, W]
    regions: np.ndarray  # [3, D, H, W] uint8, thresholded at 0.5
    labels: np.ndarray  # [D, H, W] uint8 label map 0..3
    entropy: np.ndarray  # [D, H, W]
    mc_std: np.ndarray  # [D, H, W]
    case_uncertainty: float
    volumes_ml: Dict[str, float]
    mc_samples: int


def load_checkpoint(path: str, device: str = "cpu") -> UNet2D:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt.get("config", {}))
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model


def _entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


@torch.no_grad()
def predict_volume(
    model: UNet2D,
    img: np.ndarray,
    mc_samples: int = 8,
    batch_size: int = 16,
    device: str = "cpu",
    subset: Optional[Sequence[int]] = None,
    voxel_volume_mm3: float = 1.0,
) -> Prediction:
    if subset is not None:
        img = apply_subset(img, subset)
    _, D, H, W = img.shape
    x = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (1, 0, 2, 3))))  # [D,4,H,W]
    if mc_samples > 1:
        model.enable_mc_dropout()
    else:
        model.eval()

    prob_sum = np.zeros((3, D, H, W), dtype=np.float32)
    prob_sq = np.zeros((D, H, W), dtype=np.float32)
    use_amp = device.startswith("cuda")
    for _ in range(max(1, mc_samples)):
        probs = np.zeros((D, 3, H, W), dtype=np.float32)
        for s in range(0, D, batch_size):
            xb = x[s : s + batch_size].to(device)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                logits = model(xb)
            probs[s : s + batch_size] = torch.sigmoid(logits.float()).cpu().numpy()
        probs = np.transpose(probs, (1, 0, 2, 3))
        prob_sum += probs
        prob_sq += probs[0] ** 2
    model.eval()
    T = max(1, mc_samples)
    mean_prob = prob_sum / T
    var = np.maximum(prob_sq / T - mean_prob[0] ** 2, 0.0)
    mc_std = np.sqrt(var)
    entropy = _entropy(mean_prob[0])

    regions = (mean_prob > 0.5).astype(np.uint8)
    # enforce hierarchy: ET ⊆ TC ⊆ WT
    regions[1] &= regions[0]
    regions[2] &= regions[1]
    labels = regions_to_labels(regions)

    roi = ndimage.binary_dilation(regions[0].astype(bool), iterations=2)
    roi |= (mean_prob[0] > 0.1) & (mean_prob[0] < 0.9)
    if roi.sum() == 0:
        roi = np.any(img != 0, axis=0)
    case_u = float(entropy[roi].mean()) if roi.sum() > 0 else 0.0

    ml = voxel_volume_mm3 / 1000.0
    volumes = {
        "WT_ml": float(regions[0].sum() * ml),
        "TC_ml": float(regions[1].sum() * ml),
        "ET_ml": float(regions[2].sum() * ml),
    }
    return Prediction(mean_prob, regions, labels, entropy, mc_std, case_u, volumes, T)
