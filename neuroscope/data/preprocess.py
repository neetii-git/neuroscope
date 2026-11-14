"""Reproducible MRI preprocessing shared by training, evaluation and the app.

Pipeline per case (all modalities processed identically so masks stay aligned):
    1. Brain mask = union of non-zero voxels across available modalities
       (BraTS volumes are already skull-stripped and co-registered).
    2. Per-modality z-score normalisation using brain-voxel statistics only:
           x' = (x - mean_brain) / (std_brain + eps), background stays 0.
       Rationale: MRI intensities are in arbitrary units that differ across
       scanners/sequences; normalising within the brain removes the dominant
       background from the statistics.
    3. Clip to [CLIP_LO, CLIP_HI] standard deviations to bound outliers.
    4. Crop axial plane to a fixed CROP x CROP window centred on the brain
       bounding box (BraTS grids are 240x240 with wide empty margins) and
       restrict the slice range to slices that contain brain.
    5. Optional uint8 quantisation for a compact on-disk cache.

Orientation: arrays are kept as stored in the NIfTI (H, W, D); the slice axis
is the last axis (axial for BraTS). We transpose to (D, H, W) for the cache.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .brats import MODALITIES

CROP = 192
CLIP_LO = -4.0
CLIP_HI = 6.0
EPS = 1e-6


@dataclass
class CropMeta:
    orig_shape: Tuple[int, int, int]  # (H, W, D)
    h0: int
    w0: int
    z0: int
    z1: int  # exclusive
    crop: int = CROP

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        d = dict(d)
        d["orig_shape"] = tuple(d["orig_shape"])
        return CropMeta(**d)


def brain_mask(vols: Dict[str, np.ndarray]) -> np.ndarray:
    mask = None
    for m in MODALITIES:
        if m in vols and vols[m] is not None:
            cur = vols[m] != 0
            mask = cur if mask is None else (mask | cur)
    if mask is None:
        raise ValueError("No modality volumes supplied")
    return mask


def zscore_in_mask(vol: np.ndarray, mask: np.ndarray) -> np.ndarray:
    vals = vol[mask]
    if vals.size == 0:
        return np.zeros_like(vol, dtype=np.float32)
    mean, std = float(vals.mean()), float(vals.std())
    out = np.zeros_like(vol, dtype=np.float32)
    out[mask] = (vals - mean) / (std + EPS)
    return np.clip(out, CLIP_LO, CLIP_HI)


def compute_crop(mask: np.ndarray, crop: int = CROP) -> CropMeta:
    H, W, D = mask.shape
    hs, ws, zs = np.where(mask)
    if hs.size == 0:
        return CropMeta((H, W, D), 0, 0, 0, D, crop)
    hc = int((hs.min() + hs.max()) // 2)
    wc = int((ws.min() + ws.max()) // 2)
    h0 = int(np.clip(hc - crop // 2, 0, max(H - crop, 0)))
    w0 = int(np.clip(wc - crop // 2, 0, max(W - crop, 0)))
    z0, z1 = int(zs.min()), int(zs.max()) + 1
    return CropMeta((H, W, D), h0, w0, z0, z1, crop)


def _crop_pad(vol: np.ndarray, meta: CropMeta) -> np.ndarray:
    """Crop (H,W,D) -> (D', crop, crop), zero-padding if the volume is smaller."""
    c = meta.crop
    sub = vol[meta.h0 : meta.h0 + c, meta.w0 : meta.w0 + c, meta.z0 : meta.z1]
    out = np.zeros((c, c, meta.z1 - meta.z0), dtype=sub.dtype)
    out[: sub.shape[0], : sub.shape[1], :] = sub
    return np.ascontiguousarray(np.transpose(out, (2, 0, 1)))


def uncrop(vol_dhw: np.ndarray, meta: CropMeta) -> np.ndarray:
    """Place a (D', crop, crop) array back into the original (H, W, D) grid."""
    H, W, D = meta.orig_shape
    out = np.zeros((H, W, D), dtype=vol_dhw.dtype)
    c = meta.crop
    hw = np.transpose(vol_dhw, (1, 2, 0))  # (crop, crop, D')
    h_end = min(meta.h0 + c, H)
    w_end = min(meta.w0 + c, W)
    out[meta.h0 : h_end, meta.w0 : w_end, meta.z0 : meta.z1] = hw[
        : h_end - meta.h0, : w_end - meta.w0, :
    ]
    return out


def preprocess_case(
    vols: Dict[str, np.ndarray],
    available: Optional[List[str]] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], CropMeta]:
    """Normalise + crop a case.

    Returns:
        img: float32 [4, D', CROP, CROP]; missing modalities are all-zero channels
        seg: uint8 [D', CROP, CROP] or None
        meta: CropMeta needed to map predictions back to the original grid
    """
    available = available or [m for m in MODALITIES if m in vols]
    mask = brain_mask({m: vols[m] for m in available})
    meta = compute_crop(mask)
    channels = []
    for m in MODALITIES:
        if m in available:
            channels.append(_crop_pad(zscore_in_mask(vols[m], mask), meta))
        else:
            channels.append(
                np.zeros((meta.z1 - meta.z0, meta.crop, meta.crop), dtype=np.float32)
            )
    img = np.stack(channels, axis=0).astype(np.float32)
    seg = _crop_pad(vols["seg"].astype(np.uint8), meta) if "seg" in vols else None
    return img, seg, meta


def quantize_u8(img: np.ndarray) -> np.ndarray:
    """Map clipped z-scores [CLIP_LO, CLIP_HI] -> uint8 [0, 255] (step ~0.04 sd)."""
    scaled = (img - CLIP_LO) / (CLIP_HI - CLIP_LO) * 255.0
    return np.clip(np.rint(scaled), 0, 255).astype(np.uint8)


def dequantize_u8(q: np.ndarray) -> np.ndarray:
    return (q.astype(np.float32) / 255.0) * (CLIP_HI - CLIP_LO) + CLIP_LO
