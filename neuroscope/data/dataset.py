"""PyTorch datasets over the preprocessed cache.

Cache layout (written by scripts/preprocess.py):
    cache_dir/
        index.json                      # {case_id: {"n_slices":..., "tumour_slices": [...], "meta": {...}}}
        <case_id>_img.npy               # uint8 [4, D', 192, 192]  (memory-mapped)
        <case_id>_seg.npy               # uint8 [D', 192, 192]   labels 0..3

Training samples are 2D axial slices with all 4 modality channels. Modality
dropout zeroes a random subset of channels (at least one kept) so the network
learns to segment from incomplete inputs; the zero channel is exactly what the
app feeds when a sequence is genuinely missing.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .brats import MODALITIES, labels_to_regions
from .preprocess import CropMeta, dequantize_u8

ALL_SUBSETS: List[Tuple[int, ...]] = []  # filled below: 15 non-empty modality subsets
for _mask in range(1, 16):
    ALL_SUBSETS.append(tuple(i for i in range(4) if (_mask >> i) & 1))


def subset_name(subset: Sequence[int]) -> str:
    return "+".join(MODALITIES[i] for i in subset)


def load_index(cache_dir: str) -> Dict:
    with open(os.path.join(cache_dir, "index.json")) as f:
        return json.load(f)


class CachedCase:
    """Lazy memmap access to one preprocessed case."""

    def __init__(self, cache_dir: str, case_id: str, entry: Dict):
        self.case_id = case_id
        self.img = np.load(os.path.join(cache_dir, f"{case_id}_img.npy"), mmap_mode="r")
        seg_path = os.path.join(cache_dir, f"{case_id}_seg.npy")
        self.seg = np.load(seg_path, mmap_mode="r") if os.path.exists(seg_path) else None
        self.meta = CropMeta.from_dict(entry["meta"])
        self.n_slices = int(entry["n_slices"])
        self.tumour_slices = list(entry.get("tumour_slices", []))

    def slice(self, z: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        img = dequantize_u8(np.asarray(self.img[:, z]))
        seg = np.asarray(self.seg[z]) if self.seg is not None else None
        return img, seg

    def volume(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        img = dequantize_u8(np.asarray(self.img))
        seg = np.asarray(self.seg) if self.seg is not None else None
        return img, seg


class SliceDataset(Dataset):
    """2D training dataset with modality dropout and light augmentation."""

    def __init__(
        self,
        cache_dir: str,
        case_ids: List[str],
        modality_dropout_p: float = 0.5,
        neg_slice_frac: float = 0.3,
        augment: bool = True,
        seed: int = 0,
    ):
        self.index = load_index(cache_dir)
        self.cases = {cid: CachedCase(cache_dir, cid, self.index[cid]) for cid in case_ids}
        rng = np.random.RandomState(seed)
        self.samples: List[Tuple[str, int]] = []
        for cid, c in self.cases.items():
            tum = set(c.tumour_slices)
            neg = [z for z in range(c.n_slices) if z not in tum]
            keep_neg = rng.choice(neg, int(len(neg) * neg_slice_frac), replace=False) if neg else []
            self.samples += [(cid, z) for z in sorted(tum)]
            self.samples += [(cid, int(z)) for z in keep_neg]
        self.modality_dropout_p = modality_dropout_p
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        cid, z = self.samples[i]
        img, seg = self.cases[cid].slice(z)
        img = img.copy()
        regions = labels_to_regions(seg).astype(np.float32)
        present = np.ones(4, dtype=np.float32)
        if self.modality_dropout_p > 0 and np.random.rand() < self.modality_dropout_p:
            subset = ALL_SUBSETS[np.random.randint(len(ALL_SUBSETS))]
            for m in range(4):
                if m not in subset:
                    img[m] = 0.0
                    present[m] = 0.0
        if self.augment:
            if np.random.rand() < 0.5:  # left-right flip (brain is roughly symmetric)
                img = img[:, :, ::-1].copy()
                regions = regions[:, :, ::-1].copy()
            # small random intensity scale/shift per modality (scanner gain variation)
            scale = np.random.uniform(0.9, 1.1, size=(4, 1, 1)).astype(np.float32)
            shift = np.random.uniform(-0.1, 0.1, size=(4, 1, 1)).astype(np.float32)
            nz = img != 0
            img = np.where(nz, img * scale + shift, 0.0).astype(np.float32)
        return (
            torch.from_numpy(img),
            torch.from_numpy(regions),
            torch.from_numpy(present),
        )


def apply_subset(img: np.ndarray, subset: Sequence[int]) -> np.ndarray:
    """Zero every modality channel not in `subset`. img: [4, ...]."""
    out = img.copy()
    for m in range(4):
        if m not in subset:
            out[m] = 0.0
    return out
