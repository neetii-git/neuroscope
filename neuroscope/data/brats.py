"""BraTS 2020 case discovery and raw loading.

Kaggle layout (awsaf49/brats20-dataset-training-validation):
    .../MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/
        BraTS20_Training_001_flair.nii
        BraTS20_Training_001_t1.nii
        BraTS20_Training_001_t1ce.nii
        BraTS20_Training_001_t2.nii
        BraTS20_Training_001_seg.nii
Case 355 has a mis-named segmentation file (W39_1998.09.19_Segm.nii); handled below.
"""
from __future__ import annotations

import glob
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import nibabel as nib
import numpy as np

log = logging.getLogger(__name__)

MODALITIES: List[str] = ["t1", "t1ce", "t2", "flair"]
MODALITY_INDEX: Dict[str, int] = {m: i for i, m in enumerate(MODALITIES)}

# BraTS raw labels: 0 background, 1 necrotic/non-enhancing core, 2 edema, 4 enhancing.
# We remap 4 -> 3 so labels are contiguous.
RAW_LABEL_REMAP = {0: 0, 1: 1, 2: 2, 4: 3}

REGION_NAMES = ["WT", "TC", "ET"]  # whole tumour, tumour core, enhancing tumour


@dataclass
class BratsCase:
    case_id: str
    directory: str
    modality_paths: Dict[str, str]
    seg_path: Optional[str]

    @property
    def has_all_modalities(self) -> bool:
        return all(m in self.modality_paths for m in MODALITIES)


def _find_file(directory: str, patterns: List[str]) -> Optional[str]:
    for pat in patterns:
        hits = sorted(glob.glob(os.path.join(directory, pat)))
        if hits:
            return hits[0]
    return None


def discover_cases(root: str, require_seg: bool = True) -> List[BratsCase]:
    """Find every BraTS case directory under `root` (searched recursively).

    A case directory is any directory containing at least one `*_flair.nii*` file.
    """
    flair_files = sorted(
        glob.glob(os.path.join(root, "**", "*_flair.nii*"), recursive=True)
    )
    cases: List[BratsCase] = []
    for flair in flair_files:
        directory = os.path.dirname(flair)
        case_id = os.path.basename(directory)
        modality_paths: Dict[str, str] = {}
        for m in MODALITIES:
            p = _find_file(directory, [f"*_{m}.nii.gz", f"*_{m}.nii"])
            if p is not None:
                modality_paths[m] = p
        seg = _find_file(
            directory,
            ["*_seg.nii.gz", "*_seg.nii", "*Segm.nii.gz", "*Segm.nii", "*seg*.nii*"],
        )
        if require_seg and seg is None:
            log.warning("Skipping %s: no segmentation found", case_id)
            continue
        if len(modality_paths) != len(MODALITIES):
            log.warning(
                "Skipping %s: found modalities %s", case_id, sorted(modality_paths)
            )
            continue
        cases.append(BratsCase(case_id, directory, modality_paths, seg))
    log.info("Discovered %d cases under %s", len(cases), root)
    return cases


def load_nifti(path: str) -> np.ndarray:
    """Load a NIfTI volume as float32 array in its stored orientation (H, W, D)."""
    img = nib.load(path)
    return np.asarray(img.dataobj, dtype=np.float32)


def load_nifti_with_affine(path: str):
    img = nib.load(path)
    return np.asarray(img.dataobj, dtype=np.float32), img.affine, img.header


def remap_labels(seg: np.ndarray) -> np.ndarray:
    out = np.zeros_like(seg, dtype=np.uint8)
    for raw, new in RAW_LABEL_REMAP.items():
        out[seg == raw] = new
    return out


def labels_to_regions(label: np.ndarray) -> np.ndarray:
    """Convert a label map (0..3) into 3 binary region channels [WT, TC, ET]."""
    wt = label > 0
    tc = (label == 1) | (label == 3)
    et = label == 3
    return np.stack([wt, tc, et], axis=0).astype(np.uint8)


def regions_to_labels(regions: np.ndarray) -> np.ndarray:
    """Inverse of labels_to_regions for binary region predictions [3, ...]."""
    wt, tc, et = regions[0] > 0, regions[1] > 0, regions[2] > 0
    label = np.zeros(wt.shape, dtype=np.uint8)
    label[wt] = 2  # edema by default
    label[tc] = 1  # tumour core (necrotic / non-enhancing)
    label[et] = 3  # enhancing
    return label


def load_case(case: BratsCase) -> Dict[str, np.ndarray]:
    """Load all modalities (+ seg) of a case into a dict of float32 arrays."""
    vols = {m: load_nifti(case.modality_paths[m]) for m in MODALITIES}
    if case.seg_path is not None:
        vols["seg"] = remap_labels(load_nifti(case.seg_path).astype(np.int16))
    return vols
