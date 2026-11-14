"""No-reference MRI input quality assessment (pre-inference gate).

These are *screening heuristics*, not a guarantee of diagnostic suitability.
Each check returns a value, a pass/warn status and a human-readable message.

Checks (per available modality, on the raw volume):
    missing        modality absent -> the model runs with a zero channel; warn
    shape          volume shape differs from the BraTS grid (240x240x155)
    empty/constant fewer than 1% non-zero voxels or zero variance
    snr            mean(brain) / std(background ring just outside brain); low SNR
                   indicates noise or a non-skull-stripped / mis-masked input
    clipping       fraction of brain voxels at the volume's max value; saturation
    brain_fraction brain voxels / total; too small or too large = wrong FOV / not
                   skull-stripped
    intensity_range after z-scoring, fraction of brain voxels beyond ±CLIP bounds;
                   heavy tails indicate artefacts / unusual contrast (OOD-lite)
    slice_dropout  number of in-brain slices with near-zero signal (missing slices)

Overall: status = "fail" if any check fails (e.g. empty volume), "warn" if any
warns, else "pass". Score in [0,1] = 1 - mean severity over checks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy import ndimage

from ..data.brats import MODALITIES

EXPECTED_SHAPE = (240, 240, 155)


@dataclass
class Check:
    name: str
    modality: str
    value: float
    status: str  # "pass" | "warn" | "fail"
    message: str


@dataclass
class QualityReport:
    status: str
    score: float
    checks: List[Check] = field(default_factory=list)
    missing_modalities: List[str] = field(default_factory=list)

    def to_dict(self):
        return {
            "status": self.status,
            "score": self.score,
            "missing_modalities": self.missing_modalities,
            "checks": [c.__dict__ for c in self.checks],
        }

    def warnings(self) -> List[str]:
        return [f"[{c.modality}] {c.message}" for c in self.checks if c.status != "pass"]


def _severity(status: str) -> float:
    return {"pass": 0.0, "warn": 0.5, "fail": 1.0}[status]


def assess_modality(name: str, vol: np.ndarray) -> List[Check]:
    checks: List[Check] = []
    vol = np.asarray(vol, dtype=np.float32)
    if tuple(vol.shape) != EXPECTED_SHAPE:
        checks.append(Check("shape", name, 0.0, "warn", f"shape {vol.shape} differs from BraTS grid {EXPECTED_SHAPE}; cropping/padding applied"))
    brain = vol != 0
    frac = float(brain.mean())
    if frac < 0.01 or float(vol.std()) == 0.0:
        checks.append(Check("empty", name, frac, "fail", "volume is empty or constant"))
        return checks
    status = "pass" if 0.05 <= frac <= 0.45 else "warn"
    checks.append(Check("brain_fraction", name, frac, status,
                        f"brain occupies {frac:.0%} of the volume" + ("" if status == "pass" else " (unusual field of view / not skull-stripped?)")))

    bvals = vol[brain]
    ring = ndimage.binary_dilation(brain, iterations=3) & ~brain
    bg = vol[ring] if ring.sum() > 100 else np.zeros(1, np.float32)
    snr = float(bvals.mean() / (bg.std() + 1e-6))
    snr = min(snr, 1e4)
    checks.append(Check("snr", name, snr, "pass" if snr >= 20 else "warn",
                        f"SNR estimate {snr:.0f}" + ("" if snr >= 20 else " (noisy or imperfect brain mask)")))

    clip = float((bvals >= bvals.max()).mean())
    checks.append(Check("clipping", name, clip, "pass" if clip < 0.001 else "warn",
                        f"{clip:.2%} of brain voxels saturated"))

    z = (bvals - bvals.mean()) / (bvals.std() + 1e-6)
    tails = float(((z < -4) | (z > 6)).mean())
    checks.append(Check("intensity_tails", name, tails, "pass" if tails < 0.002 else "warn",
                        f"{tails:.2%} of brain voxels outside the normal intensity range"))

    per_slice = brain.reshape(-1, brain.shape[-1]).sum(0)
    zs = np.where(per_slice > 0)[0]
    if zs.size:
        inner = per_slice[zs.min(): zs.max() + 1]
        dropped = int((inner < 0.05 * inner.max()).sum())
        checks.append(Check("slice_dropout", name, float(dropped), "pass" if dropped == 0 else "warn",
                            f"{dropped} near-empty slices inside the brain range"))
    return checks


def assess_case(vols: Dict[str, Optional[np.ndarray]]) -> QualityReport:
    checks: List[Check] = []
    missing = [m for m in MODALITIES if vols.get(m) is None]
    for m in missing:
        checks.append(Check("missing", m, 1.0, "warn", "modality not provided; model will run without it (expect reduced accuracy)"))
    for m in MODALITIES:
        if vols.get(m) is not None:
            checks += assess_modality(m, vols[m])
    if len(missing) == len(MODALITIES):
        checks.append(Check("missing", "all", 1.0, "fail", "no modality provided"))
    statuses = [c.status for c in checks]
    status = "fail" if "fail" in statuses else ("warn" if "warn" in statuses else "pass")
    score = 1.0 - (float(np.mean([_severity(s) for s in statuses])) if statuses else 0.0)
    return QualityReport(status, score, checks, missing)
