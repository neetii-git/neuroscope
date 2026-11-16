"""Clinical tools: RANO-style bidimensional measurement, component review / correction, follow-up comparison."""
from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage as ndi

LOG2 = float(np.log(2.0))


# ------------------------------------------------------------------ spacing
def inplane_spacing(affine: np.ndarray) -> Tuple[float, float]:
    """Pixel spacing (mm) of the cropped axial grid: axis 1 is original axis 0, axis 2 is original axis 1."""
    z = np.linalg.norm(np.asarray(affine)[:3, :3], axis=0)
    return float(z[0] or 1.0), float(z[1] or 1.0)


# ------------------------------------------------------------------ RANO-style bidimensional measurement
def _hull_points(mask: np.ndarray, sr: float, sc: float) -> np.ndarray:
    edge = mask & ~ndi.binary_erosion(mask)
    pts = np.argwhere(edge).astype(np.float64)
    pts[:, 0] *= sr
    pts[:, 1] *= sc
    if len(pts) > 600:
        pts = pts[np.linspace(0, len(pts) - 1, 600).astype(int)]
    return pts


def bidimensional(labels: np.ndarray, affine: np.ndarray, target: str = "ET") -> Optional[Dict]:
    """Longest in-plane diameter of the target region and the longest diameter perpendicular to it, on the axial slice
    where their product is largest (as in RANO). Measured on the largest connected component of each slice."""
    sr, sc = inplane_spacing(affine)
    mask3 = (labels == 3) if target == "ET" else ((labels == 1) | (labels == 3))
    if not mask3.any():
        return None
    best = None
    areas = mask3.reshape(mask3.shape[0], -1).sum(1)
    for z in np.argsort(areas)[::-1][:16]:
        if areas[z] < 6:
            continue
        sl = ndi.binary_fill_holes(ndi.binary_closing(mask3[z], structure=np.ones((3, 3)), iterations=2))
        cc, n = ndi.label(sl)
        m = cc == (np.argmax(np.bincount(cc.ravel())[1:]) + 1) if n > 1 else sl
        pts = _hull_points(m, sr, sc)
        if len(pts) < 3:
            continue
        d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
        i, j = np.unravel_index(np.argmax(d), d.shape)
        d1 = float(d[i, j])
        u = (pts[j] - pts[i]) / (d1 + 1e-9)
        w = np.array([-u[1], u[0]])
        allp = np.argwhere(m).astype(np.float64)
        allp[:, 0] *= sr
        allp[:, 1] *= sc
        pu, pw = allp @ u, allp @ w
        bins = np.round(pu).astype(int)
        d2, q = 0.0, None
        for bval in np.unique(bins):
            sel = bins == bval
            lo, hi = pw[sel].min(), pw[sel].max()
            if hi - lo > d2:
                d2 = float(hi - lo + min(sr, sc))
                t = float(bval)
                q = (t * u + lo * w, t * u + hi * w)
        if q is None:
            continue
        d1 += min(sr, sc)
        prod = d1 * d2
        if best is None or prod > best["product_mm2"]:
            to_px = lambda p: (float(p[0] / sr), float(p[1] / sc))
            best = {"slice": int(z), "d1_mm": d1, "d2_mm": d2, "product_mm2": prod, "target": target,
                    "p1": to_px(pts[i]), "p2": to_px(pts[j]), "q1": to_px(q[0]), "q2": to_px(q[1]),
                    "measurable": bool(d1 >= 10 and d2 >= 10)}
    return best


# ------------------------------------------------------------------ connected components and correction
def components(labels: np.ndarray, entropy: np.ndarray, voxel_ml: float) -> List[Dict]:
    wt = labels > 0
    cc, n = ndi.label(wt)
    if n == 0:
        return []
    out = []
    sizes = ndi.sum(wt, cc, range(1, n + 1))
    for k in range(1, n + 1):
        idx = np.argwhere(cc == k)
        m = cc == k
        comp_lab = labels[m]
        out.append({"id": k, "ml": float(sizes[k - 1] * voxel_ml), "voxels": int(sizes[k - 1]),
                    "slices": (int(idx[:, 0].min()), int(idx[:, 0].max())), "centre_slice": int(np.median(idx[:, 0])),
                    "mean_u": float(entropy[m].mean() / LOG2),
                    "composition": {lab: float((comp_lab == lab).sum() * voxel_ml) for lab in (1, 2, 3)}})
    out.sort(key=lambda c: -c["ml"])
    for r, c in enumerate(out):
        c["rank"] = r + 1
    return out


def component_map(labels: np.ndarray) -> np.ndarray:
    return ndi.label(labels > 0)[0]


def apply_rejections(res: Dict, rejected: List[int]) -> Dict:
    """Return a shallow copy of a result with the rejected connected components removed."""
    if not rejected:
        return res
    pred = res["pred"]
    cc = component_map(pred.labels)
    keep = ~np.isin(cc, list(rejected))
    labels = (pred.labels * keep).astype(pred.labels.dtype)
    regions = np.stack([labels > 0, (labels == 1) | (labels == 3), labels == 3]).astype(np.uint8)
    ml = res["voxel_ml"]
    vols = dict(pred.volumes_ml)
    vols.update({"WT_ml": float(regions[0].sum() * ml), "TC_ml": float(regions[1].sum() * ml), "ET_ml": float(regions[2].sum() * ml)})
    wt = labels > 0
    cu = float(pred.entropy[wt].mean()) if wt.any() else 0.0
    new_pred = dataclasses.replace(pred, labels=labels, regions=regions, volumes_ml=vols,
                                   case_uncertainty=cu if wt.any() else pred.case_uncertainty)
    removed = float((pred.labels > 0).sum() - wt.sum()) * ml
    out = dict(res)
    out.update(pred=new_pred, meshes=None, _analysis=None, rejected=list(rejected), removed_ml=removed,
               token=(res["token"], tuple(sorted(rejected))))
    out.pop("_report_imgs", None)
    return out


# ------------------------------------------------------------------ follow-up
def response_category(prior: Optional[Dict], current: Optional[Dict], prior_et_ml: float, cur_et_ml: float) -> Tuple[str, str]:
    """Imaging-only approximation of the RANO enhancing-disease criteria."""
    if prior_et_ml > 0.05 and cur_et_ml <= 0.05:
        return "Complete response (imaging only)", "No enhancing tumour on the later scan."
    if not prior or not current:
        return "Not evaluable", "Enhancing tumour too small to measure on one of the scans."
    ch = (current["product_mm2"] - prior["product_mm2"]) / max(prior["product_mm2"], 1e-6) * 100
    if ch >= 25:
        return "Progressive disease (imaging only)", f"Product of perpendicular diameters increased {ch:.0f}% (threshold +25%)."
    if ch <= -50:
        return "Partial response (imaging only)", f"Product of perpendicular diameters decreased {abs(ch):.0f}% (threshold -50%)."
    return "Stable disease (imaging only)", f"Product of perpendicular diameters changed {ch:+.0f}% (between -50% and +25%)."


def simulate_prior(vols: Dict[str, np.ndarray], scale: float = 0.78) -> Dict[str, np.ndarray]:
    """Demonstration only: a synthetic 'earlier' scan made by digitally shrinking the tumour region of the example
    patient with a smooth radial warp. Used so the follow-up workflow can be shown on a dataset with one time point."""
    seg = vols.get("seg")
    ref = vols[[k for k in vols if k != "seg"][0]]
    if seg is not None and (seg > 0).any():
        idx = np.argwhere(seg > 0)
        c = idx.mean(0)
        r0 = float(np.cbrt(3 * len(idx) / (4 * np.pi)))
    else:
        c = np.array(ref.shape) / 2.0
        r0 = 20.0
    r1 = r0 * 2.6
    grid = np.indices(ref.shape, dtype=np.float32)
    dx = [grid[i] - c[i] for i in range(3)]
    r = np.sqrt(sum(d * d for d in dx))
    w = np.clip((r1 - r) / (r1 - r0), 0, 1)
    w = 0.5 - 0.5 * np.cos(np.pi * w)
    k = 1.0 + (1.0 / scale - 1.0) * w
    coords = np.stack([c[i] + dx[i] * k for i in range(3)])
    out = {}
    for m, v in vols.items():
        if m == "seg":
            continue
        out[m] = ndi.map_coordinates(v.astype(np.float32), coords, order=1, mode="nearest").astype(v.dtype if v.dtype.kind == "f" else np.float32)
    return out
