"""Structured case report: findings, fault analysis and a printable HTML document."""
from __future__ import annotations

import html
import time
from typing import Dict, List, Optional

import numpy as np
from scipy import ndimage as ndi

from clinical import bidimensional  # noqa: E402

REGION_NAMES = {1: "Necrotic / non-enhancing core", 2: "Edema", 3: "Enhancing tumour"}
LOG2 = float(np.log(2.0))


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    s = a.sum() + b.sum()
    return float(2.0 * (a & b).sum() / s) if s else float("nan")


def _location(labels: np.ndarray, brain: np.ndarray, meta, affine: np.ndarray) -> Dict:
    """Hemisphere and anterior/superior position of the whole tumour in scanner (RAS) space."""
    try:
        import nibabel as nib
        from neuroscope.data.preprocess import uncrop
        wt = uncrop((labels > 0).astype(np.uint8), meta) > 0
        br = uncrop(brain.astype(np.uint8), meta) > 0
        if wt.sum() == 0 or br.sum() == 0:
            return {}
        c_t = nib.affines.apply_affine(affine, np.argwhere(wt).mean(0))
        c_b = nib.affines.apply_affine(affine, np.argwhere(br).mean(0))
        dx, dy, dz = c_t - c_b
        vox = np.argwhere(wt)
        x_all = nib.affines.apply_affine(affine, vox)[:, 0] - c_b[0]
        right_frac = float((x_all > 0).mean())
        side = "right hemisphere" if right_frac > 0.8 else ("left hemisphere" if right_frac < 0.2 else "crosses the midline")
        return {"side": side, "right_fraction": right_frac,
                "ap": "anterior" if dy > 8 else ("posterior" if dy < -8 else "central"),
                "si": "superior" if dz > 8 else ("inferior" if dz < -8 else "mid-level"),
                "offset_mm": [float(dx), float(dy), float(dz)]}
    except Exception:
        return {}


def analyse(res: Dict, thr: Optional[Dict], flagged: bool, sub_df, test_df, modalities: List[str]) -> Dict:
    pred, quality = res["pred"], res["quality"]
    lab, ent, vox_ml = pred.labels, pred.entropy, res["voxel_ml"]
    wt, brain = lab > 0, res["brain"]
    out: Dict = {"findings": [], "regions": {}, "slices": {}}

    def add(level: str, title: str, detail: str, where: str = ""):
        out["findings"].append({"level": level, "title": title, "detail": detail, "where": where})

    # volumes
    for k, name in REGION_NAMES.items():
        m = lab == k
        out["regions"][k] = {"name": name, "ml": float(m.sum() * vox_ml), "mean_u": float(ent[m].mean() / LOG2) if m.any() else None}
    out["brain_ml"] = float(brain.sum() * vox_ml)
    out["wt_ml"] = float(wt.sum() * vox_ml)
    out["wt_pct_brain"] = 100.0 * out["wt_ml"] / out["brain_ml"] if out["brain_ml"] else 0.0
    out["location"] = _location(lab, brain, res["meta"], res["affine"])
    if wt.any():
        z = np.linalg.norm(np.asarray(res["affine"])[:3, :3], axis=0)
        cc_, n_ = ndi.label(wt)
        main = cc_ == (np.argmax(np.bincount(cc_.ravel())[1:]) + 1) if n_ > 1 else wt
        idx = np.argwhere(main)
        span = idx.max(0) - idx.min(0) + 1          # (axial, orig axis 0, orig axis 1)
        out["extent_mm"] = [float(span[1] * z[0]), float(span[2] * z[1]), float(span[0] * z[2])]  # LR, AP, SI
        out["slice_range"] = [int(idx[:, 0].min()), int(idx[:, 0].max())]

    # case uncertainty
    u, th = float(pred.case_uncertainty), (thr or {}).get("threshold")
    out["u"], out["thr"] = u, th
    if th:
        ratio = u / th
        pct = None
        if test_df is not None and "uncertainty" in test_df.columns:
            pct = float((test_df["uncertainty"] < u).mean() * 100)
            out["n_test"] = int(len(test_df))
        out["u_percentile"] = pct
        if flagged:
            add("critical", "Model uncertainty above the review threshold",
                f"Case uncertainty {u:.4f} is {ratio:.1f} times the validated threshold ({th:.4f}). In the held-out test set, cases above this level "
                "were most likely to have a poor outline (whole-tumour Dice below 0.75).", "Whole case")
        elif ratio > 0.8:
            add("warn", "Uncertainty close to the review threshold", f"Case uncertainty {u:.4f} is {ratio*100:.0f}% of the threshold ({th:.4f}). Review border slices.", "Whole case")

    # missing sequences
    missing = [m for m in modalities if m not in res["present"]]
    if missing and sub_df is not None:
        key = "+".join(m for m in modalities if m in res["present"])
        row = sub_df[sub_df["subset"] == key]
        if len(row):
            r0 = row.iloc[0]
            lvl = "critical" if r0["dice_WT"] < 0.7 else "warn"
            add(lvl, f"Missing sequence{'s' if len(missing) > 1 else ''}: {', '.join(m.upper() for m in missing)}",
                f"With only {key.upper()} the model scored whole tumour {r0['dice_WT']:.2f}, core {r0['dice_TC']:.2f} and enhancing {r0['dice_ET']:.2f} Dice "
                "on the 74 test patients, against 0.90, 0.86 and 0.77 with all four. The regions that depend on the missing sequence are the least reliable.",
                "All slices")
        else:
            add("warn", f"Missing sequences: {', '.join(m.upper() for m in missing)}", "Accuracy is reduced for the regions that depend on them.", "All slices")

    # input quality
    bad = [c for c in quality.checks if c.status != "pass"]
    if bad:
        lvl = "critical" if any(c.status == "fail" for c in bad) else "warn"
        mods = sorted({c.modality.upper() for c in bad})
        add(lvl, f"Input quality: {len(bad)} check{'s' if len(bad) > 1 else ''} raised",
            " ".join(f"{c.modality.upper()}: {c.message.rstrip('.')}." for c in bad[:5]) + (f" {len(bad) - 5} further." if len(bad) > 5 else ""),
            ", ".join(mods))

    # uncertain slices and regions
    if wt.any():
        per = np.array([ent[i][wt[i]].mean() / LOG2 if wt[i].any() else 0.0 for i in range(lab.shape[0])])
        area = wt.reshape(lab.shape[0], -1).sum(1)
        top = [int(i) for i in np.argsort(per)[::-1] if area[i] >= 30][:3]
        out["slices"] = {"most_uncertain": top, "largest": int(np.argmax(area)), "profile": per.tolist()}
        if top:
            add("info", "Most uncertain slices", f"Disagreement between repeated runs is highest on axial slices {', '.join(str(i) for i in sorted(top))}. Recommended for first review.",
                f"Slices {', '.join(str(i) for i in sorted(top))}")
        rs = {k: v for k, v in out["regions"].items() if v["mean_u"] is not None}
        if rs:
            kmax = max(rs, key=lambda k: rs[k]["mean_u"])
            add("info", f"Least certain region: {rs[kmax]['name'].lower()}", f"Average uncertainty {rs[kmax]['mean_u']:.2f} (0 certain, 1 maximal) against "
                + ", ".join(f"{v['mean_u']:.2f} for {v['name'].lower()}" for k, v in rs.items() if k != kmax) + ".", rs[kmax]["name"])
        inner = ndi.binary_erosion(wt, iterations=2)
        edge = wt & ~inner
        if inner.any() and edge.any():
            eu, iu = ent[edge].mean(), ent[inner].mean()
            out["edge_u"], out["inner_u"] = float(eu / LOG2), float(iu / LOG2)
            if eu > 2.0 * iu:
                add("info", "Uncertainty concentrated at tumour border", f"Border voxels are {eu/max(iu,1e-9):.1f} times more uncertain than the interior. Boundary placement is the main source of disagreement.", "Tumour border")

    # fragments
    if wt.any():
        cc, n = ndi.label(wt)
        if n > 1:
            sizes = ndi.sum(wt, cc, range(1, n + 1)) * vox_ml
            order = np.argsort(sizes)[::-1]
            small = [(int(j) + 1, float(sizes[j])) for j in order[1:] if sizes[j] < max(0.5, 0.05 * sizes[order[0]])]
            if small:
                tot = sum(s for _, s in small)
                zs = []
                for j, _ in small[:4]:
                    zi = np.argwhere(cc == j)[:, 0]
                    zs.append(int(np.median(zi)))
                add("warn", f"{len(small)} small separate fragment{'s' if len(small) > 1 else ''} predicted",
                    f"{tot:.2f} mL in total sits apart from the main mass. Isolated fragments are a common source of false positives; confirm or discard.",
                    f"Near slices {', '.join(str(z) for z in sorted(set(zs)))}")
        out["components"] = int(n)

    # small structures
    for k in (1, 3):
        v = out["regions"][k]["ml"]
        if 0 < v < 0.5:
            add("info", f"Very small {REGION_NAMES[k].lower()} ({v:.2f} mL)", "Volumes of small structures are sensitive to a few voxels; interpret with caution.", REGION_NAMES[k])
    if not wt.any():
        add("critical", "No tumour found", "No tumour voxels predicted. If a lesion is expected, check scan orientation and sequences.", "Whole case")

    # ground truth disagreement
    seg = res.get("seg")
    if seg is not None:
        gt = seg > 0
        both = []
        for i in range(lab.shape[0]):
            a, b = wt[i], gt[i]
            if a.sum() + b.sum() >= 40:
                both.append((i, _dice(a, b)))
        fn = float((gt & ~wt).sum() * vox_ml)
        fp = float((wt & ~gt).sum() * vox_ml)
        out["gt"] = {"fn_ml": fn, "fp_ml": fp, "dice": {n: _dice(lab == k, seg == k) for k, n in REGION_NAMES.items()}}
        if both:
            worst = sorted(both, key=lambda t: t[1])[:3]
            out["gt"]["worst_slices"] = [(i, d) for i, d in worst]
            add("info", "Disagreement with reference segmentation",
                f"Under-segmented {fn:.1f} mL, over-segmented {fp:.1f} mL. Lowest per-slice whole-tumour Dice: "
                + ", ".join(f"{i} ({d:.2f})" for i, d in worst) + ".", f"Slices {', '.join(str(i) for i, _ in worst)}")
    # RANO-style bidimensional measurement
    out["rano"] = bidimensional(lab, res["affine"], "ET") or bidimensional(lab, res["affine"], "TC")
    r = out["rano"]
    if r and not r["measurable"]:
        add("info", "Enhancing disease below measurable size", f"Bidimensional measurement {r['d1_mm']:.0f} x {r['d2_mm']:.0f} mm; RANO requires at least 10 mm in both "
            "dimensions for measurable disease.", f"Slice {r['slice']}")
    if res.get("rejected"):
        add("info", "Clinician correction applied", f"{len(res['rejected'])} predicted region{'s' if len(res['rejected']) > 1 else ''} rejected by the reviewer "
            f"({res.get('removed_ml', 0):.2f} mL removed). Volumes and measurements reflect the corrected segmentation.", "Review tab")
    order = {"critical": 0, "warn": 1, "info": 2}
    out["findings"].sort(key=lambda f: order[f["level"]])
    return out


# ------------------------------------------------------------------ figures
_COL = {1: (255, 51, 51), 2: (51, 204, 51), 3: (255, 217, 26)}


def _png(rgb: np.ndarray, scale: int = 2) -> str:
    import base64, io
    from PIL import Image
    im = Image.fromarray(rgb.astype(np.uint8))
    im = im.resize((im.width * scale, im.height * scale), Image.LANCZOS)
    buf = io.BytesIO(); im.save(buf, "PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _edge(m: np.ndarray) -> np.ndarray:
    m = m > 0
    return m & ~ndi.binary_erosion(m)


def _orient(a: np.ndarray, plane: str) -> np.ndarray:
    # axial slices (x, y) are shown like the app's slice viewer; coronal / sagittal with superior at the top
    return np.swapaxes(a, 0, 1)[::-1] if plane == "axial" else a[::-1]


def _panel(gray, lab, ent, mode, plane, gt=None, box=None):
    if box is not None:
        gray, lab, ent = gray[box], lab[box], ent[box]
        gt = gt[box] if gt is not None else None
    rgb = np.repeat(gray[..., None], 3, -1).astype(np.float32)
    if mode == "fill":
        for k, c in _COL.items():
            m = lab == k
            rgb[m] = rgb[m] * 0.45 + np.array(c) * 0.55
    elif mode == "outline":
        for k, c in _COL.items():
            e = _edge(lab == k) if k != 2 else _edge(lab > 0)
            rgb[ndi.binary_dilation(e) if rgb.shape[0] > 120 else e] = c
    elif mode == "compare":
        if gt is not None:
            rgb[_edge(gt > 0)] = (0, 220, 255)
        rgb[_edge(lab > 0)] = (255, 217, 26)
    elif mode == "unc":
        u = np.clip(ent / LOG2, 0, 1)[..., None]
        rgb = rgb * 0.35 + (np.array([255, 245, 220]) * u + rgb * 0.65 * (1 - u)) * 0.65
    return _orient(np.clip(rgb, 0, 255), plane)


def rano_panel(img2d_gray: np.ndarray, labels2d: np.ndarray, meas: Dict, box=None, scale: int = 3) -> str:
    """Axial slice with the two perpendicular diameters drawn, cropped around the measured lesion."""
    from PIL import Image, ImageDraw
    rgb = _panel(img2d_gray, labels2d, np.zeros_like(img2d_gray, dtype=np.float32), "outline", "axial")
    W = img2d_gray.shape[1]
    im = Image.fromarray(rgb.astype(np.uint8)).resize((rgb.shape[1] * scale, rgb.shape[0] * scale), Image.LANCZOS)
    dr = ImageDraw.Draw(im)
    tr = lambda p: (p[0] * scale + scale / 2, (W - 1 - p[1]) * scale + scale / 2)
    for a, b, col in ((meas["p1"], meas["p2"], (245, 197, 24)), (meas["q1"], meas["q2"], (255, 255, 255))):
        dr.line([tr(a), tr(b)], fill=col, width=max(2, scale))
        for p in (a, b):
            x, y = tr(p); dr.ellipse([x - 4, y - 4, x + 4, y + 4], fill=col)
    cx = (meas["p1"][0] + meas["p2"][0]) / 2; cy = W - 1 - (meas["p1"][1] + meas["p2"][1]) / 2
    half = max(30, meas["d1_mm"] * 0.9)
    x0, y0 = max(0, (cx - half) * scale), max(0, (cy - half) * scale)
    x1, y1 = min(im.width, (cx + half) * scale), min(im.height, (cy + half) * scale)
    im = im.crop((int(x0), int(y0), int(x1), int(y1)))
    import base64, io
    buf = io.BytesIO(); im.save(buf, "PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def gray_volume(img: np.ndarray, base_index: int) -> np.ndarray:
    vol = img[base_index]
    nz = vol[vol != 0]
    lo, hi = (np.percentile(nz, [1, 99.5]) if nz.size else (0.0, 1.0))
    return np.clip((vol - lo) / (hi - lo + 1e-6), 0, 1) * 255


def build_figures(img: np.ndarray, labels: np.ndarray, entropy: np.ndarray, seg: Optional[np.ndarray], base_index: int,
                  base_name: str, analysis: Dict) -> List[Dict]:
    vol = img[base_index]
    nz = vol[vol != 0]
    lo, hi = (np.percentile(nz, [1, 99.5]) if nz.size else (0.0, 1.0))
    gray = np.clip((vol - lo) / (hi - lo + 1e-6), 0, 1) * 255
    wt = labels > 0
    figs: List[Dict] = []
    if not wt.any():
        return figs
    idx = np.argwhere(wt)
    cz, cx, cy = (int(round(v)) for v in idx.mean(0))
    # 1. orthogonal views through the tumour centre
    views = [("axial", gray[cz], labels[cz], entropy[cz], None if seg is None else seg[cz], f"Axial, slice {cz}"),
             ("coronal", gray[:, :, cy], labels[:, :, cy], entropy[:, :, cy], None if seg is None else seg[:, :, cy], f"Coronal, position {cy}"),
             ("sagittal", gray[:, cx, :], labels[:, cx, :], entropy[:, cx, :], None if seg is None else seg[:, cx, :], f"Sagittal, position {cx}")]
    figs.append({"cols": 3, "panels": [(_png(_panel(g, l, e, "fill", pl)), t) for pl, g, l, e, _, t in views],
                 "caption": "Orthogonal views through the centre of the segmented tumour, with predicted labels overlaid."})
    r = analysis.get("rano")
    if r:
        z = r["slice"]
        figs.append({"cols": 2, "panels": [(rano_panel(gray[z], labels[z], r), f"Axial slice {z}"),
                                           (_png(_panel(gray[z], labels[z], entropy[z], "fill", "axial"), 2), "Whole slice, predicted labels")],
                     "caption": f"Bidimensional measurement of the {'enhancing tumour' if r['target'] == 'ET' else 'tumour core'}: longest diameter "
                                f"{r['d1_mm']:.1f} mm (yellow) and longest perpendicular diameter {r['d2_mm']:.1f} mm (white)."})
    # crop box around tumour for close-ups
    cc, n = ndi.label(wt)
    main = np.argwhere(cc == (np.argmax(np.bincount(cc.ravel())[1:]) + 1)) if n > 1 else idx
    cy_, cx_ = (main[:, 1].min() + main[:, 1].max()) / 2, (main[:, 2].min() + main[:, 2].max()) / 2
    half = int(max(np.ptp(main[:, 1]), np.ptp(main[:, 2])) / 2 + 16)
    half = max(28, min(half, min(wt.shape[1], wt.shape[2]) // 2))
    y0 = int(np.clip(cy_ - half, 0, wt.shape[1] - 2 * half)); x0 = int(np.clip(cx_ - half, 0, wt.shape[2] - 2 * half))
    box = (slice(y0, y0 + 2 * half), slice(x0, x0 + 2 * half))
    zl = int((analysis.get("slices") or {}).get("largest", cz))
    g, l, e = gray[zl], labels[zl], entropy[zl]
    gt = seg[zl] if seg is not None else None
    panels = [(_png(_panel(g, l, e, "scan", "axial", box=box), 3), base_name),
              (_png(_panel(g, l, e, "fill", "axial", box=box), 3), "Predicted labels"),
              (_png(_panel(g, l, e, "outline", "axial", box=box), 3), "Predicted outline")]
    if gt is not None:
        panels.append((_png(_panel(g, l, e, "compare", "axial", gt=gt, box=box), 3), "Reference (cyan) vs prediction (yellow)"))
    panels.append((_png(_panel(g, l, e, "unc", "axial", box=box), 3), "Uncertainty"))
    figs.append({"cols": len(panels), "panels": panels, "caption": f"Axial slice {zl}, largest tumour cross-section, magnified."})
    # 3. outline montage across the tumour
    z0, z1 = int(idx[:, 0].min()), int(idx[:, 0].max())
    zs = sorted(set(int(round(v)) for v in np.linspace(z0 + (z1 - z0) * 0.06, z1 - (z1 - z0) * 0.06, 8)))
    figs.append({"cols": 4, "panels": [(_png(_panel(gray[z], labels[z], entropy[z], "outline", "axial", box=box), 2), f"Slice {z}") for z in zs],
                 "caption": f"Predicted outlines on {base_name} at {len(zs)} evenly spaced axial levels from slice {z0} to {z1}. "
                            "Red: necrotic core. Yellow: enhancing. Green: outer tumour boundary including edema."})
    # 4. most uncertain slice
    mu = (analysis.get("slices") or {}).get("most_uncertain") or []
    if mu and mu[0] != zl:
        zu = int(mu[0]); g, l, e = gray[zu], labels[zu], entropy[zu]
        figs.append({"cols": 3, "panels": [(_png(_panel(g, l, e, "scan", "axial", box=box), 3), base_name),
                                           (_png(_panel(g, l, e, "outline", "axial", box=box), 3), "Predicted outline"),
                                           (_png(_panel(g, l, e, "unc", "axial", box=box), 3), "Uncertainty")],
                     "caption": f"Axial slice {zu}, highest mean uncertainty inside the tumour. Recommended for first review."})
    return figs


_CSS = """
*{box-sizing:border-box} body{margin:0;background:#3a3c40;font-family:'Inter','Segoe UI',Arial,sans-serif;color:#111;font-size:13px}
.page{max-width:860px;margin:18px auto;background:#fff;padding:40px 48px 34px;box-shadow:0 10px 40px -12px #000}
.hd{display:flex;justify-content:space-between;align-items:flex-end;border-bottom:2px solid #111;padding-bottom:10px}
.org{font-family:'Space Grotesk','Inter',sans-serif;font-size:20px;font-weight:700;letter-spacing:-.01em}
.ttl{font-size:12px;color:#444;margin-top:2px;text-transform:uppercase;letter-spacing:.08em}
.rid{font-size:11px;color:#444;text-align:right;line-height:1.5}
.kv{display:grid;grid-template-columns:repeat(3,1fr);border-bottom:1px solid #bbb;padding:10px 0;gap:6px 18px}
.kv div{font-size:12px} .kv b{display:block;font-size:10px;font-weight:600;color:#555;text-transform:uppercase;letter-spacing:.06em;margin-bottom:1px}
.status{margin:14px 0 4px;border:1px solid #111;padding:8px 12px;font-size:12px;display:flex;justify-content:space-between}
.status b{letter-spacing:.04em}
h3{font-size:11.5px;font-weight:700;text-transform:uppercase;letter-spacing:.1em;margin:22px 0 6px;padding-bottom:3px;border-bottom:1px solid #ccc}
p{margin:4px 0;line-height:1.55}
ol{margin:4px 0;padding-left:20px;line-height:1.6} ol li{margin:2px 0}
table{width:100%;border-collapse:collapse;margin:4px 0;font-size:12px}
th{text-align:left;font-weight:600;font-size:11px;color:#333;border-bottom:1px solid #111;padding:5px 6px}
td{padding:5px 6px;border-bottom:1px solid #e3e3e3;vertical-align:top} .n{text-align:right;font-variant-numeric:tabular-nums}
.sev{font-weight:600;font-size:11px} .s0{color:#b00020} .s1{color:#8a6100} .s2{color:#555}
.figs{display:grid;gap:6px;margin-top:10px} .figs figure{margin:0} .figs img{width:100%;aspect-ratio:1/1;object-fit:contain;display:block;background:#000} .figs figcaption{font-size:10.5px;color:#444;margin-top:2px}
.cap{font-size:11px;color:#444;margin:4px 0 0}
.prof{display:flex;align-items:flex-end;gap:1px;height:56px;border-bottom:1px solid #111;margin-top:6px} .prof i{flex:1;background:#9a9a9a} .prof i.h{background:#111}
.sign{display:grid;grid-template-columns:2fr 1.4fr 1fr;gap:24px;margin-top:30px} .sign div{border-top:1px solid #111;padding-top:4px;font-size:11px;color:#444}
.pp .big{font-size:15px;line-height:1.6;margin:14px 0} .pp p{font-size:14px;line-height:1.65} .pp ol{font-size:14px}
.page+.page{margin-top:24px}
@media print{.page+.page{break-before:page;margin-top:0}}
.ft{margin-top:22px;font-size:10.5px;color:#555;line-height:1.5;border-top:1px solid #ccc;padding-top:8px}
@media print{body{background:#fff}.page{box-shadow:none;margin:0;max-width:none;padding:14mm 16mm}table,.figs,ol,.cap{break-inside:avoid}}
"""

_SEV = {"critical": ("High", 0), "warn": ("Moderate", 1), "info": ("Note", 2)}


def _fnum(v, d=2):
    return "-" if v is None or (isinstance(v, float) and v != v) else f"{v:.{d}f}"


def report_html(a: Dict, res: Dict, flagged: bool, thr: Optional[Dict], model_info: Dict, case_label: str, images: Dict[str, str],
                mc: int, device: str, infer_s: float, followup: Optional[Dict] = None) -> str:
    import hashlib
    e = html.escape
    now = time.localtime()
    rid = "NS-" + time.strftime("%Y%m%d", now) + "-" + hashlib.sha1(f"{case_label}{a['wt_ml']:.3f}{a['u']:.5f}".encode()).hexdigest()[:6].upper()
    q = res["quality"]
    mods = [m.upper() for m in res["present"]]
    miss = [m.upper() for m in ["t1", "t1ce", "t2", "flair"] if m not in res["present"]]
    loc = a.get("location") or {}
    R = a["regions"]
    th, u, pct = a.get("thr"), a["u"], a.get("u_percentile")
    ext = a.get("extent_mm")
    sr = a.get("slice_range")

    # findings narrative
    fnd = []
    if a["wt_ml"] > 0:
        where = f"{loc['side']}, {loc['ap']}, {loc['si']}" if loc else "location not determined"
        fnd.append(f"<p><b>Lesion.</b> {'A dominant' if a.get('components', 1) > 1 else 'A single'} segmented lesion is identified in the {e(where)} region"
                   + (f", spanning axial slices {sr[0]} to {sr[1]}" if sr else "") + ".</p>")
        if ext:
            fnd.append(f"<p><b>Size.</b> Maximum extent of the main lesion approximately {ext[0]:.0f} x {ext[1]:.0f} x {ext[2]:.0f} mm (left-right x anterior-posterior x superior-inferior). "
                       f"Whole tumour volume {a['wt_ml']:.1f} mL, {a['wt_pct_brain']:.1f}% of segmented brain volume.</p>")
        fnd.append(f"<p><b>Composition.</b> Enhancing tumour {R[3]['ml']:.1f} mL; necrotic or non-enhancing core {R[1]['ml']:.1f} mL; "
                   f"peritumoral edema {R[2]['ml']:.1f} mL. Tumour core (enhancing plus necrotic) {R[1]['ml'] + R[3]['ml']:.1f} mL.</p>")
        if a.get("components", 1) > 1:
            fnd.append(f"<p><b>Separate components.</b> {a['components'] - 1} additional component{'s' if a['components'] > 2 else ''} separate from the main lesion; see quality control.</p>")
    else:
        fnd.append("<p>No tumour voxels were predicted.</p>")
    if a.get("gt"):
        g = a["gt"]
        fnd.append(f"<p><b>Comparison with reference segmentation.</b> Dice per label: "
                   + "; ".join(f"{e(n.lower())} {_fnum(v, 3)}" for n, v in g["dice"].items())
                   + f". Under-segmented {g['fn_ml']:.1f} mL, over-segmented {g['fp_ml']:.1f} mL.</p>")

    # measurements
    mrows = ""
    for k in (3, 1, 2):
        mrows += f"<tr><td>{e(R[k]['name'])}</td><td class='n'>{R[k]['ml']:.2f}</td><td class='n'>{_fnum(R[k]['mean_u'])}</td></tr>"
    mrows += f"<tr><td><b>Whole tumour</b></td><td class='n'><b>{a['wt_ml']:.2f}</b></td><td class='n'></td></tr>"
    rano = a.get("rano")
    rtab = ""
    if rano:
        tgt = "enhancing tumour" if rano["target"] == "ET" else "tumour core (no enhancing tumour)"
        rtab = (f"<table><tr><th>Bidimensional measurement ({tgt}, axial slice {rano['slice']})</th><th class='n'>Value</th></tr>"
                f"<tr><td>Longest diameter</td><td class='n'>{rano['d1_mm']:.1f} mm</td></tr>"
                f"<tr><td>Longest perpendicular diameter</td><td class='n'>{rano['d2_mm']:.1f} mm</td></tr>"
                f"<tr><td>Product of perpendicular diameters</td><td class='n'>{rano['product_mm2']:.0f} mm&sup2;</td></tr>"
                f"<tr><td>Measurable disease (both diameters at least 10 mm)</td><td class='n'>{'Yes' if rano['measurable'] else 'No'}</td></tr></table>")

    # reliability
    rel = "<table><tr><th>Measure</th><th class='n'>Value</th></tr>"
    rel += f"<tr><td>Case uncertainty (mean predictive entropy in tumour region)</td><td class='n'>{u:.4f}</td></tr>"
    if th:
        rel += f"<tr><td>Review threshold (fixed on validation set)</td><td class='n'>{th:.4f}</td></tr>"
        rel += f"<tr><td>Ratio to threshold</td><td class='n'>{u/th:.2f}</td></tr>"
    if pct is not None:
        rel += f"<tr><td>Percentile among {a.get('n_test', 74)} held-out test cases</td><td class='n'>{pct:.0f}</td></tr>"
    if a.get("edge_u") is not None:
        rel += f"<tr><td>Mean uncertainty, tumour border / interior (0 to 1)</td><td class='n'>{a['edge_u']:.2f} / {a['inner_u']:.2f}</td></tr>"
    rel += "</table>"

    # QC table
    qrows = ""
    for f in a["findings"]:
        lab_, ci = _SEV[f["level"]]
        qrows += f"<tr><td class='sev s{ci}'>{lab_}</td><td>{e(f['title'])}</td><td>{e(f['where'] or '-')}</td><td>{e(f['detail'])}</td></tr>"
    if not qrows:
        qrows = "<tr><td colspan='4'>No flags raised.</td></tr>"
    bad = [c for c in q.checks if c.status != "pass"]

    # impression
    imp = []
    if a["wt_ml"] > 0:
        side = loc.get("side", "brain")
        imp.append(f"Segmented tumour in the {e(side)}, whole tumour volume {a['wt_ml']:.1f} mL"
                   + (f", maximum extent about {max(ext):.0f} mm" if ext else "") + ".")
        imp.append(f"Enhancing component {R[3]['ml']:.1f} mL with {R[1]['ml']:.1f} mL necrotic or non-enhancing core and {R[2]['ml']:.1f} mL surrounding edema.")
    else:
        imp.append("No tumour segmented.")
    if rano:
        imp.append(f"Bidimensional measurement {rano['d1_mm']:.0f} x {rano['d2_mm']:.0f} mm (product {rano['product_mm2']:.0f} mm&sup2;)"
                   + ("." if rano["measurable"] else ", below the RANO threshold for measurable disease."))
    if followup:
        imp.append(f"Compared with the {followup['other_label']}: {e(followup['category'])}. {e(followup['category_detail'])}")
    if th:
        imp.append(("Model uncertainty is above the validated review threshold. The segmentation should be reviewed and corrected before use."
                    if flagged else "Model uncertainty is within the validated threshold. Clinician confirmation is still required."))
    if miss:
        imp.append(f"Analysis performed without {', '.join(miss)}; accuracy for dependent regions is reduced.")
    n_hi = sum(1 for f in a["findings"] if f["level"] == "critical"); n_mod = sum(1 for f in a["findings"] if f["level"] == "warn")
    if n_hi or n_mod:
        imp.append(f"Quality control: {n_hi} high and {n_mod} moderate flag{'s' if n_hi + n_mod != 1 else ''} (section 5).")
    slices = (a.get("slices") or {}).get("most_uncertain") or []
    if slices:
        imp.append(f"Suggested review slices: {', '.join(str(z) for z in sorted(slices))}.")

    # figures
    figs, fno = "", 1
    for f in images.get("figs", []):
        figs += (f"<div class='figs' style='grid-template-columns:repeat({f['cols']},1fr)'>"
                 + "".join(f"<figure><img src='data:image/png;base64,{b}'><figcaption>{e(t)}</figcaption></figure>" for b, t in f["panels"])
                 + f"</div><p class='cap'><b>Fig. {fno}.</b> {e(f['caption'])}</p>")
        fno += 1
    sl = a.get("slices") or {}
    if sl.get("profile"):
        pr = np.array(sl["profile"]); mx = max(pr.max(), 1e-6); hot = set(sl.get("most_uncertain", []))
        figs += ("<div class='prof'>" + "".join(f"<i class='{'h' if i in hot else ''}' style='height:{100*v/mx:.0f}%'></i>" for i, v in enumerate(pr))
                 + f"</div><p class='cap'><b>Fig. {fno}.</b> Mean uncertainty inside the predicted tumour for each axial slice (inferior to superior). Dark bars: three highest.</p>")

    fu = ""
    if followup:
        rows = "".join(f"<tr><td>{e(r['name'])}</td><td class='n'>{r['earlier']}</td><td class='n'>{r['later']}</td><td class='n'>{r['change']}</td></tr>" for r in followup["rows"])
        fu = (f"<h3>Comparison with prior examination</h3>"
              + (f"<p class='cap'><b>Note.</b> {e(followup['note'])}</p>" if followup.get("note") else "")
              + f"<table><tr><th>Measure</th><th class='n'>Earlier</th><th class='n'>Later</th><th class='n'>Change</th></tr>{rows}</table>"
              + f"<p><b>Response assessment (imaging only):</b> {e(followup['category'])}. {e(followup['category_detail'])} "
              "Full RANO assessment also requires non-enhancing disease, corticosteroid dose and clinical status.</p>")
        if followup.get("panels"):
            fu += ("<div class='figs' style='grid-template-columns:1fr 1fr'>" + "".join(
                f"<figure><img src='data:image/png;base64,{b}'><figcaption>{e(t)}</figcaption></figure>" for b, t in followup["panels"]) + "</div>"
                "<p class='cap'>Bidimensional measurement on each examination (yellow: longest diameter, white: perpendicular).</p>")
    # patient page
    dia = (6 * a["wt_ml"] / np.pi) ** (1 / 3) if a["wt_ml"] > 0 else 0
    side_txt = {"right hemisphere": "the right side of the brain", "left hemisphere": "the left side of the brain",
                "crosses the midline": "both sides of the brain"}.get(loc.get("side"), "the brain")
    pp = f"""<div class="pp"><div class="hd"><div><div class="org">NeuroScope</div><div class="ttl">Summary for the patient</div></div>
<div class="rid">Report ID {rid}</div></div>
<p class="big">This page explains, in plain language, what the computer program found on the MRI scan. It is not a diagnosis. Your doctor will explain what it means for you.</p>
<h3>What was looked at</h3><p>{len(mods)} MRI image{'s' if len(mods) != 1 else ''} of the head ({', '.join(mods)}). Each type of MRI image shows tissue differently.
The program outlined areas that look different from healthy brain tissue.</p>
<h3>What the program found</h3>""" + (f"""<p>An abnormal area in {side_txt}. Its total size is about <b>{a['wt_ml']:.0f} millilitres</b>, roughly the volume of a ball
<b>{dia:.1f} cm</b> across. This includes swelling around it (called edema), which is usually the largest part.</p>
<p>The central part of the abnormal area, which doctors look at most closely, measures about <b>{R[1]['ml'] + R[3]['ml']:.0f} millilitres</b>.</p>""" if a["wt_ml"] > 0 else
"<p>The program did not find an abnormal area.</p>") + f"""
<h3>How sure is the program</h3><p>{'The program was <b>less sure than usual</b> about this scan, so a specialist needs to check the outline carefully before it is used.'
if flagged else 'The program was <b>about as sure as it usually is</b> on scans it handles well. A specialist still needs to check the result.'}</p>
""" + (f"<h3>Change since the earlier scan</h3><p>{e(followup['patient_text'])}</p>" if followup else "") + """
<h3>Questions you may want to ask your doctor</h3><ol><li>What do these findings mean for me?</li><li>Do I need more tests or scans?</li>
<li>What are my treatment options, and what are their benefits and risks?</li><li>When should the next scan be done?</li></ol>
<div class="ft">This summary was produced automatically by a research program and has not been reviewed unless signed on the main report. It does not replace advice from your doctor.</div></div>"""
    status = ("REVIEW REQUIRED", "Uncertainty above validated threshold") if flagged else ("WITHIN THRESHOLD", "Clinician confirmation required")
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{rid}</title>
<style>@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@700&display=swap');{_CSS}</style></head><body><div class="page">
<div class="hd"><div><div class="org">NeuroScope</div><div class="ttl">Automated MRI brain tumour segmentation report</div></div>
<div class="rid">Report ID {rid}<br>{time.strftime('%d %b %Y, %H:%M', now)}<br>{'Automated, edited by reviewer' if res.get('rejected') else 'Preliminary, automated'}</div></div>
<div class="kv"><div><b>Case</b>{e(case_label)}</div><div><b>Examination</b>MRI brain, multi-parametric</div><div><b>Sequences used</b>{', '.join(mods)}</div>
<div><b>Sequences missing</b>{', '.join(miss) or 'None'}</div><div><b>Input quality</b>{e(q.status)} ({q.score:.2f})</div><div><b>Processing</b>{mc} MC samples, {e(device.upper())}, {infer_s:.1f} s</div></div>
<div class="status"><b>{status[0]}</b><span>{status[1]}</span></div>
<h3>1. Technique</h3><p>Automated slice-wise segmentation with a 2D U-Net (7.8 M parameters) trained on BraTS 2020 with modality dropout. Labels: necrotic or non-enhancing
core, peritumoral edema, enhancing tumour. Voxel-wise uncertainty from Monte Carlo dropout ({mc} forward passes); case-level review threshold selected on the validation set and frozen.</p>
<h3>2. Findings</h3>{''.join(fnd)}
<h3>3. Measurements</h3><table><tr><th>Region</th><th class='n'>Volume (mL)</th><th class='n'>Mean uncertainty (0 to 1)</th></tr>{mrows}</table>{rtab}
<h3>4. Reliability</h3>{rel}
<h3>5. Quality control</h3><table><tr><th style="width:70px">Severity</th><th style="width:170px">Check</th><th style="width:110px">Location</th><th>Detail</th></tr>{qrows}</table>
<p class="cap">Input checks: {len(q.checks) - len(bad)} of {len(q.checks)} passed.</p>
<h3>6. Figures</h3>{figs}{fu}
<h3>7. Impression</h3><ol>{''.join(f'<li>{x}</li>' for x in imp)}</ol>
<div class="sign"><div>Reviewed by</div><div>Signature</div><div>Date</div></div>
<div class="ft">Generated by NeuroScope, a research prototype. Not a medical device and not for diagnostic use. Trained on adult glioma (BraTS 2020); not validated on other tumour types,
scanners, paediatric or post-operative imaging. Uncertainty reflects model disagreement and does not guarantee correctness. Report {rid}.</div>
</div><div class="page">{pp}</div></body></html>"""
