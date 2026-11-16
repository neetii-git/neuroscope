"""Structured case report: findings, fault analysis and a printable HTML document."""
from __future__ import annotations

import html
import time
from typing import Dict, List, Optional

import numpy as np
from scipy import ndimage as ndi

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
                "were the ones most likely to have a poor outline (whole-tumour Dice below 0.75).", "Whole case")
        elif ratio > 0.8:
            add("warn", "Uncertainty close to the review threshold", f"Case uncertainty {u:.4f} is {ratio*100:.0f}% of the threshold ({th:.4f}). Check the border slices.", "Whole case")

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
            " ".join(f"{c.modality.upper()}: {c.message.rstrip('.')}." for c in bad[:5]) + (" More in the input quality table." if len(bad) > 5 else ""),
            ", ".join(mods))

    # uncertain slices and regions
    if wt.any():
        per = np.array([ent[i][wt[i]].mean() / LOG2 if wt[i].any() else 0.0 for i in range(lab.shape[0])])
        area = wt.reshape(lab.shape[0], -1).sum(1)
        top = [int(i) for i in np.argsort(per)[::-1] if area[i] >= 30][:3]
        out["slices"] = {"most_uncertain": top, "largest": int(np.argmax(area)), "profile": per.tolist()}
        if top:
            add("info", "Most uncertain slices", f"Disagreement between repeated runs is highest on axial slices {', '.join(str(i) for i in sorted(top))}. Review these first.",
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
                add("info", "Uncertainty sits on the tumour border", f"Border voxels are {eu/max(iu,1e-9):.1f} times more uncertain than the interior. This is the usual pattern: the interior is "
                    "clear, the exact edge is a judgement call. Check the outline where it meets normal tissue.", "Tumour border")

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
                    f"{tot:.2f} mL in total sits apart from the main mass. Isolated fragments are the most common false positives; confirm or discard them.",
                    f"Near slices {', '.join(str(z) for z in sorted(set(zs)))}")
        out["components"] = int(n)

    # small structures
    for k in (1, 3):
        v = out["regions"][k]["ml"]
        if 0 < v < 0.5:
            add("info", f"Very small {REGION_NAMES[k].lower()} ({v:.2f} mL)", "Small structures are where segmentation is least reliable: a few voxels change the volume noticeably.", REGION_NAMES[k])
    if not wt.any():
        add("critical", "No tumour found", "The model predicted no tumour voxels. If a lesion is expected, check the scan orientation and sequences.", "Whole case")

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
            add("info", "Where the model disagrees with the expert outline",
                f"Missed by the model: {fn:.1f} mL. Predicted beyond the expert outline: {fp:.1f} mL. Weakest slices (whole-tumour Dice): "
                + ", ".join(f"{i} ({d:.2f})" for i, d in worst) + ".", f"Slices {', '.join(str(i) for i, _ in worst)}")
    order = {"critical": 0, "warn": 1, "info": 2}
    out["findings"].sort(key=lambda f: order[f["level"]])
    return out


_CSS = """
*{box-sizing:border-box} body{margin:0;background:#2a2c30;font-family:'Inter','Segoe UI',system-ui,sans-serif;color:#1b1c1e}
.page{max-width:900px;margin:18px auto;background:#fbfbf9;padding:44px 52px;box-shadow:0 20px 50px -20px #000;border-radius:6px}
.top{display:flex;justify-content:space-between;align-items:flex-start;border-bottom:3px solid #F5C518;padding-bottom:14px}
.brand{font-family:'Space Grotesk','Inter',sans-serif;font-size:26px;font-weight:700;letter-spacing:-.02em}
.sub{color:#6b6e73;font-size:12px;margin-top:3px} .meta{text-align:right;font-size:12px;color:#4a4d52;line-height:1.55}
h2{font-family:'Space Grotesk','Inter',sans-serif;font-size:15px;letter-spacing:.06em;text-transform:uppercase;margin:30px 0 10px;color:#1b1c1e;border-left:4px solid #F5C518;padding-left:10px}
.badge{display:inline-block;padding:8px 14px;border-radius:6px;font-weight:600;font-size:13px;margin:16px 0 4px}
.b-red{background:#2b1414;color:#ff9a9a;border:1px solid #7a2b2b} .b-ok{background:#16231a;color:#9be3ae;border:1px solid #2b6b3d}
.lead{font-size:15px;line-height:1.6;margin:10px 0}
table{width:100%;border-collapse:collapse;font-size:13px;margin:6px 0} th{text-align:left;color:#6b6e73;font-weight:500;border-bottom:1px solid #d8d9d6;padding:6px 8px}
td{padding:7px 8px;border-bottom:1px solid #ecece8} td.n{font-variant-numeric:tabular-nums;text-align:right} th.n{text-align:right}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:8px;vertical-align:middle}
.gauge{position:relative;height:14px;border-radius:7px;background:linear-gradient(90deg,#9be3ae 0%,#F5C518 55%,#ff6b6b 100%);margin:30px 6px 26px}
.mk{position:absolute;top:-8px;width:3px;height:30px;background:#1b1c1e} .mk span{position:absolute;top:32px;left:50%;transform:translateX(-50%);font-size:11px;white-space:nowrap;color:#33363a}
.mk.t{background:#6b6e73} .mk.t span{top:-20px}
.f{display:flex;gap:12px;padding:12px 14px;border:1px solid #e4e4e0;border-radius:8px;margin:8px 0;background:#fff}
.f .bar{width:5px;border-radius:3px;flex:none} .f.critical .bar{background:#d64545} .f.warn .bar{background:#F5C518} .f.info .bar{background:#9a9da3}
.f h4{margin:0 0 3px;font-size:14px} .f p{margin:0;font-size:13px;line-height:1.5;color:#3a3d42} .f .w{font-size:11px;color:#6b6e73;margin-top:5px;text-transform:uppercase;letter-spacing:.05em}
.imgs{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px} .imgs figure{margin:0} .imgs img{width:100%;border-radius:4px;display:block;background:#000}
figcaption{font-size:11px;color:#6b6e73;margin-top:4px}
.note{font-size:11.5px;color:#6b6e73;line-height:1.55;border-top:1px solid #d8d9d6;margin-top:30px;padding-top:12px}
.prof{display:flex;align-items:flex-end;gap:1px;height:70px;margin:8px 0 4px;border-bottom:1px solid #cfd0cc}
.prof i{flex:1;background:#b9bbc0;min-width:1px} .prof i.hot{background:#F5C518}
.pnt{position:sticky;top:0}
@media print{body{background:#fff}.page{box-shadow:none;margin:0;max-width:none;padding:18mm}.f,.imgs,table{break-inside:avoid}}
"""


def _fnum(v, d=2):
    return "-" if v is None or (isinstance(v, float) and v != v) else f"{v:.{d}f}"


def report_html(a: Dict, res: Dict, flagged: bool, thr: Optional[Dict], model_info: Dict, case_label: str, images: Dict[str, str],
                mc: int, device: str, infer_s: float) -> str:
    e = html.escape
    ts = time.strftime("%d %b %Y, %H:%M")
    q = res["quality"]
    loc = a.get("location") or {}
    where = (f"{loc['side']}, {loc['ap']}, {loc['si']}" if loc else "not determined")
    badge = ('<div class="badge b-red">REVIEW RECOMMENDED. Model uncertainty is above the validated threshold.</div>' if flagged
             else '<div class="badge b-ok">WITHIN VALIDATED THRESHOLD. Expert confirmation is still required.</div>')
    wt, tc, et = (a["regions"][k]["ml"] for k in (2, 1, 3))
    whole = a["wt_ml"]
    lead = (f"The model outlines an abnormal region of <b>{whole:.1f} mL</b> ({a['wt_pct_brain']:.1f}% of brain volume), located in the <b>{e(where)}</b>. "
            f"It contains <b>{a['regions'][1]['ml']:.1f} mL</b> necrotic or non-enhancing core, <b>{a['regions'][3]['ml']:.1f} mL</b> enhancing tumour and "
            f"<b>{a['regions'][2]['ml']:.1f} mL</b> surrounding edema.")
    # gauge
    u, th = a["u"], a.get("thr")
    gauge = ""
    if th:
        mx = max(u, th) * 1.6
        pu, pt = min(98, 100 * u / mx), min(98, 100 * th / mx)
        pct = a.get("u_percentile")
        gauge = (f'<div class="gauge"><div class="mk t" style="left:{pt:.1f}%"><span>threshold {th:.3f}</span></div>'
                 f'<div class="mk" style="left:{pu:.1f}%"><span>this case {u:.3f}</span></div></div>'
                 + (f'<div class="sub">Higher than {pct:.0f}% of the {a.get("n_test", 74)} held-out test patients.</div>' if pct is not None else ""))
    # volumes table
    rows = ""
    cols = {1: "#ff3333", 2: "#33cc33", 3: "#ffd91a"}
    for k in (1, 3, 2):
        r = a["regions"][k]
        rows += (f'<tr><td><span class="dot" style="background:{cols[k]}"></span>{e(r["name"])}</td><td class="n">{r["ml"]:.2f}</td>'
                 f'<td class="n">{_fnum(r["mean_u"])}</td></tr>')
    rows += f'<tr><td><b>Whole tumour (all labels)</b></td><td class="n"><b>{whole:.2f}</b></td><td class="n">-</td></tr>'
    # findings
    fh = ""
    for f in a["findings"]:
        fh += (f'<div class="f {f["level"]}"><div class="bar"></div><div><h4>{e(f["title"])}</h4><p>{e(f["detail"])}</p>'
               + (f'<div class="w">Where: {e(f["where"])}</div>' if f["where"] else "") + "</div></div>")
    if not fh:
        fh = '<div class="f info"><div class="bar"></div><div><h4>No issues detected</h4><p>No automated check raised a concern for this case.</p></div></div>'
    # images
    ih = ""
    for key, cap in (("largest", "Largest tumour slice"), ("uncertain", "Most uncertain slice")):
        if images.get(key):
            ih += f'<h2 style="margin-top:24px">{cap}</h2><div class="imgs">' + "".join(
                f'<figure><img src="data:image/png;base64,{images[key][n]}"><figcaption>{c}</figcaption></figure>'
                for n, c in (("img", "Scan"), ("pred", "Predicted outline"), ("unc", "Uncertainty (bright = unsure)"))) + "</div>"
    # profile
    prof = ""
    sl = a.get("slices") or {}
    if sl.get("profile"):
        pr = np.array(sl["profile"]); mxv = max(pr.max(), 1e-6); hot = set(sl.get("most_uncertain", []))
        prof = ('<h2>Uncertainty across slices</h2><div class="prof">' + "".join(
            f'<i class="{"hot" if i in hot else ""}" style="height:{100*v/mxv:.0f}%" title="slice {i}"></i>' for i, v in enumerate(pr)) +
            '</div><div class="sub">Mean uncertainty inside the predicted tumour per axial slice. Yellow bars are the three most uncertain.</div>')
    # gt
    gth = ""
    if a.get("gt"):
        g = a["gt"]
        gth = ('<h2>Agreement with the expert outline</h2><table><tr><th>Region</th><th class="n">Dice</th></tr>'
               + "".join(f'<tr><td>{e(n)}</td><td class="n">{_fnum(v, 3)}</td></tr>' for n, v in g["dice"].items())
               + f'</table><div class="sub">Missed: {g["fn_ml"]:.1f} mL. Over-predicted: {g["fp_ml"]:.1f} mL.</div>')
    bad = [c for c in q.checks if c.status != "pass"]
    qrows = "".join(f'<tr><td>{e(c.modality.upper())}</td><td>{e(c.status)}</td><td>{e(c.message)}</td></tr>' for c in bad) or \
        '<tr><td colspan="3">All checks passed.</td></tr>'
    qrows += f'<tr><td colspan="3" class="sub">{len(q.checks) - len(bad)} of {len(q.checks)} automated checks passed.</td></tr>'
    mods = ", ".join(m.upper() for m in res["present"]); miss = [m.upper() for m in ["t1", "t1ce", "t2", "flair"] if m not in res["present"]]
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>NeuroScope report</title>
<style>@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@600;700&display=swap');{_CSS}</style></head>
<body><div class="page">
<div class="top"><div><div class="brand">NeuroScope</div><div class="sub">Brain tumour segmentation report with uncertainty and fault analysis</div></div>
<div class="meta">Case: <b>{e(case_label)}</b><br>Generated {ts}<br>Sequences: {e(mods)}{(' (missing ' + ', '.join(miss) + ')') if miss else ''}</div></div>
{badge}
<h2>Summary</h2><p class="lead">{lead}</p>
<h2>Reliability</h2>{gauge}
<div class="sub">Uncertainty is the average disagreement between {mc} repeated runs of the network inside the tumour region. The threshold was fixed on the validation set and never tuned on test data.</div>
<h2>Findings and faults</h2>{fh}
<h2>Volumes</h2><table><tr><th>Region</th><th class="n">Volume (mL)</th><th class="n">Mean uncertainty (0 to 1)</th></tr>{rows}</table>
{prof}{ih}{gth}
<h2>Input quality</h2><table><tr><th>Sequence</th><th>Status</th><th>Detail</th></tr>{qrows}</table><div class="sub">Quality score {q.score:.2f} ({e(q.status)}).</div>
<div class="note"><b>Method.</b> 2D U-Net (7.8M parameters) trained on BraTS 2020 with modality dropout; uncertainty from Monte Carlo dropout ({mc} samples, {e(device)}, {infer_s:.1f} s).
<br><b>Limitations.</b> Trained on adult glioma only (skull-stripped, co-registered). Not validated on other tumour types, scanners, children or post-operative scans. Uncertainty measures model
disagreement, not correctness. Automated checks are heuristics.
<br><b>Research prototype. Not a medical device. Every result must be reviewed by a qualified clinician.</b></div>
</div></body></html>"""
