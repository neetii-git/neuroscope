"""NeuroScope - brain tumour segmentation from MRI with uncertainty estimation (research prototype).

Run:  python -m streamlit run app/app.py
Optional: -- --ckpt runs/base/best.pt --results runs/base/results --demo-dir demo_case
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import tempfile
import time
from typing import Dict, Optional

import nibabel as nib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from neuroscope.data.brats import MODALITIES, labels_to_regions, load_nifti_with_affine, remap_labels  # noqa: E402
from neuroscope.data.preprocess import preprocess_case, uncrop  # noqa: E402
from neuroscope.eval.metrics import region_metrics  # noqa: E402
from neuroscope.inference.predictor import load_checkpoint, predict_volume  # noqa: E402
from neuroscope.quality.iqa import assess_case  # noqa: E402
import streamlit.components.v1 as components  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from viz3d import brain_html
from report import analyse, build_figures, gray_volume, rano_panel, report_html  # noqa: E402
from clinical import apply_rejections, bidimensional, response_category, simulate_prior
from clinical import components as find_components  # noqa: E402

try:
    from skimage.measure import marching_cubes
    HAVE_SKIMAGE = True
except Exception:  # pragma: no cover
    HAVE_SKIMAGE = False

# ------------------------------------------------------------------------------------------------
LABEL_RGB = {1: (255, 51, 51), 2: (51, 204, 51), 3: (255, 217, 26)}          # core red, edema green, enhancing yellow
LABEL_NAMES = {0: "background", 1: "necrotic / non-enhancing core", 2: "edema", 3: "enhancing tumour"}
REGION_NAMES = ["Whole tumour", "Tumour core", "Enhancing tumour"]
REGION_KEYS = ["WT_ml", "TC_ml", "ET_ml"]
REGION_HEX = ["#2ca02c", "#d62728", "#ffbf00"]
SCENE_BG = "#0B0C0E"
MOD_LABEL = {"t1": "T1", "t1ce": "T1ce (contrast)", "t2": "T2", "flair": "FLAIR"}
LOG2 = float(np.log(2))
PLOT_BG = "#17181B"
ACCENT = "#F5C518"   # yellow accent, used sparingly
GREEN = "#B8BABF"   # neutral grey for bars


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.environ.get("NEUROSCOPE_CKPT", os.path.join(ROOT, "runs", "base", "best.pt")))
    ap.add_argument("--results", default=os.environ.get("NEUROSCOPE_RESULTS", os.path.join(ROOT, "runs", "base", "results")))
    ap.add_argument("--demo-dir", default=os.environ.get("NEUROSCOPE_DEMO", os.path.join(ROOT, "demo_case")))
    args, _ = ap.parse_known_args()
    return args


ARGS = parse_args()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    torch.set_num_threads(max(1, os.cpu_count() or 1))

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;600;700&display=swap');
html, body, [class*="css"], .stApp, button, input, textarea {font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;}
h1, h2, h3, h4, div[data-testid="stMetricValue"] {font-family: 'Space Grotesk', 'Inter', 'Segoe UI', sans-serif !important; letter-spacing: -0.015em;}
#MainMenu, footer {visibility: hidden;}
html, body {background: #0E0F11;}
.stApp {background: transparent; color: #ECECEA;}
header[data-testid="stHeader"], [data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="stMainBlockContainer"], [data-testid="stBottom"] > div {background: transparent !important;}
#ns-bg {z-index: 1 !important;}
.stApp > header {z-index: 3;}
.block-container, section[data-testid="stSidebar"] {position: relative; z-index: 2;}
[data-testid="stAppViewContainer"] {position: relative; z-index: 2;}
@keyframes rise {from {opacity: 0; transform: perspective(900px) translateY(28px) rotateX(8deg);} to {opacity: 1; transform: none;}}
.block-container > div > div[data-testid="stVerticalBlock"] > div {animation: rise .7s cubic-bezier(.2,.7,.2,1) both;}
.block-container > div > div[data-testid="stVerticalBlock"] > div:nth-child(2) {animation-delay: .08s;}
.block-container > div > div[data-testid="stVerticalBlock"] > div:nth-child(3) {animation-delay: .16s;}
.block-container > div > div[data-testid="stVerticalBlock"] > div:nth-child(4) {animation-delay: .24s;}
.block-container {padding-top: 1.2rem; max-width: 1280px;}
section[data-testid="stSidebar"] {background: #141517ee; backdrop-filter: blur(10px); box-shadow: 6px 0 24px -14px #000;}
h1, h2, h3, h4 {color: #F2F2F0;}
div[data-testid="stMetric"] {background: linear-gradient(150deg, #1D1F23d9, #141517b3); backdrop-filter: blur(8px); border: 1px solid #2C2E33; border-radius: 14px; padding: 12px 16px;
  box-shadow: 0 1px 0 #ffffff0d inset, 0 12px 22px -14px #000, 0 2px 4px #0006; transform-style: preserve-3d; transition: transform .18s ease-out, box-shadow .18s ease-out, border-color .18s; will-change: transform;}
div[data-testid="stMetric"]:hover {border-color: #F5C51877; box-shadow: 0 18px 30px -14px #000, 0 0 0 1px #F5C51833;}
div[data-testid="stMetricValue"] {color: #F5C518; transform: translateZ(24px);}
div[data-testid="stMetricLabel"] {color: #A3A5AA;}
div[data-testid="stTabs"] button[role="tab"] {border-radius: 10px 10px 0 0; transition: background .15s, transform .15s; color: #A3A5AA;}
div[data-testid="stTabs"] button[role="tab"]:hover {background: #F5C5180F; transform: translateY(-2px); color: #ECECEA;}
div[data-testid="stTabs"] button[aria-selected="true"] {color: #F5C518; background: #F5C51810;}
.stButton > button, .stDownloadButton > button {background: #1D1F23cc; color: #ECECEA; border: 1px solid #34363B; border-radius: 12px; transition: transform .15s, box-shadow .15s, border-color .15s; box-shadow: 0 8px 16px -10px #000;}
.stButton > button:hover, .stDownloadButton > button:hover {border-color: #F5C518; color: #F5C518; transform: translateY(-2px) perspective(400px) rotateX(4deg); box-shadow: 0 14px 22px -12px #000;}
.stButton > button[kind="primary"] {background: #F5C518; color: #0E0F11; border-color: #F5C518; font-weight: 600;}
.stButton > button[kind="primary"]:hover {background: #FFD84A; color: #0E0F11;}
div[data-testid="stPlotlyChart"], div[data-testid="stDataFrame"], iframe {border-radius: 14px;}
div[data-testid="stPlotlyChart"] {box-shadow: 0 14px 28px -16px #000; border: 1px solid #25272B;}
div[data-testid="stAlert"] {border-radius: 12px; backdrop-filter: blur(6px);}
div[data-testid="stExpander"], div[data-testid="stTable"], div[data-testid="stDataFrame"] {background: #141517a6; backdrop-filter: blur(6px); border-radius: 14px;}
@media (prefers-reduced-motion: reduce) {* {transition: none !important;}}
</style>
"""

TILT_JS = """
<script>
try {
  const d = window.parent.document;
  if (!d.__nsTilt) {
    d.__nsTilt = true;
    d.addEventListener('mousemove', e => {
      const card = e.target.closest && e.target.closest('div[data-testid="stMetric"]');
      d.querySelectorAll('div[data-testid="stMetric"]').forEach(c => { if (c !== card) c.style.transform = ''; });
      if (!card) return;
      const r = card.getBoundingClientRect(), x = (e.clientX - r.left) / r.width - .5, y = (e.clientY - r.top) / r.height - .5;
      card.style.transform = 'perspective(700px) rotateY(' + (x * 14).toFixed(1) + 'deg) rotateX(' + (-y * 14).toFixed(1) + 'deg) translateY(-3px)';
    }, {passive: true});
    d.addEventListener('mouseleave', () => d.querySelectorAll('div[data-testid="stMetric"]').forEach(c => c.style.transform = ''));
  }
} catch (err) {}
</script>
"""

def _bg_script() -> str:
    vend = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")
    with open(os.path.join(vend, "three.min.js"), encoding="utf-8") as f1, open(os.path.join(vend, "bg.js"), encoding="utf-8") as f2:
        js = f1.read() + "\n" + f2.read()
    js = js.replace("</script>", "<\\/script>")
    return ("<script>try{var d=window.parent.document;if(!d.getElementById('ns-bg-src')){var s=d.createElement('script');s.id='ns-bg-src';"
            "s.textContent=" + json.dumps(js) + ";d.head.appendChild(s);}}catch(e){}</script>")



def dark(fig: go.Figure, h: int = 420) -> go.Figure:
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(23,24,27,0.78)", plot_bgcolor="rgba(0,0,0,0)", height=h,
                      margin=dict(l=10, r=10, t=36, b=10), font=dict(color="#ECECEA", family="Inter, Segoe UI, sans-serif"))
    fig.update_xaxes(gridcolor="#2C2E33", zerolinecolor="#2C2E33"); fig.update_yaxes(gridcolor="#2C2E33", zerolinecolor="#2C2E33")
    return fig


# ------------------------------------------------------------------------------------------------
# Cached loaders
# ------------------------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_model(ckpt_path: str, device: str):
    model = load_checkpoint(ckpt_path, device)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return model, {"epoch": ckpt.get("epoch"), "val": ckpt.get("val")}


@st.cache_data(show_spinner=False)
def load_json(path: str) -> Optional[Dict]:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


@st.cache_data(show_spinner=False)
def load_csv(path: str) -> Optional[pd.DataFrame]:
    return pd.read_csv(path) if os.path.exists(path) else None


@st.cache_resource(show_spinner=False)
def load_demo_case(demo_dir: str):
    vols, affine = {}, np.eye(4)
    for m in MODALITIES:
        hits = glob.glob(os.path.join(demo_dir, f"*_{m}.nii*"))
        if hits:
            vols[m], affine, _ = load_nifti_with_affine(hits[0])
    seg_hits = glob.glob(os.path.join(demo_dir, "*seg*.nii*")) + glob.glob(os.path.join(demo_dir, "*Segm*.nii*"))
    if seg_hits:
        vols["seg"] = remap_labels(load_nifti_with_affine(seg_hits[0])[0].astype(np.int16))
    return vols, affine


@st.cache_resource(show_spinner=False)
def run_demo_inference(demo_dir: str, mc: int, ckpt_path: str, drop: tuple):
    vols, affine = load_demo_case(demo_dir)
    vols = {k: v for k, v in vols.items() if k not in drop}
    return run_case(vols, affine, mc, ckpt_path)


def run_case(vols, affine, mc: int, ckpt_path: str):
    model, _ = get_model(ckpt_path, DEVICE)
    present = [m for m in MODALITIES if m in vols]
    quality = assess_case({m: vols.get(m) for m in MODALITIES})
    img, seg, meta = preprocess_case(vols, available=present)
    t0 = time.time()
    voxel_mm3 = float(abs(np.linalg.det(affine[:3, :3]))) or 1.0
    pred = predict_volume(model, img, mc_samples=mc, device=DEVICE, batch_size=32, voxel_volume_mm3=voxel_mm3)
    infer_s = time.time() - t0
    metrics = region_metrics(pred.regions, labels_to_regions(seg), with_hd95=False) if seg is not None else None
    brain = (np.abs(img).sum(0) > 0)
    return dict(img=img, seg=seg, meta=meta, pred=pred, infer_s=infer_s, present=present, quality=quality,
                metrics=metrics, affine=affine, brain=brain, voxel_ml=voxel_mm3 / 1000.0, meshes=None, token=time.time())


@st.cache_resource(show_spinner=False)
def run_simulated_prior(demo_dir: str, mc: int, ckpt_path: str):
    vols, affine = load_demo_case(demo_dir)
    return run_case(simulate_prior(vols), affine, mc, ckpt_path)


def fig_rano(res, meas, mod_index: int):
    img, pred = res["img"], res["pred"]
    z = meas["slice"]
    rgb = compose_rgb(img[mod_index][z], pred.labels[z], pred.entropy[z], True, False, 0.4, 0)
    W = img.shape[3]
    fig = go.Figure(go.Image(z=rgb, hoverinfo="skip"))
    tr = lambda p: (p[0], W - 1 - p[1])
    for (a, b, col, name, d) in ((meas["p1"], meas["p2"], ACCENT, "Longest diameter", meas["d1_mm"]),
                                 (meas["q1"], meas["q2"], "#FFFFFF", "Perpendicular", meas["d2_mm"])):
        (x0, y0), (x1, y1) = tr(a), tr(b)
        fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines+markers", line=dict(color=col, width=3), marker=dict(size=7, color=col),
                                 name=f"{name} {d:.1f} mm", hovertemplate=f"{name}: {d:.1f} mm<extra></extra>"))
    cx, cy = tr(((meas["p1"][0] + meas["p2"][0]) / 2, (meas["p1"][1] + meas["p2"][1]) / 2))
    h = max(35, meas["d1_mm"] * 1.0)
    fig.update_layout(xaxis=dict(visible=False, range=[cx - h, cx + h]), yaxis=dict(visible=False, range=[cy + h, cy - h], scaleanchor="x"),
                      legend=dict(orientation="h", y=-0.04), dragmode="pan")
    return dark(fig, 480)


def build_followup(cur, other, other_is_earlier: bool, simulated: bool):
    """Compare two analysed examinations. Returns a dict used by the app and the report."""
    earlier, later = (other, cur) if other_is_earlier else (cur, other)
    def meas(r):
        return bidimensional(r["pred"].labels, r["affine"], "ET")
    me, ml_ = meas(earlier), meas(later)
    ve, vl = earlier["pred"].volumes_ml, later["pred"].volumes_ml
    def ed(r): return float(((r["pred"].labels == 2).sum()) * r["voxel_ml"])
    rows = []
    for name, a, b in (("Whole tumour (mL)", ve["WT_ml"], vl["WT_ml"]), ("Tumour core (mL)", ve["TC_ml"], vl["TC_ml"]),
                       ("Enhancing tumour (mL)", ve["ET_ml"], vl["ET_ml"]), ("Edema (mL)", ed(earlier), ed(later))):
        ch = (b - a) / a * 100 if a > 0.05 else float("nan")
        rows.append({"name": name, "earlier": f"{a:.1f}", "later": f"{b:.1f}", "change": f"{b - a:+.1f} ({ch:+.0f}%)" if ch == ch else f"{b - a:+.1f}", "a": a, "b": b})
    pa, pb = (me or {}).get("product_mm2"), (ml_ or {}).get("product_mm2")
    if pa or pb:
        ch = (pb - pa) / pa * 100 if (pa and pb) else float("nan")
        rows.append({"name": "Product of perpendicular diameters (mm2)", "earlier": f"{pa:.0f}" if pa else "-", "later": f"{pb:.0f}" if pb else "-",
                     "change": f"{ch:+.0f}%" if ch == ch else "-", "a": pa or 0, "b": pb or 0})
    cat, detail = response_category(me, ml_, ve["ET_ml"], vl["ET_ml"])
    dv = vl["WT_ml"] - ve["WT_ml"]
    trend = "smaller" if dv < -0.05 * max(ve["WT_ml"], 1e-6) else ("larger" if dv > 0.05 * max(ve["WT_ml"], 1e-6) else "about the same size")
    panels = []
    for r, m, t in ((earlier, me, "Earlier"), (later, ml_, "Later")):
        if m:
            bi = MODALITIES.index("flair") if "flair" in r["present"] else MODALITIES.index(r["present"][0])
            g = gray_volume(r["img"], bi)
            panels.append((rano_panel(g[m["slice"]], r["pred"].labels[m["slice"]], m), f"{t}: {m['d1_mm']:.0f} x {m['d2_mm']:.0f} mm, slice {m['slice']}"))
    return {"rows": rows, "category": cat, "category_detail": detail, "earlier": earlier, "later": later, "m_earlier": me, "m_later": ml_,
            "other_label": "earlier examination" if other_is_earlier else "later examination", "panels": panels,
            "note": "The earlier examination is SIMULATED for demonstration: the example scan with the tumour region digitally shrunk. "
                    "BraTS 2020 has one time point per patient." if simulated else "",
            "patient_text": f"Compared with the earlier scan, the abnormal area is {trend} ({ve['WT_ml']:.0f} mL then, {vl['WT_ml']:.0f} mL now)."
                            + (" The earlier scan in this example is simulated for demonstration." if simulated else "")}


def read_upload(upload):
    if upload is None:
        return None
    suffix = ".nii.gz" if upload.name.endswith(".gz") else ".nii"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(upload.getbuffer())
        path = tmp.name
    try:
        return load_nifti_with_affine(path)
    finally:
        os.unlink(path)


# ------------------------------------------------------------------------------------------------
# 2D viewer
# ------------------------------------------------------------------------------------------------
def take_slice(vol: np.ndarray, axis: int, i: int) -> np.ndarray:
    return vol[i] if axis == 0 else (vol[:, i, :] if axis == 1 else vol[:, :, i])


def contour(mask: np.ndarray) -> np.ndarray:
    m = mask > 0
    p = np.pad(m, 1)
    inner = p[2:, 1:-1] & p[:-2, 1:-1] & p[1:-1, 2:] & p[1:-1, :-2]
    return m & ~inner


def compose_rgb(base, labels, ent, show_seg, show_unc, seg_alpha, unc_alpha, gt=None):
    nz = base[base != 0]
    lo, hi = (np.percentile(nz, [1, 99]) if nz.size else (0.0, 1.0))
    g = np.clip((base - lo) / (hi - lo + 1e-6), 0, 1)
    rgb = np.stack([g, g, g], -1)
    if show_unc:
        e = np.clip(ent / LOG2, 0, 1)
        w = e[..., None] * unc_alpha
        rgb = rgb * (1 - w) + np.array([0.98, 0.96, 0.88]) * w   # white glow for uncertainty
    if show_seg:
        for lab, col in LABEL_RGB.items():
            m = labels == lab
            rgb[m] = rgb[m] * (1 - seg_alpha) + np.array(col) / 255.0 * seg_alpha
    if gt is not None:
        rgb[contour(gt)] = (0.2, 0.9, 1.0)   # cyan contour
    return (np.transpose(rgb, (1, 0, 2))[::-1] * 255).astype(np.uint8)


def slice_figure(base, labels, ent, show_seg, show_unc, seg_alpha, unc_alpha, gt, title):
    img = compose_rgb(base, labels, ent, show_seg, show_unc, seg_alpha, unc_alpha, gt)
    names = np.vectorize(LABEL_NAMES.get)(labels)
    custom = np.stack([names.T[::-1], np.round(ent, 3).T[::-1], np.round(base, 2).T[::-1]], -1)
    fig = go.Figure(go.Image(z=img, customdata=custom,
                             hovertemplate="tissue: %{customdata[0]}<br>uncertainty: %{customdata[1]}<br>intensity (z): %{customdata[2]}<extra></extra>"))
    fig.update_layout(title=dict(text=title, x=0.02, font=dict(size=14)), xaxis=dict(visible=False),
                      yaxis=dict(visible=False, scaleanchor="x"), dragmode="pan")
    return dark(fig, 520)


# ------------------------------------------------------------------------------------------------
# 3D viewer
# ------------------------------------------------------------------------------------------------
def mesh(mask: np.ndarray, step: int):
    v = mask[::step, ::step, ::step].astype(np.float32)
    if v.sum() < 8:
        return None
    v = np.pad(v, 1)
    verts, faces, _, _ = marching_cubes(v, level=0.5, step_size=1)
    return (verts - 1) * step, faces


def build_meshes(res):
    out = {"brain": None, "regions": [None] * 3}
    if not HAVE_SKIMAGE:
        return out
    out["brain"] = mesh(res["brain"], 3)
    for i in range(3):
        out["regions"][i] = mesh(res["pred"].regions[i], 2)
    return out


def fig_3d(res, z_axial: int, show: Dict[str, bool], opacity: float, show_plane: bool, base_vol: np.ndarray):
    pred, meshes = res["pred"], res["meshes"]
    fig = go.Figure()
    if HAVE_SKIMAGE and meshes is not None:
        if show.get("brain") and meshes["brain"] is not None:
            v, f = meshes["brain"]
            fig.add_trace(go.Mesh3d(x=v[:, 2], y=v[:, 1], z=v[:, 0], i=f[:, 0], j=f[:, 1], k=f[:, 2], color="#94A3B8", opacity=0.07,
                                    name="Brain surface", showlegend=True, hoverinfo="skip", lighting=dict(ambient=0.6)))
        for i, (nm, col) in enumerate(zip(REGION_NAMES, REGION_HEX)):
            if show.get(nm) and meshes["regions"][i] is not None:
                v, f = meshes["regions"][i]
                fig.add_trace(go.Mesh3d(x=v[:, 2], y=v[:, 1], z=v[:, 0], i=f[:, 0], j=f[:, 1], k=f[:, 2], color=col,
                                        opacity=(opacity * 0.4) if i == 0 else opacity * 0.7,
                                        name=f"{nm} ({pred.volumes_ml[REGION_KEYS[i]]:.1f} mL)", showlegend=True,
                                        hovertemplate=nm + "<extra></extra>",
                                        lighting=dict(ambient=0.45, diffuse=0.8, specular=0.25, roughness=0.6)))
    else:
        for i, (nm, col) in enumerate(zip(REGION_NAMES, REGION_HEX)):
            if not show.get(nm):
                continue
            v = pred.regions[i][::2, ::2, ::2].astype(np.float32)
            if v.sum() == 0:
                continue
            D, H, W = v.shape
            zz, yy, xx = np.mgrid[0:D, 0:H, 0:W] * 2
            fig.add_trace(go.Isosurface(x=xx.flatten(), y=yy.flatten(), z=zz.flatten(), value=v.flatten(), isomin=0.5, isomax=1,
                                        surface_count=1, opacity=opacity * (0.4 if i == 0 else 0.7), colorscale=[[0, col], [1, col]],
                                        showscale=False, name=nm, caps=dict(x_show=False, y_show=False, z_show=False)))
    if show_plane:
        sl = base_vol[z_axial]
        nz = sl[sl != 0]
        lo, hi = (np.percentile(nz, [1, 99]) if nz.size else (0, 1))
        H, W = sl.shape
        fig.add_trace(go.Surface(x=np.arange(W), y=np.arange(H), z=np.full((H, W), float(z_axial)),
                                 surfacecolor=np.clip((sl - lo) / (hi - lo + 1e-6), 0, 1), colorscale="Gray", cmin=0, cmax=1,
                                 showscale=False, opacity=0.95, name=f"Axial slice {z_axial}", hoverinfo="skip", showlegend=True))
    D = pred.regions.shape[1]
    fig.update_layout(scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False, range=[0, D]),
                                 aspectmode="data", bgcolor=SCENE_BG, camera=dict(eye=dict(x=1.6, y=1.2, z=0.9))),
                      legend=dict(orientation="h", y=0.0, x=0.5, xanchor="center", bgcolor="rgba(0,0,0,0)", font=dict(color="#E5E7EB")))
    dark(fig, 640)
    fig.update_layout(paper_bgcolor=SCENE_BG)
    return fig



def _norm(sl):
    nz = sl[sl != 0]
    lo, hi = (np.percentile(nz, [1, 99]) if nz.size else (0, 1))
    return np.clip((sl - lo) / (hi - lo + 1e-6), 0, 1)


@st.cache_resource(show_spinner=False)
def fig_hero(demo_dir: str):
    """Landing view: the example patient's MRI as three orthogonal planes inside a translucent brain. No inference."""
    vols, _ = load_demo_case(demo_dir)
    key = "flair" if "flair" in vols else next(k for k in vols if k != "seg")
    v = np.asarray(vols[key], dtype=np.float32)
    v = np.transpose(v, (2, 1, 0))  # (z, y, x)
    mask = v > 0
    zs = np.where(mask.any((1, 2)))[0]; ys = np.where(mask.any((0, 2)))[0]; xs = np.where(mask.any((0, 1)))[0]
    v = v[zs[0]:zs[-1] + 1, ys[0]:ys[-1] + 1, xs[0]:xs[-1] + 1]; mask = v > 0
    D, H, W = v.shape
    fig = go.Figure()
    if HAVE_SKIMAGE:
        m = mesh(mask, 3)
        if m is not None:
            vv, ff = m
            fig.add_trace(go.Mesh3d(x=vv[:, 2], y=vv[:, 1], z=vv[:, 0], i=ff[:, 0], j=ff[:, 1], k=ff[:, 2], color="#7DD3FC", opacity=0.08, hoverinfo="skip", lighting=dict(ambient=0.7)))
    else:
        s3 = mask[::4, ::4, ::4].astype(np.float32); d, h, w = s3.shape
        zz, yy, xx = np.mgrid[0:d, 0:h, 0:w] * 4
        fig.add_trace(go.Isosurface(x=xx.flatten(), y=yy.flatten(), z=zz.flatten(), value=s3.flatten(), isomin=0.5, isomax=1, surface_count=1,
                                    opacity=0.08, colorscale=[[0, "#7DD3FC"], [1, "#7DD3FC"]], showscale=False, hoverinfo="skip", caps=dict(x_show=False, y_show=False, z_show=False)))
    z0, y0, x0 = D // 2, H // 2, W // 2
    gx, gy = np.meshgrid(np.arange(W), np.arange(H))
    fig.add_trace(go.Surface(x=gx, y=gy, z=np.full((H, W), float(z0)), surfacecolor=_norm(v[z0]), colorscale="Gray", cmin=0, cmax=1, showscale=False, hoverinfo="skip"))
    gx, gz = np.meshgrid(np.arange(W), np.arange(D))
    fig.add_trace(go.Surface(x=gx, y=np.full((D, W), float(y0)), z=gz, surfacecolor=_norm(v[:, y0, :]), colorscale="Gray", cmin=0, cmax=1, showscale=False, hoverinfo="skip"))
    gy, gz = np.meshgrid(np.arange(H), np.arange(D))
    fig.add_trace(go.Surface(x=np.full((D, H), float(x0)), y=gy, z=gz, surfacecolor=_norm(v[:, :, x0]), colorscale="Gray", cmin=0, cmax=1, showscale=False, hoverinfo="skip"))
    fig.update_layout(height=520, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor=SCENE_BG, showlegend=False,
                      scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), aspectmode="data",
                                 bgcolor=SCENE_BG, camera=dict(eye=dict(x=1.5, y=-1.4, z=0.8))))
    return fig

# ------------------------------------------------------------------------------------------------
# Analysis charts
# ------------------------------------------------------------------------------------------------
def fig_slice_profile(res, z_cur: int):
    pred, brain = res["pred"], res["brain"]
    D = pred.regions.shape[1]
    area = pred.regions[0].reshape(D, -1).sum(1) * res["voxel_ml"]
    ent = np.array([pred.entropy[i][brain[i]].mean() if brain[i].any() else 0.0 for i in range(D)])
    fig = go.Figure()
    fig.add_trace(go.Bar(x=np.arange(D), y=area, name="Tumour volume per slice (mL)", marker_color=GREEN, opacity=0.55))
    fig.add_trace(go.Scatter(x=np.arange(D), y=ent, name="Mean uncertainty in brain", yaxis="y2", line=dict(color=ACCENT, width=2)))
    fig.add_vline(x=z_cur, line=dict(color="#ECECEA", dash="dot"), annotation_text=f"slice {z_cur}", annotation_position="top")
    fig.update_layout(title=dict(text="Per-slice profile", x=0.02, font=dict(size=14)), xaxis_title="axial slice",
                      yaxis=dict(title="mL"), yaxis2=dict(title="entropy", overlaying="y", side="right", showgrid=False),
                      legend=dict(orientation="h", y=1.12), barmode="overlay")
    return dark(fig, 340)


def fig_volumes(pred):
    vals = [pred.volumes_ml[k] for k in REGION_KEYS]
    fig = go.Figure(go.Bar(x=vals, y=REGION_NAMES, orientation="h", marker_color=REGION_HEX,
                           text=[f"{v:.1f} mL" for v in vals], textposition="outside"))
    fig.update_layout(title=dict(text="Volumes (nested regions)", x=0.02, font=dict(size=14)), xaxis_title="mL",
                      yaxis=dict(autorange="reversed"), xaxis=dict(range=[0, max(vals) * 1.3 + 1]))
    return dark(fig, 260)


def fig_case_vs_test(test_df: pd.DataFrame, thr: Dict, case_u: float, case_dice: Optional[float]):
    tau = thr.get("tau_fail", 0.75)
    fail = test_df["dice_WT"] < tau
    ids = test_df["case_id"] if "case_id" in test_df else pd.Series([""] * len(test_df))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test_df["uncertainty"][~fail], y=test_df["dice_WT"][~fail], mode="markers", name="test patients, good outline",
                             marker=dict(color=GREEN, size=7, opacity=0.8), text=ids[~fail],
                             hovertemplate="%{text}<br>uncertainty %{x:.4f}<br>Dice %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=test_df["uncertainty"][fail], y=test_df["dice_WT"][fail], mode="markers", name=f"test patients, Dice < {tau}",
                             marker=dict(color="#8B3A3A", size=9, symbol="x"), text=ids[fail],
                             hovertemplate="%{text}<br>uncertainty %{x:.4f}<br>Dice %{y:.3f}<extra></extra>"))
    fig.add_vline(x=thr["threshold"], line=dict(color="#ECECEA", dash="dash"), annotation_text="threshold", annotation_position="top right")
    if case_dice is not None:
        fig.add_trace(go.Scatter(x=[case_u], y=[case_dice], mode="markers", name="this case", marker=dict(color=ACCENT, size=18, symbol="star", line=dict(color="#ECECEA", width=1))))
    else:
        fig.add_vline(x=case_u, line=dict(color=ACCENT, width=3), annotation_text="this case", annotation_position="bottom left")
    fig.update_layout(title=dict(text=f"This case against {len(test_df)} held-out test patients", x=0.02, font=dict(size=14)),
                      xaxis_title="case uncertainty (mean predictive entropy in ROI)", yaxis_title="Dice, whole tumour",
                      legend=dict(orientation="h", y=-0.25))
    return dark(fig, 420)


def fig_subsets(sub: pd.DataFrame):
    sub = sub.sort_values("dice_WT", ascending=True)
    fig = go.Figure()
    for col, nm, c in [("dice_WT", "Whole tumour", REGION_HEX[0]), ("dice_TC", "Tumour core", REGION_HEX[1]), ("dice_ET", "Enhancing", REGION_HEX[2])]:
        fig.add_trace(go.Bar(y=sub["subset"].str.upper(), x=sub[col], name=nm, orientation="h", marker_color=c))
    fig.update_layout(title=dict(text="Test Dice for every combination of input sequences", x=0.02, font=dict(size=14)),
                      barmode="group", xaxis=dict(range=[0, 1], title="Dice"), legend=dict(orientation="h", y=1.08))
    return dark(fig, 560)


ABLATION = load_json(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ablation.json"))


def fig_ablation(abl: Dict) -> go.Figure:
    ns = ["1", "2", "3", "4"]
    w, wo = abl["with_modality_dropout_by_n_sequences"], abl["no_modality_dropout_by_n_sequences"]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[f"{n} sequence" + ("s" if n != "1" else "") for n in ns], y=[w[n]["WT"] for n in ns], name="With modality dropout", marker_color=GREEN))
    fig.add_trace(go.Bar(x=[f"{n} sequence" + ("s" if n != "1" else "") for n in ns], y=[wo[n]["WT"] for n in ns], name="Without modality dropout", marker_color=ACCENT))
    fig.update_layout(title=dict(text="Whole-tumour Dice by number of sequences available (mean over subsets)", x=0.02, font=dict(size=14)),
                      barmode="group", yaxis=dict(range=[0, 1], title="Dice"), legend=dict(orientation="h", y=1.1))
    return dark(fig, 360)


def fig_history(hist: pd.DataFrame):
    fig = go.Figure()
    for c, nm, col in [("val_dice_WT", "val WT", REGION_HEX[0]), ("val_dice_TC", "val TC", REGION_HEX[1]), ("val_dice_ET", "val ET", REGION_HEX[2])]:
        if c in hist:
            fig.add_trace(go.Scatter(x=hist["epoch"], y=hist[c], name=nm, mode="lines+markers", line=dict(color=col)))
    if "train_loss" in hist:
        fig.add_trace(go.Scatter(x=hist["epoch"], y=hist["train_loss"], name="train loss", yaxis="y2", line=dict(dash="dot", color="#7C7F86")))
    fig.update_layout(title=dict(text="Training curve", x=0.02, font=dict(size=14)), xaxis_title="epoch", yaxis_title="Dice",
                      yaxis2=dict(title="loss", overlaying="y", side="right", showgrid=False), legend=dict(orientation="h", y=1.1))
    return dark(fig, 320)


# ------------------------------------------------------------------------------------------------
# Report
# ------------------------------------------------------------------------------------------------
def build_report(res, flagged, thr, model_info) -> Dict:
    pred, quality = res["pred"], res["quality"]
    return {
        "tool": "NeuroScope research prototype - NOT a medical device; outputs require expert review",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "modalities_used": res["present"],
        "modalities_missing": [m for m in MODALITIES if m not in res["present"]],
        "input_quality": quality.to_dict(),
        "tumour_volumes_ml": pred.volumes_ml,
        "case_uncertainty": pred.case_uncertainty,
        "review_threshold": thr,
        "needs_human_review": bool(flagged),
        "mc_samples": pred.mc_samples,
        "inference_seconds": res["infer_s"],
        "device": DEVICE,
        "model": model_info,
        "metrics_vs_ground_truth": res["metrics"],
    }


def report_markdown(r: Dict) -> str:
    v = r["tumour_volumes_ml"]
    lines = ["# NeuroScope analysis report", "", f"_{r['tool']}_", "", f"Generated: {r['timestamp']}", "",
             "## Summary",
             f"Abnormal region (whole tumour, incl. edema): **{v['WT_ml']:.1f} mL**. Tumour core: **{v['TC_ml']:.1f} mL**. "
             f"Enhancing tumour: **{v['ET_ml']:.1f} mL**. "
             + ("**Flagged for specialist review** (model uncertainty above the validated threshold)." if r["needs_human_review"]
                else "Model uncertainty below the validated review threshold; expert confirmation still required."),
             "", "## Input", f"- Sequences used: {', '.join(m.upper() for m in r['modalities_used'])}",
             f"- Missing: {', '.join(m.upper() for m in r['modalities_missing']) or 'none'}",
             f"- Quality status: **{r['input_quality']['status']}** (score {r['input_quality']['score']:.2f})"]
    for c in r["input_quality"]["checks"]:
        if c["status"] != "pass":
            lines.append(f"  - [{c['modality']}] {c['message']}")
    lines += ["", "## Reliability", f"- Case uncertainty (mean predictive entropy in ROI): {r['case_uncertainty']:.4f}",
              f"- Monte Carlo dropout samples: {r['mc_samples']}", f"- Inference: {r['inference_seconds']:.1f} s on {r['device']}",
              f"- **Needs human review: {'YES' if r['needs_human_review'] else 'no'}**"]
    if r["review_threshold"]:
        t = r["review_threshold"]
        lines.append(f"- Review threshold {t['threshold']:.4f} chosen on the validation set (rule: {t.get('rule')}, failure = WT Dice < {t.get('tau_fail')})")
    if r.get("metrics_vs_ground_truth"):
        m = r["metrics_vs_ground_truth"]
        lines += ["", "## Agreement with provided ground truth", f"- Dice WT {m['dice_WT']:.3f} | TC {m['dice_TC']:.3f} | ET {m['dice_ET']:.3f}"]
    lines += ["", "## Limitations", "- 2D slice-wise model trained on BraTS 2020 adult glioma only.",
              "- Uncertainty reflects model disagreement, not clinical reliability.",
              "- Quality checks are heuristics; passing them does not certify diagnostic suitability."]
    return "\n".join(lines)


def report_images(res) -> Dict:
    """Figures for the printable report: orthogonal views, magnified key slice, outline montage, most uncertain slice."""
    a = analyse_cache(res)
    bi = MODALITIES.index("flair") if "flair" in res["present"] else MODALITIES.index(res["present"][0])
    return {"figs": build_figures(res["img"], res["pred"].labels, res["pred"].entropy, res.get("seg"), bi, MODALITIES[bi].upper(), a)}


def analyse_cache(res):
    if res.get("_analysis") is None:
        res["_analysis"] = analyse(res, thr, bool(thr and res["pred"].case_uncertainty >= thr["threshold"]), sub_df, test_df, MODALITIES)
    return res["_analysis"]


def ci_txt(d):
    if not d:
        return None, ""
    lo, hi = d.get("ci95", [None, None])
    return d["mean"], (f"95% CI {lo:.3f}-{hi:.3f}" if lo is not None else "")


# ================================================================================================
# Page
# ================================================================================================
st.set_page_config(page_title="NeuroScope", layout="wide", initial_sidebar_state="expanded")
st.markdown(CSS, unsafe_allow_html=True)
components.html(TILT_JS, height=0)
components.html(_bg_script(), height=0)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("## NeuroScope")
    simple = st.radio("View", ["Simple", "Detailed"], horizontal=True, help="Simple hides the technical numbers. The analysis is identical.") == "Simple"
    st.divider()
    st.markdown("**Step 1. Scans**")
    has_demo = os.path.isdir(ARGS.demo_dir)
    drop = ()
    if not simple and has_demo:
        drop = tuple(st.multiselect("Simulate missing sequences on the example", MODALITIES, default=[],
                                    format_func=lambda m: MOD_LABEL[m], max_selections=3,
                                    help="Shows how the model behaves when a scan is unavailable. At least one sequence must remain."))
    use_demo = st.button("Load example patient", type="primary", width="stretch", disabled=not has_demo,
                         help="An anonymised case from the BraTS 2020 test split (never used in training)")
    st.caption("Or upload MRI files (.nii / .nii.gz). Any subset of the four sequences works. "
               "Privacy: scans are processed in memory for this session only. Nothing is stored on the server.")
    uploads = {m: st.file_uploader(MOD_LABEL[m], type=["nii", "gz"], key=m) for m in MODALITIES}
    seg_upload = st.file_uploader("Ground-truth mask (optional)", type=["nii", "gz"], key="seg") if not simple else None
    st.markdown("**Step 2. Run**")
    mc_default = 4 if DEVICE == "cpu" else 8
    if simple:
        mc = mc_default
        st.caption(f"Device: {DEVICE}. The model is run {mc} times per scan to estimate its confidence.")
    else:
        mc = st.slider("Monte Carlo dropout samples", 1, 16, mc_default, help="More samples give a better uncertainty estimate but take longer. Validated at 8.")
        st.caption(f"Device: {DEVICE}")
    run = st.button("Analyse uploaded scans", width="stretch")
    with st.expander("Advanced"):
        ckpt_path = st.text_input("Checkpoint", ARGS.ckpt)
        results_dir = st.text_input("Results dir", ARGS.results)

summary = load_json(os.path.join(results_dir, "summary.json"))
thr = load_json(os.path.join(results_dir, "threshold.json"))
test_df = load_csv(os.path.join(results_dir, "test_per_case.csv"))
sub_df = load_csv(os.path.join(results_dir, "subsets.csv"))
hist_df = load_csv(os.path.join(os.path.dirname(os.path.abspath(ckpt_path)), "history.csv"))

# ---------------- Header ----------------
_landing = st.session_state.get("result") is None
if _landing and os.path.isdir(ARGS.demo_dir):
    @st.cache_data(show_spinner=False)
    def _hero_html(demo_dir: str) -> str:
        vols, _ = load_demo_case(demo_dir)
        key = "flair" if "flair" in vols else next(k for k in vols if k != "seg")
        v = np.transpose(np.asarray(vols[key], np.float32), (2, 1, 0))
        lab = np.transpose(vols["seg"], (2, 1, 0)) if "seg" in vols else None
        return brain_html(v, lab, None, brain_step=3, tumour_step=2, height=600, title="NeuroScope",
                          subtitle="Brain tumour segmentation from MRI with a built-in measure of how much to trust each result. "
                                   "Shown: the example patient's FLAIR scan with the expert annotation. Load the example patient to see what the model finds.",
                          autorotate=True, show_controls=True)
    components.html(_hero_html(ARGS.demo_dir), height=600, scrolling=False)
    st.caption("Every point is a voxel of the real scan, rendered in your browser. Drag to rotate, scroll to zoom. Research prototype, not a medical device.")
else:
    st.title("NeuroScope")
    st.markdown("Brain tumour segmentation from MRI with a built-in uncertainty estimate, so the result says when it should be checked by a specialist. "
                "Works with any subset of the four standard sequences. Research prototype, not a medical device.")

if summary:
    n_test = summary.get("n_test", "?")
    dwt, dwt_ci = ci_txt(summary.get("dice_WT")); dtc, _ = ci_txt(summary.get("dice_TC")); det, _ = ci_txt(summary.get("dice_ET"))
    rel, flag = summary.get("reliability", {}), summary.get("review_flag", {})
    inf, _ = ci_txt(summary.get("infer_s"))
    c = st.columns(4)
    if simple:
        c[0].metric("Overlap with expert outlines", f"{100*dwt:.0f}%", help=f"Dice, whole tumour, on {n_test} test patients never seen in training")
        c[1].metric("Poor outlines caught by the confidence score", f"{100*rel.get('failure_auroc', float('nan')):.0f}%", help="AUROC of case uncertainty for detecting WT Dice < 0.75")
        c[2].metric("Sequence combinations supported", "15 of 15", help="Any subset of T1, T1ce, T2, FLAIR")
        c[3].metric("Time per patient on GPU", f"{inf:.1f} s" if inf else "n/a", help="Including the repeated runs used for the confidence estimate")
    else:
        c[0].metric("Test Dice  WT / TC / ET", f"{dwt:.3f} / {dtc:.3f} / {det:.3f}", help=f"n = {n_test}; WT {dwt_ci}")
        c[1].metric("Failure-detection AUROC", f"{rel.get('failure_auroc', float('nan')):.3f}", help=f"Spearman rho = {rel.get('spearman_rho', float('nan')):.2f}; AURC {rel.get('aurc', float('nan')):.3f}")
        c[2].metric("Review flag", f"{100*flag.get('flag_rate', float('nan')):.0f}% of cases",
                    help=f"sensitivity {flag.get('flag_sensitivity', float('nan')):.2f}, specificity {flag.get('flag_specificity', float('nan')):.2f}; Dice on accepted cases {flag.get('dice_accepted', float('nan')):.3f}")
        c[3].metric("Inference per case (GPU)", f"{inf:.1f} s" if inf else "n/a", help=f"MC x{summary.get('mc_samples')}; 7.8M parameters")

if not os.path.exists(ckpt_path):
    st.error(f"Model file not found: {ckpt_path}. Train first (scripts/train.py) or pass --ckpt.")
    st.stop()
model, model_info = get_model(ckpt_path, DEVICE)

# ---------------- Run ----------------
if "result" not in st.session_state:
    st.session_state.result, st.session_state.source = None, None

if use_demo:
    with st.status("Analysing example patient", expanded=True) as status:
        st.write("Loading scans, checking quality, preprocessing")
        st.write(f"Running the model {mc} times with dropout (first CPU run takes 1-2 min; repeats are cached)")
        try:
            st.session_state.result = run_demo_inference(ARGS.demo_dir, mc, ckpt_path, drop)
        except Exception as err:
            status.update(label="Analysis could not run", state="error")
            st.error(f"The analysis could not run: {err}. Keep at least one sequence and try again.")
            st.stop()
        st.session_state.source = "Example patient, BraTS 2020 test split" + (f" (without {', '.join(m.upper() for m in drop)})" if drop else "")
        status.update(label="Analysis complete", state="complete", expanded=False)
elif run:
    vols, affine = {}, np.eye(4)
    for m in MODALITIES:
        r = read_upload(uploads[m])
        if r is not None:
            vols[m], affine, _ = r
    if seg_upload is not None:
        vols["seg"] = remap_labels(read_upload(seg_upload)[0].astype(np.int16))
    if not vols:
        st.warning("Upload at least one MRI sequence, or load the example patient.")
    else:
        q = assess_case({m: vols.get(m) for m in MODALITIES})
        if q.status == "fail":
            st.error("Scans failed the quality check: " + "; ".join(q.warnings()))
            st.stop()
        with st.status("Analysing scans", expanded=True) as status:
            st.write(f"Checking quality, preprocessing, running the model {mc} times")
            try:
                st.session_state.result = run_case(vols, affine, mc, ckpt_path)
            except Exception as err:
                st.error(f"The analysis could not run on these files: {err}. Check that they are brain MRI volumes in NIfTI format.")
                st.stop()
            st.session_state.source = "Uploaded scans"
            status.update(label="Analysis complete", state="complete", expanded=False)

res = st.session_state.result
if res is not None:
    _tok = str(res["token"])
    if res.get("_comps") is None:
        res["_comps"] = find_components(res["pred"].labels, res["pred"].entropy, res["voxel_ml"])
    _rej = [c["id"] for c in res["_comps"] if st.session_state.get(f"rej_{_tok}_{c['id']}")]
    if _rej:
        _cache = st.session_state.setdefault("_corr", {})
        _key = (_tok, tuple(sorted(_rej)))
        if _key not in _cache:
            _r2 = apply_rejections(res, _rej)
            if _r2["seg"] is not None:
                _r2["metrics"] = region_metrics(_r2["pred"].regions, labels_to_regions(_r2["seg"]), with_hd95=False)
            _cache.clear()
            _cache[_key] = _r2
        base_res, res = res, _cache[_key]
    else:
        base_res = res

# ---------------- Landing ----------------
if res is None:
    st.markdown("#### What is different from a standard segmentation model")
    st.table(pd.DataFrame({
        "": ["Input", "Output", "Self-assessment", "Bad scans", "Evidence"],
        "Typical U-Net demo": ["Needs all four sequences; fails or degrades silently if one is missing",
                               "An outline, presented as fact",
                               "None. A wrong outline looks exactly like a right one",
                               "Accepted without checks",
                               "A single Dice number on an unspecified split"],
        "NeuroScope": ["Any of the 15 sequence combinations; trained with modality dropout and tested on every subset",
                       "Outline, volumes, and a per-pixel uncertainty map",
                       "Case-level uncertainty compared with a threshold fixed on validation data; the case is flagged for a specialist when it is unreliable",
                       "Quality gate: missing sequences, shape, brain fraction, SNR, saturation, slice dropout",
                       "74 held-out patients, bootstrap confidence intervals, failure-detection AUROC, risk-coverage curve, per-case data in the app"],
    }).set_index(""))
    st.markdown("#### How to use")
    st.markdown("""
1. **Scans** - click *Load example patient* in the sidebar, or upload your own MRI files. Missing sequences are allowed.
2. **Run** - the app checks scan quality, segments the tumour, and runs the model several times to measure how much its answers vary.
3. **Explore** - volumes, an interactive slice viewer with hover readout, a 3D reconstruction, where this case sits among the test patients, and a downloadable report.
    """)
    with st.expander("Glossary"):
        st.markdown("""
- **MRI sequences (T1, T1ce, T2, FLAIR)** - four scan types from one session; each shows different tissue. T1ce uses contrast dye and shows active tumour; FLAIR shows swelling.
- **Segmentation** - outlining the tumour on every slice. Manually this takes a radiologist roughly 20-60 minutes per patient.
- **Whole tumour / tumour core / enhancing tumour** - everything abnormal including swelling; the solid centre; the part that takes up contrast dye.
- **Uncertainty** - the model is run several times with dropout active. Where the outputs disagree, the model is unsure. The case-level score is compared with a threshold fixed on validation data.
- **Dice score** - overlap between two outlines, 0 to 1. Radiologists typically agree with each other at about 0.85-0.90.
        """)
    if summary and sub_df is not None:
        with st.expander("Validated performance"):
            st.plotly_chart(fig_subsets(sub_df), width="stretch")
    st.caption("Built on the public BraTS 2020 dataset. Not a medical device; not for diagnosis or treatment decisions.")

# ---------------- Results ----------------
else:
    pred, img, quality, present, metrics = res["pred"], res["img"], res["quality"], res["present"], res["metrics"]
    v = pred.volumes_ml
    flagged = bool(thr and pred.case_uncertainty >= thr["threshold"])
    missing = [m for m in MODALITIES if m not in present]
    if res["meshes"] is None:
        with st.spinner("Building 3D surfaces"):
            res["meshes"] = build_meshes(res)

    st.caption(st.session_state.source)
    if flagged:
        st.error(("Needs specialist review. The model's repeated outputs disagreed more than usual on this scan, so the outline may be unreliable." if simple else
                  f"Needs review: case uncertainty {pred.case_uncertainty:.4f} >= validated threshold {thr['threshold']:.4f} (Youden's J on validation, failure = WT Dice < {thr.get('tau_fail')})."))
    elif thr:
        extra = f" On the test set, accepted cases averaged WT Dice {summary['review_flag']['dice_accepted']:.3f}." if (summary and not simple) else ""
        st.success(("Confident result. The model's repeated outputs agreed closely. A specialist should still confirm it." if simple else
                    f"Below review threshold: case uncertainty {pred.case_uncertainty:.4f} < {thr['threshold']:.4f}.{extra}"))
    else:
        st.warning("No validated threshold found. Run scripts/evaluate.py to enable the review flag.")

    k = st.columns(5)
    k[0].metric("Whole tumour", f"{v['WT_ml']:.1f} mL", help="Everything abnormal, including edema")
    k[1].metric("Tumour core", f"{v['TC_ml']:.1f} mL", help="Solid tumour without edema")
    k[2].metric("Enhancing tumour", f"{v['ET_ml']:.1f} mL", help="Takes up contrast dye; usually the most active part")
    k[3].metric("Case uncertainty", f"{pred.case_uncertainty:.4f}", help="Mean predictive entropy in the tumour region over the repeated runs")
    k[4].metric("Inference", f"{res['infer_s']:.1f} s", help=f"MC x{pred.mc_samples} on {DEVICE}")
    if metrics:
        st.info(f"Agreement with the expert outline (Dice): whole tumour {metrics['dice_WT']:.3f}, core {metrics['dice_TC']:.3f}, enhancing {metrics['dice_ET']:.3f}")
    if missing:
        st.warning(f"Running without {', '.join(m.upper() for m in missing)}. Accuracy for the affected regions is lower; see the subset chart under Performance.")

    _tab_names = ["3D brain", "Slice viewer", "Measure and review", "Follow-up", "3D surfaces", "Uncertainty", "Missing sequences", "Performance", "Report"]
    try:   # keep the open tab across reruns (Streamlit >= 1.6x); older versions fall back to plain tabs
        _tabs = st.tabs(_tab_names, key="main_tabs", on_change="rerun")
    except TypeError:
        _tabs = st.tabs(_tab_names)
    tab_gl, tab_slices, tab_meas, tab_fu, tab_3d, tab_unc, tab_rob, tab_perf, tab_report = _tabs
    if res.get("rejected"):
        st.info(f"Reviewer correction active: {len(res['rejected'])} region(s) rejected, {res['removed_ml']:.2f} mL removed. All tabs and the report use the corrected segmentation.")

    tum_z = np.where(pred.regions[0].reshape(pred.regions.shape[1], -1).sum(1) > 0)[0]
    default_z = int(tum_z[len(tum_z) // 2]) if tum_z.size else img.shape[1] // 2
    if "z" not in st.session_state:
        st.session_state.z = default_z

    with tab_gl:
        g1, g2 = st.columns([3, 1])
        with g2:
            gl_mod = st.selectbox("Brain from", present, index=present.index("flair") if "flair" in present else 0, format_func=lambda m: MOD_LABEL[m], key="glmod")
            gl_plane = st.checkbox("MRI slice plane in the scene", True, key="glplane")
            if gl_plane:
                st.session_state.z = st.slider("Axial slice", 0, img.shape[1] - 1, st.session_state.z, key="z_gl")
            st.caption("Buttons in the scene: Brain / Tumour / Uncertainty / Slice / Rotate / Reset. "
                       "Uncertainty recolours the tumour: bright white where the repeated runs disagreed. Every point is a voxel of this patient's scan.")
        with g1:
            @st.cache_data(show_spinner=False)
            def _gl_html(src: str, mod: str, z: Optional[int], mc_: int, tok: float):
                return brain_html(img[MODALITIES.index(mod)], pred.labels, pred.entropy, brain_step=2, tumour_step=1, height=620,
                                  title="", subtitle="", slice_z=z, autorotate=True, show_controls=True,
                                  volumes={k: float((pred.labels == k).sum()) * res["voxel_ml"] for k in (1, 2, 3)})
            components.html(_gl_html(st.session_state.source, gl_mod, st.session_state.z if gl_plane else None, pred.mc_samples, res["token"]), height=620, scrolling=False)

    with tab_slices:
        a, b = st.columns([1, 2.2])
        with a:
            plane = st.radio("Plane", ["Axial", "Coronal", "Sagittal"], horizontal=True)
            axis = ["Axial", "Coronal", "Sagittal"].index(plane)
            base_mod = st.selectbox("Background sequence", present, index=present.index("flair") if "flair" in present else 0, format_func=lambda m: MOD_LABEL[m])
            show_seg = st.checkbox("Tumour outline", True)
            show_unc = st.checkbox("Uncertainty map", False, help="Brighter = the repeated runs disagreed more. Usually at the tumour edge.")
            seg_alpha = st.slider("Outline opacity", 0.1, 1.0, 0.55)
            unc_alpha = st.slider("Uncertainty opacity", 0.1, 1.0, 0.8) if show_unc else 0.8
            show_gt = st.checkbox("Expert outline (contour)", False) if (res["seg"] is not None and not simple) else False
            st.caption("Red: necrotic / non-enhancing core. Green: edema. Yellow: enhancing tumour. White glow: uncertainty. Hover for a per-pixel readout; scroll to zoom, drag to pan.")
        with b:
            n = img.shape[1 + axis]
            if axis == 0:
                st.session_state.z = st.slider("Axial slice", 0, n - 1, st.session_state.z, key="z_axial")
                i = st.session_state.z
            else:
                i = st.slider(f"{plane} slice", 0, n - 1, n // 2, key=f"z_{axis}")
            base = take_slice(img[MODALITIES.index(base_mod)], axis, i)
            labs = take_slice(pred.labels, axis, i)
            ent = take_slice(pred.entropy, axis, i)
            gt = take_slice(res["seg"], axis, i) if show_gt else None
            st.plotly_chart(slice_figure(base, labs, ent, show_seg, show_unc, seg_alpha, unc_alpha, gt, f"{plane} slice {i} - {MOD_LABEL[base_mod]}"),
                            width="stretch", config={"scrollZoom": True, "displaylogo": False})

    with tab_meas:
        meas = bidimensional(pred.labels, res["affine"], "ET") or bidimensional(pred.labels, res["affine"], "TC")
        m1, m2 = st.columns([1.5, 1])
        bi = MODALITIES.index("flair") if "flair" in present else MODALITIES.index(present[0])
        with m1:
            if meas:
                st.plotly_chart(fig_rano(res, meas, bi), width="stretch", key="rano_fig")
            else:
                st.info("No tumour to measure.")
        with m2:
            st.markdown("#### Bidimensional measurement")
            if meas:
                tgt = "enhancing tumour" if meas["target"] == "ET" else "tumour core (no enhancing tumour found)"
                mc1, mc2 = st.columns(2)
                mc1.metric("Longest diameter", f"{meas['d1_mm']:.1f} mm")
                mc2.metric("Perpendicular", f"{meas['d2_mm']:.1f} mm")
                mc1.metric("Product", f"{meas['product_mm2']:.0f} mm²")
                mc2.metric("Measurable", "Yes" if meas["measurable"] else "No", help="RANO: both diameters at least 10 mm")
                st.caption(f"Measured automatically on the {tgt}, axial slice {meas['slice']}: the slice where the longest diameter times the longest "
                           "perpendicular diameter is largest. This is the measurement radiologists use in the RANO response criteria, which normally "
                           "takes a manual measurement on screen.")
        st.markdown("#### Review detected regions")
        comps = base_res.get("_comps") or []
        if not comps:
            st.caption("No regions detected.")
        else:
            st.caption("Each separate region the model predicted. Small isolated regions are a common source of false positives. "
                       "Tick Reject to remove a region; volumes, 3D views, measurements and the report update.")
            hdr = st.columns([0.7, 0.6, 1, 1.4, 1.1, 2.2])
            for c_, t_ in zip(hdr, ["Reject", "#", "Volume", "Slices", "Uncertainty", "Composition (mL)"]):
                c_.markdown(f"<span style='color:#A3A5AA;font-size:13px'>{t_}</span>", unsafe_allow_html=True)
            for c in comps[:12]:
                row = st.columns([0.7, 0.6, 1, 1.4, 1.1, 2.2])
                row[0].checkbox("Reject", key=f"rej_{_tok}_{c['id']}", label_visibility="collapsed", disabled=(c["rank"] == 1 and len(comps) == 1))
                row[1].write(f"{c['rank']}" + (" (main)" if c["rank"] == 1 else ""))
                row[2].write(f"{c['ml']:.2f} mL")
                row[3].write(f"{c['slices'][0]} to {c['slices'][1]}")
                row[4].write(f"{c['mean_u']:.2f}")
                comp = c["composition"]
                row[5].write(f"core {comp[1]:.2f}, enh {comp[3]:.2f}, edema {comp[2]:.2f}")
            if len(comps) > 12:
                st.caption(f"{len(comps) - 12} further regions not shown.")
            if st.button("Restore all regions"):
                for c in comps:
                    st.session_state[f"rej_{_tok}_{c['id']}"] = False
                st.rerun()

    with tab_fu:
        st.markdown("#### Compare with another examination of the same patient")
        st.caption("Analyses a second scan with the same model and compares volumes and the bidimensional measurement. "
                   "The response category is an imaging-only approximation of RANO.")
        is_demo = str(st.session_state.source or "").startswith("Example patient")
        opts = (["Simulated earlier scan (demonstration)"] if is_demo else []) + ["Upload another examination"]
        how = st.radio("Second examination", opts, horizontal=True, key="fu_how")
        fu = st.session_state.get("followup")
        if fu and fu.get("base") != _tok:
            fu = st.session_state.followup = None
        if how.startswith("Simulated"):
            st.caption("BraTS 2020 has one scan per patient. For demonstration, the earlier scan is synthesised from the example patient by digitally "
                       "shrinking the tumour region by about 22% in each direction; the model then analyses it like any other scan. Clearly labelled as simulated everywhere.")
            if st.button("Analyse simulated earlier scan", type="primary"):
                with st.spinner("Creating and analysing the simulated earlier scan (first time 1-2 min on CPU)"):
                    other = run_simulated_prior(ARGS.demo_dir, mc, ckpt_path)
                st.session_state.followup = fu = {"base": _tok, "other": other, "earlier": True, "simulated": True}
        else:
            fcols = st.columns(4)
            fu_up = {m: fcols[i].file_uploader(MOD_LABEL[m], type=["nii", "gz"], key=f"fu_{m}") for i, m in enumerate(MODALITIES)}
            when = st.radio("This examination was taken", ["Before the current one", "After the current one"], horizontal=True, key="fu_when")
            if st.button("Analyse and compare", type="primary"):
                v2, aff2 = {}, np.eye(4)
                for m in MODALITIES:
                    r_ = read_upload(fu_up[m])
                    if r_ is not None:
                        v2[m], aff2, _ = r_
                if not v2:
                    st.warning("Upload at least one sequence of the other examination.")
                else:
                    try:
                        with st.spinner("Analysing the other examination"):
                            other = run_case(v2, aff2, mc, ckpt_path)
                        st.session_state.followup = fu = {"base": _tok, "other": other, "earlier": when.startswith("Before"), "simulated": False}
                    except Exception as err:
                        st.error(f"The other examination could not be analysed: {err}.")
        if fu:
            cmp_ = build_followup(res, fu["other"], fu["earlier"], fu["simulated"])
            st.session_state.followup_cmp = cmp_
            if cmp_["note"]:
                st.warning(cmp_["note"])
            st.markdown(f"**{cmp_['category']}.** {cmp_['category_detail']}")
            st.table(pd.DataFrame([{"Measure": r_["name"], "Earlier": r_["earlier"], "Later": r_["later"], "Change": r_["change"]} for r_ in cmp_["rows"]]).set_index("Measure"))
            fc = st.columns(2)
            for col_, (b64, cap) in zip(fc, cmp_["panels"]):
                col_.image("data:image/png;base64," + b64, caption=cap, width="stretch")
            fig_c = go.Figure()
            names = [r_["name"].replace(" (mL)", "") for r_ in cmp_["rows"][:4]]
            fig_c.add_trace(go.Bar(x=names, y=[r_["a"] for r_ in cmp_["rows"][:4]], name="Earlier", marker_color="#6B6E74"))
            fig_c.add_trace(go.Bar(x=names, y=[r_["b"] for r_ in cmp_["rows"][:4]], name="Later", marker_color=ACCENT))
            fig_c.update_layout(barmode="group", yaxis_title="mL", legend=dict(orientation="h", y=1.1))
            st.plotly_chart(dark(fig_c, 340), width="stretch", key="fu_bars")
            st.caption("Full RANO assessment also uses non-enhancing (FLAIR) disease, corticosteroid dose and clinical status. No image registration is performed; "
                       "measurements are compared, not voxels.")
        else:
            st.session_state.followup_cmp = None

    with tab_3d:
        a, b = st.columns([1, 3])
        with a:
            st.markdown("**Layers**")
            show = {"brain": st.checkbox("Brain surface", True)}
            for nm in REGION_NAMES:
                show[nm] = st.checkbox(nm, True)
            show_plane = st.checkbox("MRI slice plane", False, help="Follows the axial slice slider")
            opacity = st.slider("Tumour opacity", 0.2, 1.0, 0.9)
            base3d = st.selectbox("Plane sequence", present, index=present.index("flair") if "flair" in present else 0, format_func=lambda m: MOD_LABEL[m], key="b3d")
            if show_plane:
                st.session_state.z = st.slider("Axial slice", 0, img.shape[1] - 1, st.session_state.z, key="z_3d")
            st.caption("Drag to rotate, scroll to zoom, right-drag to pan. Surfaces are reconstructed from the model output with marching cubes." if HAVE_SKIMAGE
                       else "Install scikit-image for smooth surfaces: pip install scikit-image")
        with b:
            if pred.regions[0].sum() == 0:
                st.info("No tumour voxels predicted; nothing to render.")
            else:
                st.plotly_chart(fig_3d(res, st.session_state.z, show, opacity, show_plane, img[MODALITIES.index(base3d)]),
                                width="stretch", config={"displaylogo": False})

    with tab_unc:
        a, b = st.columns([1.3, 1])
        with a:
            st.plotly_chart(fig_slice_profile(res, st.session_state.z), width="stretch")
        with b:
            st.plotly_chart(fig_volumes(pred), width="stretch")
            if not simple:
                mc_std = float(pred.mc_std[pred.regions[0] > 0].mean()) if pred.regions[0].sum() else 0.0
                st.markdown(f"- Mean entropy in tumour ROI: **{pred.case_uncertainty:.4f}**  \n- Mean MC std in ROI: **{mc_std:.4f}**  \n- MC samples: **{pred.mc_samples}**")
        if test_df is not None and thr:
            st.markdown("**Review threshold, live on the test set**")
            st.caption("Drag the threshold. Everything to its right is sent for specialist review. The validated setting is marked; it was chosen on the validation set, not here.")
            umax = float(test_df["uncertainty"].max())
            t_live = st.slider("Threshold (case uncertainty)", 0.0, round(umax * 1.05, 3), float(thr["threshold"]), step=0.001, format="%.3f")
            tau = thr.get("tau_fail", 0.75)
            fl = test_df["uncertainty"] >= t_live
            bad = test_df["dice_WT"] < tau
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Sent for review", f"{100*fl.mean():.0f}%", help="Share of test patients flagged at this threshold")
            m2.metric("Poor outlines caught", f"{int((fl & bad).sum())} of {int(bad.sum())}", help=f"Cases with WT Dice < {tau} that are flagged")
            m3.metric("Dice of accepted cases", f"{test_df['dice_WT'][~fl].mean():.3f}" if (~fl).any() else "n/a", help="Mean WT Dice of cases not flagged")
            m4.metric("This case", "flagged" if pred.case_uncertainty >= t_live else "accepted")
            thr_live = dict(thr, threshold=t_live)
            st.plotly_chart(fig_case_vs_test(test_df, thr_live, pred.case_uncertainty, metrics["dice_WT"] if metrics else None), width="stretch")
            st.caption(f"Each dot is one held-out test patient; crosses are poor outlines (WT Dice < {tau}). Uncertainty rises as outline quality falls, "
                       f"which is what lets the flag work. Validated threshold: {thr['threshold']:.4f}. The star (or vertical line without ground truth) is this case.")

    with tab_rob:
        st.markdown("**Same patient, fewer scans.** Most models need all four sequences. Run this patient again with some removed and compare.")
        if st.session_state.source and st.session_state.source.startswith("Example patient"):
            cA, cB = st.columns([2, 1])
            drop2 = cA.multiselect("Remove sequences", MODALITIES, default=["t1ce"], format_func=lambda m: MOD_LABEL[m], max_selections=3, key="drop2")
            go_rob = cB.button("Run comparison", type="primary", width="stretch", disabled=(len(drop2) == 0 or len(drop2) == 4))
            if go_rob:
                with st.spinner("Running the reduced-input case (first time 1-2 min on CPU; cached afterwards)"):
                    st.session_state.rob = (tuple(drop2), run_demo_inference(ARGS.demo_dir, mc, ckpt_path, tuple(drop2)))
                full = run_demo_inference(ARGS.demo_dir, mc, ckpt_path, ())
                st.session_state.rob_full = full
            if st.session_state.get("rob"):
                dr, r2 = st.session_state.rob
                full = st.session_state.get("rob_full") or res
                p1, p2 = full["pred"], r2["pred"]
                rows = [("Sequences", "T1, T1ce, T2, FLAIR", ", ".join(MOD_LABEL[m] for m in r2["present"]))]
                for nm, kk in zip(REGION_NAMES, REGION_KEYS):
                    rows.append((f"{nm} (mL)", f"{p1.volumes_ml[kk]:.1f}", f"{p2.volumes_ml[kk]:.1f}"))
                if full["metrics"] and r2["metrics"]:
                    for nm, kk in zip(REGION_NAMES, ["dice_WT", "dice_TC", "dice_ET"]):
                        rows.append((f"Dice vs expert, {nm.lower()}", f"{full['metrics'][kk]:.3f}", f"{r2['metrics'][kk]:.3f}"))
                rows.append(("Case uncertainty", f"{p1.case_uncertainty:.4f}", f"{p2.case_uncertainty:.4f}"))
                f1 = bool(thr and p1.case_uncertainty >= thr["threshold"]); f2 = bool(thr and p2.case_uncertainty >= thr["threshold"])
                rows.append(("Review flag", "flagged" if f1 else "accepted", "flagged" if f2 else "accepted"))
                st.table(pd.DataFrame(rows, columns=["", "All four sequences", "Without " + ", ".join(m.upper() for m in dr)]).set_index(""))
                zc = st.session_state.z
                bm = "flair" if "flair" in r2["present"] else r2["present"][0]
                g1, g2 = st.columns(2)
                g1.plotly_chart(key="rob_a", figure_or_data=slice_figure(full["img"][MODALITIES.index(bm)][zc], p1.labels[zc], p1.entropy[zc], True, True, 0.55, 0.8, None, f"All four, slice {zc}"), width="stretch")
                g2.plotly_chart(key="rob_b", figure_or_data=slice_figure(r2["img"][MODALITIES.index(bm)][zc], p2.labels[zc], p2.entropy[zc], True, True, 0.55, 0.8, None, f"Without {', '.join(m.upper() for m in dr)}, slice {zc}"), width="stretch")
                st.caption("Uncertainty is shown as the white glow. When a sequence is missing, the model usually becomes less certain in exactly the regions that depend on it, which is the intended behaviour.")
        else:
            st.info("Available for the example patient. For uploaded scans, simply leave out a file in the sidebar and run again.")
        if sub_df is not None:
            st.plotly_chart(fig_subsets(sub_df), width="stretch", key="subsets_rob")

    with tab_perf:
        if summary:
            if simple:
                st.markdown(f"Tested on **{summary.get('n_test')}** patients held out from training. Overlap with expert outlines (Dice): whole tumour "
                            f"**{summary['dice_WT']['mean']:.3f}**, core **{summary['dice_TC']['mean']:.3f}**, enhancing **{summary['dice_ET']['mean']:.3f}**. "
                            f"The uncertainty score separated poor outlines from good ones with AUROC **{summary['reliability']['failure_auroc']:.3f}**.")
            else:
                st.json({k: summary[k] for k in ["n_test", "n_val", "mc_samples", "dice_WT", "dice_TC", "dice_ET", "hd95_WT", "sensitivity_WT",
                                                 "specificity_WT", "reliability", "review_flag"] if k in summary}, expanded=False)
            if sub_df is not None:
                st.plotly_chart(fig_subsets(sub_df), width="stretch", key="subsets_perf")
        if ABLATION and summary:
            st.markdown("#### Ablations")
            st.caption("Two controlled comparisons on the same 74 test patients, run to check that each design choice earns its place.")
            a1, a2 = st.columns(2)
            a1.plotly_chart(fig_ablation(ABLATION), width="stretch", key="ablation_moddrop")
            nm, mc1 = ABLATION["no_modality_dropout"]["test"], ABLATION["mc1_baseline"]["test"]
            rel = summary.get("reliability", {})
            tbl = pd.DataFrame([
                {"Setting": "Full model (modality dropout, MC=8)", "Dice WT": summary["dice_WT"]["mean"], "Dice TC": summary["dice_TC"]["mean"], "Dice ET": summary["dice_ET"]["mean"], "Failure AUROC": rel.get("failure_auroc")},
                {"Setting": "No modality dropout (MC=8)", "Dice WT": nm["WT"], "Dice TC": nm["TC"], "Dice ET": nm["ET"], "Failure AUROC": nm["failure_auroc"]},
                {"Setting": "Single pass, no MC sampling (MC=1)", "Dice WT": mc1["WT"], "Dice TC": mc1["TC"], "Dice ET": mc1["ET"], "Failure AUROC": mc1["failure_auroc"]},
            ]).set_index("Setting").round(3)
            a2.dataframe(tbl, width="stretch")
            a2.caption("Failure AUROC: how well the uncertainty score separates poor outlines (WT Dice below 0.75) from good ones.")
            auc8 = rel.get("failure_auroc")
            auc_txt = f"{auc8:.3f} with eight samples" if isinstance(auc8, (int, float)) and auc8 == auc8 else "see the table for eight samples"
            st.markdown(
                "**Reading the ablations.** With all four sequences the two models score the same, so modality dropout costs nothing at full input. "
                "Its value shows when sequences are missing: with one sequence the dropout-trained model reaches whole-tumour Dice "
                f"{ABLATION['with_modality_dropout_by_n_sequences']['1']['WT']:.2f} against {ABLATION['no_modality_dropout_by_n_sequences']['1']['WT']:.2f} without it, "
                f"and with two sequences {ABLATION['with_modality_dropout_by_n_sequences']['2']['WT']:.2f} against {ABLATION['no_modality_dropout_by_n_sequences']['2']['WT']:.2f}. "
                f"Monte Carlo sampling is what powers the review flag: failure AUROC is {mc1['failure_auroc']:.3f} with a single pass and {auc_txt}.")
            cols = st.columns(2)
            if hist_df is not None:
                cols[0].plotly_chart(fig_history(hist_df), width="stretch")
            p = os.path.join(results_dir, "fig_risk_coverage.png")
            if os.path.exists(p):
                cols[1].image(p, caption="Risk-coverage: Dice of accepted cases as the review flag rejects more")
        else:
            st.write("No results/summary.json found. Run scripts/evaluate.py.")
        st.markdown("""
**Model.** 2D U-Net, 7.8M parameters, trained with modality dropout so it works with any subset of the four sequences.
Uncertainty from Monte Carlo dropout: the network runs several times and the disagreement (predictive entropy) is measured.
The review threshold was fixed on the validation set and never tuned on the test set.

**Limitations.** Trained on BraTS 2020 (adult glioma, skull-stripped, co-registered, 1 mm isotropic). Not validated on other tumour types,
scanners, children, or post-operative scans. Uncertainty measures model disagreement, not correctness. Research prototype for
human-in-the-loop review, not a diagnostic device.
        """)

    with tab_report:
        report = build_report(res, flagged, thr, model_info)
        md = report_markdown(report)
        analysis = analyse_cache(res)
        if "_report_imgs" not in res:
            res["_report_imgs"] = report_images(res)
        rhtml = report_html(analysis, res, flagged, thr, model_info, str(st.session_state.get("source", "case")), res["_report_imgs"],
                            pred.mc_samples, DEVICE, res["infer_s"], followup=st.session_state.get("followup_cmp"))
        d0, d1, d2, d3 = st.columns(4)
        d0.download_button("Report (.html, print to PDF)", rhtml, file_name="neuroscope_report.html", mime="text/html", width="stretch", type="primary")
        d1.download_button("Report (.md)", md, file_name="neuroscope_report.md", width="stretch")
        d2.download_button("Report (.json)", json.dumps({**report, "findings": analysis["findings"]}, indent=2, default=float), file_name="neuroscope_report.json", width="stretch")
        lab_full = uncrop(pred.labels, res["meta"]).astype(np.uint8)
        lab_full[lab_full == 3] = 4
        d3.download_button("Segmentation mask (.nii)", nib.Nifti1Image(lab_full, res["affine"]).to_bytes(), file_name="neuroscope_seg.nii", width="stretch")
        components.html(rhtml, height=1500, scrolling=True)

    if st.button("Clear"):
        st.session_state.result = None
        st.rerun()

st.caption("NeuroScope. Research prototype built on BraTS 2020. Not a medical device.")
