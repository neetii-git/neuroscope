"""NeuroScope — brain tumour MRI analysis with built-in self-checks (research prototype).

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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import streamlit as st
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from neuroscope.data.brats import MODALITIES, labels_to_regions, load_nifti_with_affine, remap_labels  # noqa: E402
from neuroscope.data.preprocess import preprocess_case, uncrop  # noqa: E402
from neuroscope.eval.metrics import region_metrics  # noqa: E402
from neuroscope.inference.predictor import Prediction, load_checkpoint, predict_volume  # noqa: E402
from neuroscope.quality.iqa import assess_case  # noqa: E402

# ------------------------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------------------------
LABEL_COLORS = {1: (0.86, 0.15, 0.15), 2: (0.16, 0.65, 0.35), 3: (0.98, 0.75, 0.14)}
LABEL_NAMES = {1: "Dead / non-enhancing core", 2: "Swelling (edema)", 3: "Active (enhancing) tumour"}
MOD_HELP = {
    "t1": "T1 — basic anatomy scan",
    "t1ce": "T1ce — T1 with contrast dye (shows active tumour)",
    "t2": "T2 — fluid-sensitive scan",
    "flair": "FLAIR — shows swelling around the tumour",
}
BRAIN_ML = 1300.0  # typical adult brain volume, for plain-language context only

PRIMARY = "#0F766E"


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

# ------------------------------------------------------------------------------------------------
# Styling
# ------------------------------------------------------------------------------------------------
CSS = """
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1250px;}
section[data-testid="stSidebar"] {background: #FFFFFF; border-right: 1px solid #E2E8F0;}
.ns-hero {background: linear-gradient(120deg, #0F766E 0%, #115E59 55%, #0B3B5A 100%);
          color: white; border-radius: 18px; padding: 28px 32px; margin-bottom: 14px;
          box-shadow: 0 10px 30px rgba(15,118,110,0.25);}
.ns-hero h1 {color: white; font-size: 2.2rem; margin: 0 0 6px 0; letter-spacing: -0.02em;}
.ns-hero p {color: #E6FFFA; font-size: 1.08rem; margin: 0; max-width: 820px;}
.ns-pill {display: inline-block; background: rgba(255,255,255,0.16); border: 1px solid rgba(255,255,255,0.35);
          padding: 4px 12px; border-radius: 999px; font-size: 0.8rem; margin: 10px 8px 0 0; color: white;}
.ns-stat {background: white; border: 1px solid #E2E8F0; border-radius: 14px; padding: 16px 18px; height: 100%;
          box-shadow: 0 2px 8px rgba(2,6,23,0.04);}
.ns-stat .v {font-size: 1.9rem; font-weight: 700; color: #0F172A; line-height: 1.1;}
.ns-stat .k {font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.06em; color: #64748B; margin-bottom: 6px;}
.ns-stat .s {font-size: 0.82rem; color: #475569; margin-top: 6px;}
.ns-step {background: white; border: 1px solid #E2E8F0; border-radius: 14px; padding: 16px 18px; height: 100%;}
.ns-step .n {display: inline-block; width: 28px; height: 28px; border-radius: 50%; background: #0F766E; color: white;
             text-align: center; line-height: 28px; font-weight: 700; margin-right: 8px;}
.ns-step b {font-size: 1rem;}
.ns-step p {color: #475569; font-size: 0.9rem; margin: 8px 0 0 0;}
.ns-banner {border-radius: 14px; padding: 18px 22px; margin: 6px 0 12px 0; font-size: 1.05rem; border: 1px solid;}
.ns-ok {background: #ECFDF5; border-color: #A7F3D0; color: #065F46;}
.ns-warn {background: #FFF7ED; border-color: #FED7AA; color: #9A3412;}
.ns-bad {background: #FEF2F2; border-color: #FECACA; color: #991B1B;}
.ns-banner h3 {margin: 0 0 4px 0; font-size: 1.25rem; color: inherit;}
.ns-card {background: white; border: 1px solid #E2E8F0; border-radius: 14px; padding: 18px 20px; margin-bottom: 12px;}
.ns-card h4 {margin: 0 0 10px 0; color: #0F172A;}
.ns-legend span {display: inline-block; width: 12px; height: 12px; border-radius: 3px; margin: 0 6px 0 14px; vertical-align: middle;}
.ns-note {font-size: 0.85rem; color: #64748B;}
.ns-foot {font-size: 0.8rem; color: #64748B; border-top: 1px solid #E2E8F0; padding-top: 10px; margin-top: 24px;}
div[data-testid="stMetric"] {background: white; border: 1px solid #E2E8F0; border-radius: 12px; padding: 12px 14px;}
.stButton > button[kind="primary"] {border-radius: 10px; font-weight: 600; padding: 0.6rem 1.2rem;}
.stDownloadButton > button {border-radius: 10px;}
</style>
"""


# ------------------------------------------------------------------------------------------------
# Cached loaders
# ------------------------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_model(ckpt_path: str, device: str):
    model = load_checkpoint(ckpt_path, device)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return model, {"epoch": ckpt.get("epoch"), "val": ckpt.get("val"), "config": ckpt.get("full_config", ckpt.get("config"))}


@st.cache_data(show_spinner=False)
def load_json(path: str) -> Optional[Dict]:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


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
def run_demo_inference(demo_dir: str, mc: int, ckpt_path: str):
    """Demo case prediction is cached: first run is slow on CPU, every later run is instant."""
    vols, affine = load_demo_case(demo_dir)
    return _run_case(vols, affine, mc, ckpt_path)


def _run_case(vols, affine, mc: int, ckpt_path: str):
    model, _ = get_model(ckpt_path, DEVICE)
    present = [m for m in MODALITIES if m in vols]
    quality = assess_case({m: vols.get(m) for m in MODALITIES})
    img, seg, meta = preprocess_case(vols, available=present)
    t0 = time.time()
    voxel_mm3 = float(abs(np.linalg.det(affine[:3, :3]))) or 1.0
    pred = predict_volume(model, img, mc_samples=mc, device=DEVICE, batch_size=32, voxel_volume_mm3=voxel_mm3)
    infer_s = time.time() - t0
    metrics = region_metrics(pred.regions, labels_to_regions(seg), with_hd95=False) if seg is not None else None
    return dict(img=img, seg=seg, meta=meta, pred=pred, infer_s=infer_s, present=present, quality=quality,
                metrics=metrics, affine=affine)


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
# Figures
# ------------------------------------------------------------------------------------------------
def overlay_figure(base: np.ndarray, labels: np.ndarray, entropy: np.ndarray, z: int, simple: bool):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), facecolor="white")
    sl = base[z]
    nz = sl[sl != 0]
    vmin, vmax = np.percentile(nz, [1, 99]) if nz.size else (0, 1)
    titles = ["MRI scan", "Tumour found by the AI", "Where the AI is unsure"] if simple else \
             ["Input slice", "Predicted segmentation", "Predictive entropy"]
    for ax, t in zip(axes, titles):
        ax.imshow(sl.T, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
        ax.set_title(t, fontsize=12, color="#0F172A", pad=8)
        ax.axis("off")
    rgba = np.zeros(sl.shape + (4,), dtype=np.float32)
    for lab, col in LABEL_COLORS.items():
        m = labels[z] == lab
        rgba[m, :3] = col
        rgba[m, 3] = 0.55
    axes[1].imshow(np.transpose(rgba, (1, 0, 2)), origin="lower")
    ent = entropy[z].copy()
    ent[sl == 0] = 0
    axes[2].imshow(ent.T, cmap="magma", alpha=0.75, vmin=0, vmax=float(np.log(2)), origin="lower")
    fig.tight_layout()
    return fig


def plotly_3d(regions: np.ndarray, step: int = 2):
    import plotly.graph_objects as go
    fig = go.Figure()
    names = ["Whole tumour (incl. swelling)", "Tumour core", "Active tumour"]
    colors = ["#2E8B57", "#DC2626", "#F59E0B"]
    for i in range(3):
        v = regions[i][::step, ::step, ::step].astype(np.float32)
        if v.sum() == 0:
            continue
        D, H, W = v.shape
        zz, yy, xx = np.mgrid[0:D, 0:H, 0:W]
        fig.add_trace(go.Isosurface(x=xx.flatten(), y=yy.flatten(), z=zz.flatten(), value=v.flatten(),
                                    isomin=0.5, isomax=1.0, surface_count=1, opacity=0.3 if i == 0 else 0.65,
                                    colorscale=[[0, colors[i]], [1, colors[i]]], showscale=False, name=names[i],
                                    caps=dict(x_show=False, y_show=False, z_show=False), showlegend=True))
    fig.update_layout(height=520, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="white",
                      scene=dict(aspectmode="data", xaxis_visible=False, yaxis_visible=False, zaxis_visible=False),
                      legend=dict(orientation="h", y=0.02, x=0.5, xanchor="center"))
    return fig


# ------------------------------------------------------------------------------------------------
# Report
# ------------------------------------------------------------------------------------------------
def build_report(res, flagged, thr, model_info) -> Dict:
    pred, quality = res["pred"], res["quality"]
    return {
        "tool": "NeuroScope research prototype — NOT a medical device; outputs require expert review",
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
        "model": {k: model_info.get(k) for k in ["epoch", "val"]},
        "metrics_vs_ground_truth": res["metrics"],
    }


def report_markdown(r: Dict) -> str:
    v = r["tumour_volumes_ml"]
    lines = ["# NeuroScope analysis report", "", f"_{r['tool']}_", "", f"Generated: {r['timestamp']}", "",
             "## Summary in plain language",
             f"The AI found a tumour region of about **{v['WT_ml']:.0f} mL** (including swelling). "
             f"The solid core is about **{v['TC_ml']:.0f} mL**, of which **{v['ET_ml']:.0f} mL** is active (enhancing) tumour. "
             f"{'The AI was **not confident** on this scan, so a specialist must check it carefully.' if r['needs_human_review'] else 'The AI was confident on this scan, but a specialist must still confirm the result.'}",
             "", "## Input", f"- Scans used: {', '.join(m.upper() for m in r['modalities_used'])}",
             f"- Missing: {', '.join(m.upper() for m in r['modalities_missing']) or 'none'}",
             f"- Quality status: **{r['input_quality']['status']}** (score {r['input_quality']['score']:.2f})"]
    for c in r["input_quality"]["checks"]:
        if c["status"] != "pass":
            lines.append(f"  - [{c['modality']}] {c['message']}")
    lines += ["", "## Segmentation volumes", f"- Whole tumour (WT): {v['WT_ml']:.1f} mL",
              f"- Tumour core (TC): {v['TC_ml']:.1f} mL", f"- Enhancing tumour (ET): {v['ET_ml']:.1f} mL",
              "", "## Reliability", f"- Case uncertainty (mean predictive entropy in ROI): {r['case_uncertainty']:.4f}",
              f"- Monte Carlo dropout samples: {r['mc_samples']}",
              f"- Inference: {r['inference_seconds']:.1f} s on {r['device']}",
              f"- **Needs human review: {'YES' if r['needs_human_review'] else 'no'}**"]
    if r["review_threshold"]:
        t = r["review_threshold"]
        lines.append(f"- Review threshold {t['threshold']:.4f} chosen on the validation set (rule: {t.get('rule')}, failure = WT Dice < {t.get('tau_fail')})")
    if r.get("metrics_vs_ground_truth"):
        m = r["metrics_vs_ground_truth"]
        lines += ["", "## Agreement with provided ground truth", f"- Dice WT {m['dice_WT']:.3f} | TC {m['dice_TC']:.3f} | ET {m['dice_ET']:.3f}"]
    lines += ["", "## Limitations", "- 2D slice-wise model; trained on BraTS 2020 adult glioma only.",
              "- Uncertainty reflects model disagreement, not clinical reliability.",
              "- Quality checks are heuristics; passing them does not certify diagnostic suitability."]
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------
# Small UI helpers
# ------------------------------------------------------------------------------------------------
def stat(col, label, value, sub=""):
    col.markdown(f'<div class="ns-stat"><div class="k">{label}</div><div class="v">{value}</div><div class="s">{sub}</div></div>',
                 unsafe_allow_html=True)


def fmt_ci(d):
    if not d:
        return None, ""
    lo, hi = d.get("ci95", [None, None])
    return d["mean"], (f"95% CI {lo:.2f}–{hi:.2f}" if lo is not None else "")


def plain_size(ml: float) -> str:
    pct = 100.0 * ml / BRAIN_ML
    return f"{ml:.0f} mL (about {pct:.1f}% of a typical adult brain)"


# ================================================================================================
# Page
# ================================================================================================
st.set_page_config(page_title="NeuroScope — Brain tumour MRI analysis", page_icon="🧠", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown(CSS, unsafe_allow_html=True)

summary = load_json(os.path.join(ARGS.results, "summary.json"))
thr = load_json(os.path.join(ARGS.results, "threshold.json"))

# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("### 🧠 NeuroScope")
    mode = st.radio("I am a…", ["Patient / first-time user", "Clinician / researcher"], index=0,
                    help="Changes how much technical detail is shown. The analysis is identical.")
    simple = mode.startswith("Patient")
    st.divider()

    st.markdown("#### 1 · Choose the scans")
    has_demo = os.path.isdir(ARGS.demo_dir)
    use_demo = st.button("▶ Try the example patient", type="primary", width="stretch",
                         disabled=not has_demo, help="A real anonymised case from the public BraTS 2020 test split")
    st.caption("…or upload your own MRI files (.nii / .nii.gz). Any subset of the four scans works.")
    uploads = {m: st.file_uploader(MOD_HELP[m], type=["nii", "gz"], key=m) for m in MODALITIES}
    seg_upload = None
    if not simple:
        seg_upload = st.file_uploader("Ground-truth mask (optional, for evaluation)", type=["nii", "gz"], key="seg")

    st.markdown("#### 2 · Run")
    mc_default = 4 if DEVICE == "cpu" else 8
    if simple:
        mc = mc_default
        st.caption(f"Running on **{DEVICE.upper()}**. The AI checks each scan {mc} times to measure its own confidence.")
    else:
        mc = st.slider("Monte Carlo dropout samples", 1, 16, mc_default,
                       help="More samples → better uncertainty estimate, slower. Validated at 8.")
        st.caption(f"Device: `{DEVICE}`")
    run = st.button("Analyse uploaded scans", width="stretch")
    show_3d = st.checkbox("Show 3D tumour view", value=True)

    with st.expander("Advanced"):
        ckpt_path = st.text_input("Checkpoint", ARGS.ckpt)
        results_dir = st.text_input("Results dir", ARGS.results)
        if results_dir != ARGS.results:
            summary = load_json(os.path.join(results_dir, "summary.json"))
            thr = load_json(os.path.join(results_dir, "threshold.json"))

# ---------------- Hero ----------------
pills = ["Works even if some scans are missing", "Tells you when it is unsure", "Checks scan quality before analysing",
         "Research prototype · not a medical device"]
st.markdown(
    '<div class="ns-hero"><h1>NeuroScope</h1>'
    '<p>AI that outlines a brain tumour on an MRI in seconds, measures its size, and — unlike most AI tools — '
    '<b>tells you how sure it is</b>, so a doctor knows which results to double-check.</p>'
    + "".join(f'<span class="ns-pill">{p}</span>' for p in pills) + "</div>",
    unsafe_allow_html=True)

# ---------------- Validated performance strip ----------------
if summary:
    n_test = summary.get("n_test", "?")
    dwt, dwt_ci = fmt_ci(summary.get("dice_WT"))
    dtc, _ = fmt_ci(summary.get("dice_TC"))
    det, _ = fmt_ci(summary.get("dice_ET"))
    rel = summary.get("reliability", {})
    flag = summary.get("review_flag", {})
    inf, _ = fmt_ci(summary.get("infer_s"))
    c = st.columns(4)
    if simple:
        stat(c[0], "Accuracy on unseen patients", f"{100*dwt:.0f}%", f"Overlap with expert outlines, tested on {n_test} patients it never saw")
        stat(c[1], "Catches its own mistakes", f"{100*rel.get('failure_auroc', float('nan')):.0f}%",
             f"How reliably it flags its weak results for a doctor ({flag.get('flag_sensitivity', float('nan'))*100:.0f}% of failures caught)")
        stat(c[2], "Incomplete scans", "15 / 15", "Every combination of the 4 MRI types still produces a result")
        stat(c[3], "Speed per patient", f"{inf:.1f} s" if inf else "—", "On a standard GPU, incl. the confidence check")
    else:
        stat(c[0], "Test Dice · WT / TC / ET", f"{dwt:.3f} / {dtc:.3f} / {det:.3f}", f"n = {n_test} held-out patients · {dwt_ci}")
        stat(c[1], "Failure-detection AUROC", f"{rel.get('failure_auroc', float('nan')):.3f}",
             f"Spearman ρ(uncertainty, Dice) = {rel.get('spearman_rho', float('nan')):.2f} · AURC {rel.get('aurc', float('nan')):.3f}")
        stat(c[2], "Review flag", f"{100*flag.get('flag_rate', float('nan')):.0f}% flagged",
             f"Sens {flag.get('flag_sensitivity', float('nan')):.2f} · Spec {flag.get('flag_specificity', float('nan')):.2f} · "
             f"Dice accepted {flag.get('dice_accepted', float('nan')):.3f}")
        stat(c[3], "Inference", f"{inf:.1f} s / case" if inf else "—", f"MC ×{summary.get('mc_samples')} on GPU · 7.8 M params")
    st.markdown("")

# ---------------- Load model ----------------
if not os.path.exists(ckpt_path):
    st.error(f"Model file not found: {ckpt_path}. Train first (scripts/train.py) or pass --ckpt.")
    st.stop()
model, model_info = get_model(ckpt_path, DEVICE)

# ---------------- Case selection ----------------
if "result" not in st.session_state:
    st.session_state.result = None
    st.session_state.source = None

if use_demo:
    with st.status("Analysing the example patient…", expanded=True) as status:
        st.write("Loading the four MRI scans")
        st.write("Checking scan quality")
        st.write(f"Running the AI {mc} times to measure confidence (first run on CPU can take 1–2 min; later runs are instant)")
        st.session_state.result = run_demo_inference(ARGS.demo_dir, mc, ckpt_path)
        st.session_state.source = "Example patient (BraTS 2020, held-out test split)"
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
        st.warning("Please upload at least one MRI scan in the sidebar, or try the example patient.")
    else:
        q = assess_case({m: vols.get(m) for m in MODALITIES})
        if q.status == "fail":
            st.error("These scans did not pass the quality check, so the AI will not analyse them: " + "; ".join(q.warnings()))
            st.stop()
        with st.status("Analysing your scans…", expanded=True) as status:
            st.write("Checking scan quality")
            st.write(f"Running the AI {mc} times to measure confidence")
            st.session_state.result = _run_case(vols, affine, mc, ckpt_path)
            st.session_state.source = "Uploaded scans"
            status.update(label="Analysis complete", state="complete", expanded=False)

res = st.session_state.result

# ---------------- Empty state: how it works ----------------
if res is None:
    st.markdown("#### How it works")
    s = st.columns(3)
    steps = [("Choose the scans", "Click <b>Try the example patient</b> in the left panel, or upload MRI files. "
                                  "Don't have all four scan types? That's fine — NeuroScope was trained to cope with missing ones."),
             ("The AI analyses them", "It first checks the scans are usable, then outlines the tumour and runs itself several "
                                       "times to see how much its answers disagree. Big disagreement = low confidence."),
             ("Read the result", "You get the tumour's size, a picture of where it is, and a clear verdict: "
                                 "<b>confident</b> or <b>needs a specialist's review</b>. Download a report to share.")]
    for col, (i, (t, d)) in zip(s, enumerate(steps, 1)):
        col.markdown(f'<div class="ns-step"><span class="n">{i}</span><b>{t}</b><p>{d}</p></div>', unsafe_allow_html=True)
    st.markdown("")
    with st.expander("What do the words mean?  (plain-language glossary)"):
        st.markdown("""
- **MRI** — a scan that takes detailed pictures of the brain without radiation. Doctors usually take four *types* (T1, T1ce, T2, FLAIR); each highlights different tissue.
- **Segmentation** — drawing an outline around the tumour on every slice of the scan. Done by hand it takes a radiologist 20–60 minutes; NeuroScope does it in seconds.
- **Whole tumour / core / active tumour** — three nested regions: everything abnormal including swelling; the solid centre; and the part that lights up with contrast dye (usually the most aggressive).
- **Uncertainty / confidence** — NeuroScope runs the AI several times with small random changes. If the answers agree, it is confident. If they disagree, it says so and asks for a human to check.
- **Dice score** — how well two outlines overlap, from 0 (no overlap) to 1 (identical). Expert radiologists typically agree with each other at about 0.85–0.90.
        """)
    st.markdown('<p class="ns-foot">NeuroScope is a research and decision-support prototype built on the public BraTS 2020 dataset. '
                'It is not a medical device and must not be used to make a diagnosis or treatment decision on its own.</p>',
                unsafe_allow_html=True)

# ---------------- Results ----------------
else:
    pred, img, quality, present, metrics = res["pred"], res["img"], res["quality"], res["present"], res["metrics"]
    v = pred.volumes_ml
    flagged = bool(thr and pred.case_uncertainty >= thr["threshold"])
    missing = [m for m in MODALITIES if m not in present]

    st.markdown(f"#### Result · <span class='ns-note'>{st.session_state.source}</span>", unsafe_allow_html=True)

    # verdict banner
    if flagged:
        cls, head = "ns-bad", "⚠️ Needs a specialist's review"
        body = ("The AI's repeated answers disagreed more than usual on this scan, so its outline may be unreliable. "
                "A radiologist should check this case carefully before anything is concluded.") if simple else \
               (f"Case uncertainty {pred.case_uncertainty:.4f} ≥ validated review threshold {thr['threshold']:.4f} "
                f"(chosen on validation via Youden's J to detect WT Dice < {thr.get('tau_fail')}).")
    elif thr:
        cls, head = "ns-ok", "✅ The AI is confident in this result"
        body = ("Its repeated answers agreed closely. A doctor should still confirm it — this tool supports decisions, it does not make them.") if simple else \
               (f"Case uncertainty {pred.case_uncertainty:.4f} < validated review threshold {thr['threshold']:.4f}. "
                f"On the held-out test set, accepted cases averaged Dice {summary['review_flag']['dice_accepted']:.3f} (WT)." if summary else
                f"Case uncertainty {pred.case_uncertainty:.4f} < review threshold {thr['threshold']:.4f}.")
    else:
        cls, head, body = "ns-warn", "No validated threshold found", "Run scripts/evaluate.py to enable the review flag."
    st.markdown(f'<div class="ns-banner {cls}"><h3>{head}</h3>{body}</div>', unsafe_allow_html=True)

    # plain-language summary + key numbers
    left, right = st.columns([1.15, 1])
    with left:
        if simple:
            st.markdown(f"""<div class="ns-card"><h4>In plain words</h4>
            The AI found an abnormal region of about <b>{plain_size(v['WT_ml'])}</b>, which includes swelling around the tumour.
            The solid tumour itself is about <b>{v['TC_ml']:.0f} mL</b>, and <b>{v['ET_ml']:.0f} mL</b> of that is
            <i>active</i> tumour that lights up with contrast dye.<br><br>
            <span class="ns-note">Volumes are estimates from the AI outline. 1 mL = 1 cm³, roughly a sugar cube.</span></div>""",
                        unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="ns-card"><h4>Case summary</h4>
            <b>Modalities:</b> {', '.join(m.upper() for m in present)}{' · <b>missing:</b> ' + ', '.join(m.upper() for m in missing) if missing else ''}<br>
            <b>Input quality:</b> {quality.status} (score {quality.score:.2f})<br>
            <b>MC samples:</b> {pred.mc_samples} · <b>inference:</b> {res['infer_s']:.1f} s on {DEVICE}<br>
            <b>Case uncertainty:</b> {pred.case_uncertainty:.4f} (mean predictive entropy in tumour ROI)</div>""",
                        unsafe_allow_html=True)
    with right:
        k = st.columns(3)
        k[0].metric("Whole tumour" if simple else "WT", f"{v['WT_ml']:.1f} mL", help="Everything abnormal, including swelling")
        k[1].metric("Tumour core" if simple else "TC", f"{v['TC_ml']:.1f} mL", help="Solid tumour without the swelling")
        k[2].metric("Active tumour" if simple else "ET", f"{v['ET_ml']:.1f} mL", help="Enhancing part — takes up contrast dye")
        if metrics:
            st.markdown(f'<div class="ns-banner ns-ok" style="padding:10px 14px;font-size:0.95rem;margin-top:8px">'
                        f'Matches the expert outline: <b>{100*metrics["dice_WT"]:.0f}%</b> whole tumour · '
                        f'<b>{100*metrics["dice_TC"]:.0f}%</b> core · <b>{100*metrics["dice_ET"]:.0f}%</b> active (Dice)</div>',
                        unsafe_allow_html=True)

    if missing:
        st.warning(f"Running without {', '.join(m.upper() for m in missing)}. The AI still works, but accuracy for the affected regions is lower (see the model card).")
    if quality.status == "warn":
        with st.expander(f"Scan quality: usable, with {len(quality.warnings())} note(s)"):
            for w in quality.warnings():
                st.write("- " + w)

    # viewer
    st.markdown("#### Where is the tumour?")
    cA, cB = st.columns([1, 3])
    base_mod = cA.selectbox("Background scan", present, index=present.index("flair") if "flair" in present else 0,
                            format_func=lambda m: MOD_HELP[m].split(" — ")[0])
    tum_z = np.where(pred.regions[0].reshape(pred.regions.shape[1], -1).sum(1) > 0)[0]
    default_z = int(tum_z[len(tum_z) // 2]) if tum_z.size else img.shape[1] // 2
    z = cB.slider("Slice (drag to move up and down through the head)", 0, img.shape[1] - 1, default_z)
    st.pyplot(overlay_figure(img[MODALITIES.index(base_mod)], pred.labels, pred.entropy, z, simple), clear_figure=True)
    st.markdown('<div class="ns-legend ns-note">Colours: <span style="background:#DC2626"></span>dead / non-enhancing core '
                '<span style="background:#28A65A"></span>swelling (edema) <span style="background:#FABE24"></span>active tumour &nbsp;·&nbsp; '
                'Right panel: brighter = AI less sure (usually the tumour edge).</div>', unsafe_allow_html=True)

    if show_3d and pred.regions[0].sum() > 0:
        st.markdown("#### 3D view  <span class='ns-note'>drag to rotate · scroll to zoom</span>", unsafe_allow_html=True)
        st.plotly_chart(plotly_3d(pred.regions), width="stretch")

    # downloads
    st.markdown("#### Share the result")
    report = build_report(res, flagged, thr, model_info)
    md = report_markdown(report)
    d1, d2, d3 = st.columns(3)
    d1.download_button("📄 Report (readable, .md)", md, file_name="neuroscope_report.md", width="stretch")
    d2.download_button("🧾 Report (machine-readable, .json)", json.dumps(report, indent=2, default=float),
                       file_name="neuroscope_report.json", width="stretch")
    lab_full = uncrop(pred.labels, res["meta"]).astype(np.uint8)
    lab_full[lab_full == 3] = 4
    d3.download_button("🧠 Tumour outline (.nii, opens in any MRI viewer)", nib.Nifti1Image(lab_full, res["affine"]).to_bytes(),
                       file_name="neuroscope_seg.nii", width="stretch")
    with st.expander("Preview the report"):
        st.markdown(md)

    if st.button("↺ Start over"):
        st.session_state.result = None
        st.rerun()

# ---------------- Model card ----------------
st.markdown("")
with st.expander("📊 How well does it work? — validated performance & model card", expanded=False):
    if summary:
        if simple:
            st.markdown(f"""
**Tested on {summary.get('n_test')} patients the AI had never seen.** Its outlines overlapped expert outlines by
**{100*summary['dice_WT']['mean']:.0f}%** (whole tumour), **{100*summary['dice_TC']['mean']:.0f}%** (core) and
**{100*summary['dice_ET']['mean']:.0f}%** (active tumour). When its outline was poor, its own confidence score spotted that
with **{100*summary['reliability']['failure_auroc']:.0f}%** reliability, which is what lets it ask for a human review at the right time.
            """)
        else:
            st.json({k: summary[k] for k in ["n_test", "n_val", "mc_samples", "dice_WT", "dice_TC", "dice_ET", "hd95_WT",
                                             "sensitivity_WT", "specificity_WT", "reliability", "review_flag"] if k in summary})
            st.write({"checkpoint_epoch": model_info.get("epoch"), "val_at_checkpoint": model_info.get("val")})
        sub_csv = os.path.join(results_dir, "subsets.csv")
        if os.path.exists(sub_csv):
            import pandas as pd
            st.markdown("**Accuracy with missing scans** — every combination of the four MRI types (Dice on the test set):")
            df = pd.read_csv(sub_csv)[["subset", "n_modalities", "dice_WT", "dice_TC", "dice_ET", "uncertainty", "flag_rate"]].round(3)
            st.dataframe(df, width="stretch", hide_index=True)
        figs = [("fig_subsets.png", "Dice per modality subset"), ("fig_uncertainty_vs_dice.png", "Uncertainty vs Dice (per test patient)"),
                ("fig_risk_coverage.png", "Risk–coverage curve")]
        cols = st.columns(3)
        for col, (f, cap) in zip(cols, figs):
            p = os.path.join(results_dir, f)
            if os.path.exists(p):
                col.image(p, caption=cap)
    else:
        st.write("No results/summary.json found — run scripts/evaluate.py.")
    st.markdown("""
**Model.** 2D U-Net (7.8 M parameters), trained with *modality dropout* so it learns to work with any subset of the four MRI sequences.
Confidence comes from Monte Carlo dropout: the network is run several times and the disagreement (predictive entropy) is measured.
The review threshold was fixed on the validation set and never tuned on the test set.

**Limitations.** Trained on BraTS 2020 (adult glioma, skull-stripped, co-registered, 1 mm isotropic). Not validated on other
tumour types, scanners, children, or post-operative scans. Uncertainty measures model disagreement, not correctness.
This is a research prototype for human-in-the-loop review, **not a diagnostic device**.
    """)
st.markdown('<p class="ns-foot">NeuroScope · built on public BraTS 2020 data · research & decision-support prototype, not a medical device.</p>',
            unsafe_allow_html=True)
