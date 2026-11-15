"""NeuroScope - brain tumour MRI analysis with built-in self-checks (research prototype).

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
    "t1": "T1 - basic anatomy scan",
    "t1ce": "T1ce - T1 with contrast dye (shows active tumour)",
    "t2": "T2 - fluid-sensitive scan",
    "flair": "FLAIR - shows swelling around the tumour",
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
.block-container {padding-top: 1.5rem; max-width: 1200px;}
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
    return d["mean"], (f"95% CI {lo:.2f}-{hi:.2f}" if lo is not None else "")


def plain_size(ml: float) -> str:
    pct = 100.0 * ml / BRAIN_ML
    return f"{ml:.0f} mL (about {pct:.1f}% of a typical adult brain)"


# ================================================================================================
# Page
# ================================================================================================
st.set_page_config(page_title="NeuroScope - Brain tumour MRI analysis", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown(CSS, unsafe_allow_html=True)

summary = load_json(os.path.join(ARGS.results, "summary.json"))
thr = load_json(os.path.join(ARGS.results, "threshold.json"))

# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("### NeuroScope")
    mode = st.radio("View", ["Simple", "Detailed"], index=0, help="Simple hides the technical numbers. The analysis is the same.")
    simple = mode == "Simple"
    st.divider()

    st.markdown("**Step 1. Choose scans**")
    has_demo = os.path.isdir(ARGS.demo_dir)
    use_demo = st.button("Load example patient", type="primary", width="stretch",
                         disabled=not has_demo, help="An anonymised case from the BraTS 2020 test split")
    st.caption("Or upload MRI files (.nii / .nii.gz). Any subset of the four scans works.")
    uploads = {m: st.file_uploader(MOD_HELP[m], type=["nii", "gz"], key=m) for m in MODALITIES}
    seg_upload = None
    if not simple:
        seg_upload = st.file_uploader("Ground-truth mask (optional, for evaluation)", type=["nii", "gz"], key="seg")

    st.markdown("**Step 2. Run**")
    mc_default = 4 if DEVICE == "cpu" else 8
    if simple:
        mc = mc_default
        st.caption(f"Device: {DEVICE}. The model is run {mc} times per scan to estimate confidence.")
    else:
        mc = st.slider("Monte Carlo dropout samples", 1, 16, mc_default,
                       help="More samples give a better uncertainty estimate, slower. Validated at 8.")
        st.caption(f"Device: `{DEVICE}`")
    run = st.button("Analyse uploaded scans", width="stretch")
    show_3d = st.checkbox("3D view", value=True)

    with st.expander("Advanced"):
        ckpt_path = st.text_input("Checkpoint", ARGS.ckpt)
        results_dir = st.text_input("Results dir", ARGS.results)
        if results_dir != ARGS.results:
            summary = load_json(os.path.join(results_dir, "summary.json"))
            thr = load_json(os.path.join(results_dir, "threshold.json"))

# ---------------- Header ----------------
st.title("NeuroScope")
st.markdown("Brain tumour segmentation from MRI, with an uncertainty estimate that says when the result should be checked by a specialist. "
            "Works with any subset of the four standard MRI sequences. Research prototype, not a medical device.")
st.markdown("")

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
        c[0].metric("Overlap with expert outlines", f"{100*dwt:.0f}%", help=f"Dice score, whole tumour, on {n_test} test patients the model never saw")
        c[1].metric("Weak results detected", f"{100*rel.get('failure_auroc', float('nan')):.0f}%", help="How well the confidence score separates good from poor outlines (AUROC)")
        c[2].metric("Scan combinations supported", "15 of 15", help="Any subset of T1, T1ce, T2, FLAIR")
        c[3].metric("Time per patient (GPU)", f"{inf:.1f} s" if inf else "n/a", help="Including the repeated runs for the confidence estimate")
        st.caption(f"Measured on {n_test} held-out BraTS 2020 patients. Details under 'Validated performance' at the bottom of the page.")
    else:
        c[0].metric("Test Dice WT / TC / ET", f"{dwt:.3f} / {dtc:.3f} / {det:.3f}", help=f"n = {n_test}, {dwt_ci} (WT)")
        c[1].metric("Failure-detection AUROC", f"{rel.get('failure_auroc', float('nan')):.3f}",
                    help=f"Spearman rho(uncertainty, Dice) = {rel.get('spearman_rho', float('nan')):.2f}, AURC {rel.get('aurc', float('nan')):.3f}")
        c[2].metric("Review flag rate", f"{100*flag.get('flag_rate', float('nan')):.0f}%",
                    help=f"Sensitivity {flag.get('flag_sensitivity', float('nan')):.2f}, specificity {flag.get('flag_specificity', float('nan')):.2f}, Dice on accepted cases {flag.get('dice_accepted', float('nan')):.3f}")
        c[3].metric("Inference per case (GPU)", f"{inf:.1f} s" if inf else "n/a", help=f"MC x{summary.get('mc_samples')}, 7.8M parameters")
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
    with st.status("Analysing example patient", expanded=True) as status:
        st.write("Loading scans and checking quality")
        st.write(f"Running the model {mc} times (first run on CPU takes 1-2 min; later runs are cached)")
        st.session_state.result = run_demo_inference(ARGS.demo_dir, mc, ckpt_path)
        st.session_state.source = "Example patient (BraTS 2020 test split)"
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
        st.warning("Upload at least one MRI scan, or load the example patient.")
    else:
        q = assess_case({m: vols.get(m) for m in MODALITIES})
        if q.status == "fail":
            st.error("Scans failed the quality check: " + "; ".join(q.warnings()))
            st.stop()
        with st.status("Analysing scans", expanded=True) as status:
            st.write(f"Checking quality, then running the model {mc} times")
            st.session_state.result = _run_case(vols, affine, mc, ckpt_path)
            st.session_state.source = "Uploaded scans"
            status.update(label="Analysis complete", state="complete", expanded=False)

res = st.session_state.result

# ---------------- Empty state: how it works ----------------
if res is None:
    st.markdown("#### How to use")
    st.markdown("""
1. **Choose scans** - click *Load example patient* in the sidebar, or upload your own MRI files. Missing sequences are allowed.
2. **Run** - the app checks scan quality, segments the tumour, and runs the model several times to measure how much its answers vary.
3. **Read the result** - tumour volumes, a slice viewer, a 3D view, and a verdict: *confident* or *needs specialist review*. A report can be downloaded.
    """)
    with st.expander("Glossary"):
        st.markdown("""
- **MRI sequences (T1, T1ce, T2, FLAIR)** - four scan types taken in the same session; each shows different tissue. T1ce uses contrast dye and shows active tumour; FLAIR shows swelling.
- **Segmentation** - outlining the tumour on every slice. Done manually this takes a radiologist roughly 20-60 minutes.
- **Whole tumour / tumour core / enhancing tumour** - everything abnormal including swelling; the solid centre; the part that takes up contrast dye.
- **Uncertainty** - the model is run several times with dropout switched on. If the outputs disagree, the case is flagged for review. The flag threshold was fixed on validation data.
- **Dice score** - overlap between two outlines, 0 to 1. Radiologists agree with each other at roughly 0.85-0.90.
        """)
    st.caption("Built on the public BraTS 2020 dataset. Not a medical device; not for diagnosis or treatment decisions.")

# ---------------- Results ----------------
else:
    pred, img, quality, present, metrics = res["pred"], res["img"], res["quality"], res["present"], res["metrics"]
    v = pred.volumes_ml
    flagged = bool(thr and pred.case_uncertainty >= thr["threshold"])
    missing = [m for m in MODALITIES if m not in present]

    st.markdown(f"#### Result")
    st.caption(st.session_state.source)

    # verdict
    if flagged:
        if simple:
            st.error("Needs specialist review. The model's repeated outputs disagreed more than usual on this scan, so the outline may be unreliable.")
        else:
            st.error(f"Needs review: case uncertainty {pred.case_uncertainty:.4f} >= validated threshold {thr['threshold']:.4f} "
                     f"(set on validation data via Youden's J to detect WT Dice < {thr.get('tau_fail')}).")
    elif thr:
        if simple:
            st.success("Confident result. The model's repeated outputs agreed closely. A specialist should still confirm it.")
        else:
            extra = f" On the test set, accepted cases averaged WT Dice {summary['review_flag']['dice_accepted']:.3f}." if summary else ""
            st.success(f"Below review threshold: case uncertainty {pred.case_uncertainty:.4f} < {thr['threshold']:.4f}.{extra}")
    else:
        st.warning("No validated threshold found. Run scripts/evaluate.py to enable the review flag.")

    # plain-language summary + key numbers
    left, right = st.columns([1.15, 1])
    with left:
        if simple:
            st.markdown(f"""**Summary**

The model found an abnormal region of about **{plain_size(v['WT_ml'])}**, including swelling around the tumour.
The solid tumour is about **{v['TC_ml']:.0f} mL**, of which **{v['ET_ml']:.0f} mL** is enhancing (active) tumour.

Volumes are estimates from the model outline. 1 mL = 1 cm3.""")
        else:
            st.markdown(f"""**Case summary**

- Modalities: {', '.join(m.upper() for m in present)}{'; missing: ' + ', '.join(m.upper() for m in missing) if missing else ''}
- Input quality: {quality.status} (score {quality.score:.2f})
- MC samples: {pred.mc_samples}; inference {res['infer_s']:.1f} s on {DEVICE}
- Case uncertainty: {pred.case_uncertainty:.4f} (mean predictive entropy in tumour ROI)""")
    with right:
        k = st.columns(3)
        k[0].metric("Whole tumour" if simple else "WT", f"{v['WT_ml']:.1f} mL", help="Everything abnormal, including swelling")
        k[1].metric("Tumour core" if simple else "TC", f"{v['TC_ml']:.1f} mL", help="Solid tumour without the swelling")
        k[2].metric("Active tumour" if simple else "ET", f"{v['ET_ml']:.1f} mL", help="Enhancing part - takes up contrast dye")
        if metrics:
            st.info(f"Agreement with expert outline (Dice): whole tumour {metrics['dice_WT']:.3f}, core {metrics['dice_TC']:.3f}, enhancing {metrics['dice_ET']:.3f}")

    if missing:
        st.warning(f"Running without {', '.join(m.upper() for m in missing)}. Accuracy for the affected regions is lower; see the subset table under validated performance.")
    if quality.status == "warn":
        with st.expander(f"Scan quality: {len(quality.warnings())} warning(s)"):
            for w in quality.warnings():
                st.write("- " + w)

    # viewer
    st.markdown("#### Slice viewer")
    cA, cB = st.columns([1, 3])
    base_mod = cA.selectbox("Background scan", present, index=present.index("flair") if "flair" in present else 0,
                            format_func=lambda m: MOD_HELP[m].split(" - ")[0])
    tum_z = np.where(pred.regions[0].reshape(pred.regions.shape[1], -1).sum(1) > 0)[0]
    default_z = int(tum_z[len(tum_z) // 2]) if tum_z.size else img.shape[1] // 2
    z = cB.slider("Axial slice", 0, img.shape[1] - 1, default_z)
    st.pyplot(overlay_figure(img[MODALITIES.index(base_mod)], pred.labels, pred.entropy, z, simple), clear_figure=True)
    st.caption("Red: necrotic / non-enhancing core. Green: edema (swelling). Yellow: enhancing tumour. Right panel: brighter = less certain, usually at the tumour edge.")

    if show_3d and pred.regions[0].sum() > 0:
        st.markdown("#### 3D view")
        st.caption("Drag to rotate, scroll to zoom.")
        st.plotly_chart(plotly_3d(pred.regions), width="stretch")

    # downloads
    st.markdown("#### Download")
    report = build_report(res, flagged, thr, model_info)
    md = report_markdown(report)
    d1, d2, d3 = st.columns(3)
    d1.download_button("Report (.md)", md, file_name="neuroscope_report.md", width="stretch")
    d2.download_button("Report (.json)", json.dumps(report, indent=2, default=float),
                       file_name="neuroscope_report.json", width="stretch")
    lab_full = uncrop(pred.labels, res["meta"]).astype(np.uint8)
    lab_full[lab_full == 3] = 4
    d3.download_button("Segmentation mask (.nii)", nib.Nifti1Image(lab_full, res["affine"]).to_bytes(),
                       file_name="neuroscope_seg.nii", width="stretch")
    with st.expander("Report preview"):
        st.markdown(md)

    if st.button("Clear"):
        st.session_state.result = None
        st.rerun()

# ---------------- Model card ----------------
st.markdown("")
with st.expander("Validated performance and model card", expanded=False):
    if summary:
        if simple:
            st.markdown(f"""
Tested on {summary.get('n_test')} patients held out from training. Overlap with expert outlines (Dice):
whole tumour {summary['dice_WT']['mean']:.3f}, core {summary['dice_TC']['mean']:.3f}, enhancing {summary['dice_ET']['mean']:.3f}.
The uncertainty score separated poor outlines from good ones with AUROC {summary['reliability']['failure_auroc']:.3f},
which is what the review flag is based on.
            """)
        else:
            st.json({k: summary[k] for k in ["n_test", "n_val", "mc_samples", "dice_WT", "dice_TC", "dice_ET", "hd95_WT",
                                             "sensitivity_WT", "specificity_WT", "reliability", "review_flag"] if k in summary})
            st.write({"checkpoint_epoch": model_info.get("epoch"), "val_at_checkpoint": model_info.get("val")})
        sub_csv = os.path.join(results_dir, "subsets.csv")
        if os.path.exists(sub_csv):
            import pandas as pd
            st.markdown("Dice on the test set for every combination of the four sequences:")
            df = pd.read_csv(sub_csv)[["subset", "n_modalities", "dice_WT", "dice_TC", "dice_ET", "uncertainty", "flag_rate"]].round(3)
            st.dataframe(df, width="stretch", hide_index=True)
        figs = [("fig_subsets.png", "Dice per modality subset"), ("fig_uncertainty_vs_dice.png", "Uncertainty vs Dice (per test patient)"),
                ("fig_risk_coverage.png", "Risk-coverage curve")]
        cols = st.columns(3)
        for col, (f, cap) in zip(cols, figs):
            p = os.path.join(results_dir, f)
            if os.path.exists(p):
                col.image(p, caption=cap)
    else:
        st.write("No results/summary.json found - run scripts/evaluate.py.")
    st.markdown("""
**Model.** 2D U-Net (7.8 M parameters), trained with *modality dropout* so it learns to work with any subset of the four MRI sequences.
Confidence comes from Monte Carlo dropout: the network is run several times and the disagreement (predictive entropy) is measured.
The review threshold was fixed on the validation set and never tuned on the test set.

**Limitations.** Trained on BraTS 2020 (adult glioma, skull-stripped, co-registered, 1 mm isotropic). Not validated on other
tumour types, scanners, children, or post-operative scans. Uncertainty measures model disagreement, not correctness.
This is a research prototype for human-in-the-loop review, **not a diagnostic device**.
    """)
st.caption("NeuroScope. Research prototype built on BraTS 2020. Not a medical device.")
