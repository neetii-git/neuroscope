"""NeuroScope — modality-robust, uncertainty-aware brain tumour segmentation (research prototype).

Run:  streamlit run app/app.py -- --ckpt runs/base/best.pt --results runs/base/results
"""
from __future__ import annotations

import argparse
import io
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

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from neuroscope.data.brats import MODALITIES, load_nifti_with_affine, remap_labels  # noqa: E402
from neuroscope.data.preprocess import preprocess_case, uncrop  # noqa: E402
from neuroscope.eval.metrics import region_metrics  # noqa: E402
from neuroscope.data.brats import labels_to_regions  # noqa: E402
from neuroscope.inference.predictor import Prediction, load_checkpoint, predict_volume  # noqa: E402
from neuroscope.quality.iqa import assess_case  # noqa: E402

LABEL_COLORS = {1: (1.0, 0.2, 0.2), 2: (0.2, 0.8, 0.2), 3: (1.0, 0.85, 0.1)}  # NCR red, ED green, ET yellow
LABEL_NAMES = {1: "necrotic / non-enhancing core", 2: "peritumoural edema", 3: "enhancing tumour"}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.environ.get("NEUROSCOPE_CKPT", os.path.join(ROOT, "runs", "base", "best.pt")))
    ap.add_argument("--results", default=os.environ.get("NEUROSCOPE_RESULTS", os.path.join(ROOT, "runs", "base", "results")))
    ap.add_argument("--demo-dir", default=os.environ.get("NEUROSCOPE_DEMO", os.path.join(ROOT, "demo_case")))
    args, _ = ap.parse_known_args()
    return args


ARGS = parse_args()


@st.cache_resource(show_spinner=False)
def get_model(ckpt_path: str, device: str):
    model = load_checkpoint(ckpt_path, device)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return model, {"epoch": ckpt.get("epoch"), "val": ckpt.get("val"), "config": ckpt.get("full_config", ckpt.get("config"))}


@st.cache_data(show_spinner=False)
def load_threshold(results_dir: str) -> Optional[Dict]:
    p = os.path.join(results_dir, "threshold.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


@st.cache_data(show_spinner=False)
def load_summary(results_dir: str) -> Optional[Dict]:
    p = os.path.join(results_dir, "summary.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


def read_upload(upload) -> Optional[tuple]:
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


def overlay_figure(base: np.ndarray, labels: np.ndarray, entropy: np.ndarray, z: int, title_mod: str):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    sl = base[z]
    vmin, vmax = np.percentile(sl[sl != 0], [1, 99]) if (sl != 0).any() else (0, 1)
    for ax in axes:
        ax.imshow(sl.T, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
        ax.axis("off")
    axes[0].set_title(f"{title_mod.upper()} — slice {z}")
    rgba = np.zeros(sl.shape + (4,), dtype=np.float32)
    for lab, col in LABEL_COLORS.items():
        m = labels[z] == lab
        rgba[m, :3] = col
        rgba[m, 3] = 0.45
    axes[1].imshow(np.transpose(rgba, (1, 0, 2)), origin="lower")
    axes[1].set_title("Predicted segmentation")
    ent = entropy[z]
    axes[2].imshow(ent.T, cmap="magma", alpha=0.8, vmin=0, vmax=float(np.log(2)), origin="lower")
    axes[2].set_title("Predictive entropy (uncertainty)")
    fig.tight_layout()
    return fig


def plotly_3d(regions: np.ndarray, step: int = 2):
    import plotly.graph_objects as go
    fig = go.Figure()
    names = ["Whole tumour", "Tumour core", "Enhancing"]
    colors = ["#2ca02c", "#d62728", "#ffbf00"]
    for i in range(3):
        v = regions[i][::step, ::step, ::step].astype(np.float32)
        if v.sum() == 0:
            continue
        D, H, W = v.shape
        zz, yy, xx = np.mgrid[0:D, 0:H, 0:W]
        fig.add_trace(go.Isosurface(x=xx.flatten(), y=yy.flatten(), z=zz.flatten(), value=v.flatten(),
                                    isomin=0.5, isomax=1.0, surface_count=1, opacity=0.35 if i == 0 else 0.6,
                                    colorscale=[[0, colors[i]], [1, colors[i]]], showscale=False, name=names[i], caps=dict(x_show=False, y_show=False, z_show=False)))
    fig.update_layout(height=520, margin=dict(l=0, r=0, t=10, b=0), scene=dict(aspectmode="data"))
    return fig


def build_report(pred: Prediction, quality, present, flagged, thr, infer_s, model_info, metrics=None) -> Dict:
    return {
        "tool": "NeuroScope research prototype — NOT a medical device; outputs require expert review",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "modalities_used": present,
        "modalities_missing": [m for m in MODALITIES if m not in present],
        "input_quality": quality.to_dict(),
        "tumour_volumes_ml": pred.volumes_ml,
        "case_uncertainty": pred.case_uncertainty,
        "review_threshold": thr,
        "needs_human_review": bool(flagged),
        "mc_samples": pred.mc_samples,
        "inference_seconds": infer_s,
        "model": {k: model_info.get(k) for k in ["epoch", "val"]},
        "metrics_vs_ground_truth": metrics,
    }


def report_markdown(r: Dict) -> str:
    lines = ["# NeuroScope analysis report", "", f"_{r['tool']}_", "", f"Generated: {r['timestamp']}", "",
             "## Input", f"- Modalities used: {', '.join(r['modalities_used'])}",
             f"- Missing: {', '.join(r['modalities_missing']) or 'none'}",
             f"- Quality status: **{r['input_quality']['status']}** (score {r['input_quality']['score']:.2f})"]
    for c in r["input_quality"]["checks"]:
        if c["status"] != "pass":
            lines.append(f"  - [{c['modality']}] {c['message']}")
    lines += ["", "## Segmentation", f"- Whole tumour: {r['tumour_volumes_ml']['WT_ml']:.1f} mL",
              f"- Tumour core: {r['tumour_volumes_ml']['TC_ml']:.1f} mL", f"- Enhancing: {r['tumour_volumes_ml']['ET_ml']:.1f} mL",
              "", "## Reliability", f"- Case uncertainty (mean entropy in ROI): {r['case_uncertainty']:.4f}",
              f"- MC dropout samples: {r['mc_samples']}", f"- **Needs human review: {'YES' if r['needs_human_review'] else 'no'}**"]
    if r["review_threshold"]:
        t = r["review_threshold"]
        lines.append(f"- Threshold {t['threshold']:.4f} chosen on validation set (rule: {t.get('rule')}, failure = WT Dice < {t.get('tau_fail')})")
    if r.get("metrics_vs_ground_truth"):
        m = r["metrics_vs_ground_truth"]
        lines += ["", "## Agreement with provided ground truth", f"- Dice WT {m['dice_WT']:.3f} | TC {m['dice_TC']:.3f} | ET {m['dice_ET']:.3f}"]
    lines += ["", "## Limitations", "- 2D slice-wise model; trained on BraTS 2020 adult glioma only.",
              "- Uncertainty reflects model disagreement, not clinical reliability.",
              "- Quality checks are heuristics; passing them does not certify diagnostic suitability."]
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------
st.set_page_config(page_title="NeuroScope", layout="wide")
st.title("NeuroScope")
st.caption("Modality-robust, uncertainty-aware brain tumour segmentation — research & decision-support prototype, not a diagnostic device.")

with st.sidebar:
    st.header("Model")
    ckpt_path = st.text_input("Checkpoint", ARGS.ckpt)
    results_dir = st.text_input("Results dir (threshold/summary)", ARGS.results)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.write(f"Device: `{device}`")
    mc = st.slider("MC dropout samples", 1, 16, 6, help="More samples = better uncertainty estimate, slower")
    show_3d = st.checkbox("3D tumour view", value=True)
    st.header("Input")
    st.write("Upload any subset of the four sequences (.nii / .nii.gz). Missing ones are handled by the model and reported.")
    uploads = {m: st.file_uploader(m.upper(), type=["nii", "gz"], key=m) for m in MODALITIES}
    seg_upload = st.file_uploader("Ground-truth mask (optional, for evaluation)", type=["nii", "gz"], key="seg")
    use_demo = st.button("Load bundled demo case") if os.path.isdir(ARGS.demo_dir) else False
    run = st.button("Run analysis", type="primary")

if not os.path.exists(ckpt_path):
    st.error(f"Checkpoint not found: {ckpt_path}. Train first (scripts/train.py) or pass --ckpt.")
    st.stop()
model, model_info = get_model(ckpt_path, device)
thr = load_threshold(results_dir)
summary = load_summary(results_dir)

if "case" not in st.session_state:
    st.session_state.case = None

if use_demo:
    vols, affine = {}, np.eye(4)
    import glob as _glob
    for m in MODALITIES:
        hits = _glob.glob(os.path.join(ARGS.demo_dir, f"*_{m}.nii*"))
        if hits:
            vols[m], affine, _ = load_nifti_with_affine(hits[0])
    seg_hits = _glob.glob(os.path.join(ARGS.demo_dir, "*seg*.nii*")) + _glob.glob(os.path.join(ARGS.demo_dir, "*Segm*.nii*"))
    if seg_hits:
        vols["seg"] = remap_labels(load_nifti_with_affine(seg_hits[0])[0].astype(np.int16))
    st.session_state.case = (vols, affine)
    run = True
elif run:
    vols, affine = {}, np.eye(4)
    for m in MODALITIES:
        r = read_upload(uploads[m])
        if r is not None:
            vols[m], affine, _ = r
    if seg_upload is not None:
        vols["seg"] = remap_labels(read_upload(seg_upload)[0].astype(np.int16))
    st.session_state.case = (vols, affine) if vols else None

if run and st.session_state.case is None:
    st.warning("Upload at least one MRI sequence.")

if st.session_state.case is not None:
    vols, affine = st.session_state.case
    present = [m for m in MODALITIES if m in vols]
    quality = assess_case({m: vols.get(m) for m in MODALITIES})
    if quality.status == "fail":
        st.error("Input rejected by quality gate: " + "; ".join(quality.warnings()))
        st.stop()

    with st.spinner("Preprocessing + Monte Carlo inference…"):
        img, seg, meta = preprocess_case(vols, available=present)
        t0 = time.time()
        voxel_mm3 = float(abs(np.linalg.det(affine[:3, :3]))) or 1.0
        pred = predict_volume(model, img, mc_samples=mc, device=device, batch_size=16, voxel_volume_mm3=voxel_mm3)
        infer_s = time.time() - t0
    flagged = bool(thr and pred.case_uncertainty >= thr["threshold"])
    metrics = region_metrics(pred.regions, labels_to_regions(seg), with_hd95=False) if seg is not None else None

    # ---- status row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Whole tumour", f"{pred.volumes_ml['WT_ml']:.1f} mL")
    c2.metric("Tumour core", f"{pred.volumes_ml['TC_ml']:.1f} mL")
    c3.metric("Enhancing", f"{pred.volumes_ml['ET_ml']:.1f} mL")
    c4.metric("Case uncertainty", f"{pred.case_uncertainty:.4f}", help="Mean predictive entropy in the tumour ROI over MC samples")
    c5.metric("Inference", f"{infer_s:.1f} s", f"MC x{pred.mc_samples}, {device}")

    if flagged:
        st.error(f"NEEDS HUMAN REVIEW — uncertainty {pred.case_uncertainty:.4f} ≥ validated threshold {thr['threshold']:.4f}")
    elif thr:
        st.success(f"Uncertainty below review threshold ({thr['threshold']:.4f}); still requires expert confirmation.")
    else:
        st.info("No validated threshold found (run scripts/evaluate.py) — review flag disabled.")
    missing = [m for m in MODALITIES if m not in present]
    if missing:
        st.warning(f"Running without: {', '.join(m.upper() for m in missing)}. Accuracy for affected regions is reduced (see subset table).")
    if quality.status == "warn":
        with st.expander(f"Input quality: {quality.status} (score {quality.score:.2f}) — {len(quality.warnings())} warning(s)", expanded=False):
            for w in quality.warnings():
                st.write("- " + w)
    else:
        st.caption(f"Input quality: pass (score {quality.score:.2f})")
    if metrics:
        st.info(f"Agreement with uploaded ground truth — Dice WT {metrics['dice_WT']:.3f} | TC {metrics['dice_TC']:.3f} | ET {metrics['dice_ET']:.3f}")

    # ---- slice viewer
    st.subheader("Slice viewer")
    base_mod = st.selectbox("Background sequence", present, index=present.index("flair") if "flair" in present else 0)
    tum_z = np.where(pred.regions[0].reshape(pred.regions.shape[1], -1).sum(1) > 0)[0]
    default_z = int(tum_z[len(tum_z) // 2]) if tum_z.size else img.shape[1] // 2
    z = st.slider("Axial slice", 0, img.shape[1] - 1, default_z)
    st.pyplot(overlay_figure(img[MODALITIES.index(base_mod)], pred.labels, pred.entropy, z, base_mod), clear_figure=True)
    st.caption("Overlay: red = necrotic/non-enhancing core, green = edema, yellow = enhancing tumour. Entropy: brighter = less certain.")

    if show_3d and pred.regions[0].sum() > 0:
        st.subheader("3D tumour view")
        st.plotly_chart(plotly_3d(pred.regions), width="stretch")

    # ---- report & downloads
    st.subheader("Report")
    report = build_report(pred, quality, present, flagged, thr, infer_s, model_info, metrics)
    md = report_markdown(report)
    d1, d2, d3 = st.columns(3)
    d1.download_button("Download report (.md)", md, file_name="neuroscope_report.md")
    d2.download_button("Download report (.json)", json.dumps(report, indent=2, default=float), file_name="neuroscope_report.json")
    lab_full = uncrop(pred.labels, meta).astype(np.uint8)
    lab_full[lab_full == 3] = 4  # back to BraTS label convention
    nii_bytes = nib.Nifti1Image(lab_full, affine).to_bytes()
    d3.download_button("Download segmentation (.nii)", nii_bytes, file_name="neuroscope_seg.nii")
    with st.expander("Report preview"):
        st.markdown(md)

with st.expander("Model card & validated performance"):
    st.write({"checkpoint_epoch": model_info.get("epoch"), "val_at_checkpoint": model_info.get("val")})
    if summary:
        keys = ["dice_WT", "dice_TC", "dice_ET", "hd95_WT"]
        st.write({k: summary[k] for k in keys if k in summary})
        st.write({"reliability": summary.get("reliability"), "review_flag": summary.get("review_flag")})
        if summary.get("subsets"):
            st.write({"single_modality_dice_WT": summary["subsets"].get("single_modality_dice_WT")})
        sub_csv = os.path.join(results_dir, "subsets.csv")
        if os.path.exists(sub_csv):
            import pandas as pd
            st.dataframe(pd.read_csv(sub_csv)[["subset", "n_modalities", "dice_WT", "dice_TC", "dice_ET", "uncertainty", "flag_rate"]].round(3), width="stretch")
        for fig in ["fig_subsets.png", "fig_uncertainty_vs_dice.png", "fig_risk_coverage.png"]:
            p = os.path.join(results_dir, fig)
            if os.path.exists(p):
                st.image(p)
    else:
        st.write("No results/summary.json yet — run scripts/evaluate.py.")
    st.markdown("**Limitations.** 2D slice-wise U-Net trained on BraTS 2020 (adult glioma, 1 mm isotropic, skull-stripped, co-registered). "
                "Not validated on other tumour types, scanners or populations. Uncertainty measures model disagreement, not correctness. "
                "This is a research prototype for human-in-the-loop review, not a diagnostic device.")
