"""Hospital scan import: a zipped DICOM folder (CD / portal download) -> the four BraTS-style volumes the model expects.

Steps: unzip to a temporary folder -> group files into series -> keep brain MR series and classify them as FLAIR, T1, T1ce, T2
-> load each as a 3D image -> resample onto one 1 mm axis-aligned grid -> rigid registration to the reference sequence
-> skull removal (HD-BET when installed, otherwise an approximate morphological method) -> NIfTI arrays.
Patient-identifying header fields are never read into the result or shown; the temporary folder is deleted afterwards.
"""
from __future__ import annotations

import os
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import pydicom
    import SimpleITK as sitk
    HAVE_DICOM = True
except Exception:  # pragma: no cover
    HAVE_DICOM = False

LABELS = {"flair": "FLAIR", "t1": "T1", "t1ce": "T1 with contrast", "t2": "T2"}
_EXCLUDE = re.compile(r"(dwi|diff|\badc\b|trace|\bswi\b|swan|perf|pwi|\bdsc\b|\basl\b|locali[sz]er|\bloc\b|scout|survey|3.?plane|\btof\b|\bmra\b|"
                      r"spine|calib|field.?map|\bb0\b|dti|tensor|\bmip\b|\bmin.?ip\b|screen.?save|report|dose|phase|\bpha\b|\bmag\b.*swi)")
_FLAIR = re.compile(r"(flair|tirm|dark.?fl|darkfluid|\bfl\b.*\bt2\b|\bspace.?fl)")
_T2 = re.compile(r"(\bt2\b|t2w|t2_|_t2|t2.?tse|tse.?t2|t2.?frfse|propeller.*t2)")
_T1 = re.compile(r"(\bt1\b|t1w|t1_|_t1|mprage|mp.?rage|spgr|fspgr|bravo|\btfl\b|vibe|\bt1.?se\b|\bse.?t1\b|t1.?fl|t1.?space|cube.?t1|ffe.*t1)")
_POST = re.compile(r"(post|\+\s*c\b|\bc\s*\+|gad|\bgd\b|\+gd|contrast|\bce\b|_ce\b|\bce_|enhanc|\bpc\b|dotarem|magnevist|gadovist|multihance|prohance)")


@dataclass
class Series:
    uid: str
    files: List[str]
    modality: str
    description: str
    number: int
    n: int
    image_type: str
    tr: Optional[float]
    te: Optional[float]
    ti: Optional[float]
    contrast: str
    acq_time: str
    thickness: Optional[float]
    rows: int
    cols: int
    label: Optional[str] = None
    reason: str = ""
    excluded: str = ""


def _f(v) -> Optional[float]:
    try:
        return float(v)
    except Exception:
        return None


def extract_zip(data: bytes, limit_files: int = 20000) -> str:
    tmp = tempfile.mkdtemp(prefix="ns_dicom_")
    path = os.path.join(tmp, "upload.zip")
    with open(path, "wb") as fh:
        fh.write(data)
    with zipfile.ZipFile(path) as z:
        members = [m for m in z.infolist() if not m.is_dir()]
        if len(members) > limit_files:
            raise ValueError(f"The archive has {len(members)} files; the limit is {limit_files}.")
        root = os.path.realpath(os.path.join(tmp, "x"))
        for m in members:
            target = os.path.realpath(os.path.join(root, m.filename))
            if not target.startswith(root + os.sep):   # zip-slip guard
                continue
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with z.open(m) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
    os.remove(path)
    return tmp


def find_series(root: str) -> List[Series]:
    groups: Dict[str, dict] = {}
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.upper() == "DICOMDIR":
                continue
            p = os.path.join(dp, fn)
            try:
                ds = pydicom.dcmread(p, stop_before_pixels=True, force=True,
                                     specific_tags=["SeriesInstanceUID", "Modality", "SeriesDescription", "ProtocolName", "SequenceName",
                                                    "SeriesNumber", "ImageType", "RepetitionTime", "EchoTime", "InversionTime",
                                                    "ContrastBolusAgent", "AcquisitionTime", "SeriesTime", "SliceThickness", "Rows", "Columns",
                                                    "BodyPartExamined", "SOPClassUID"])
            except Exception:
                continue
            uid = getattr(ds, "SeriesInstanceUID", None)
            if not uid or not getattr(ds, "Rows", None):
                continue
            g = groups.setdefault(uid, {"files": [], "ds": ds})
            g["files"].append(p)
    out = []
    for uid, g in groups.items():
        ds = g["ds"]
        desc = " ".join(str(getattr(ds, k, "") or "") for k in ("SeriesDescription", "ProtocolName", "SequenceName")).strip()
        out.append(Series(uid=uid, files=g["files"], modality=str(getattr(ds, "Modality", "")), description=desc or "(no description)",
                          number=int(_f(getattr(ds, "SeriesNumber", 0)) or 0), n=len(g["files"]),
                          image_type="\\".join(getattr(ds, "ImageType", []) or []), tr=_f(getattr(ds, "RepetitionTime", None)),
                          te=_f(getattr(ds, "EchoTime", None)), ti=_f(getattr(ds, "InversionTime", None)),
                          contrast=str(getattr(ds, "ContrastBolusAgent", "") or ""),
                          acq_time=str(getattr(ds, "AcquisitionTime", "") or getattr(ds, "SeriesTime", "") or ""),
                          thickness=_f(getattr(ds, "SliceThickness", None)), rows=int(ds.Rows), cols=int(getattr(ds, "Columns", 0) or 0)))
    out.sort(key=lambda s: s.number)
    return out


def classify(series: List[Series]) -> Tuple[Dict[str, Series], List[str]]:
    """Label every series, then pick the best candidate per sequence. Returns (chosen, notes)."""
    notes = []
    mods = {s.modality for s in series}
    if series and "MR" not in mods:
        raise ValueError(f"No MRI series found (modalities in the archive: {', '.join(sorted(m for m in mods if m)) or 'unknown'}). "
                         "NeuroScope works on brain MRI only, not CT or X-ray.")
    for s in series:
        t = s.description.lower()
        if s.modality != "MR":
            s.excluded = f"not MRI ({s.modality or 'unknown'})"
            continue
        if _EXCLUDE.search(t):
            s.excluded = "not a structural sequence (diffusion, perfusion, localiser or similar)"
            continue
        if s.n < 10 and "MOSAIC" not in s.image_type:
            s.excluded = "too few images for a 3D volume"
            continue
        if _FLAIR.search(t) or (s.ti and s.ti > 1500 and (s.tr or 0) > 4000):
            s.label, s.reason = "flair", "FLAIR keyword" if _FLAIR.search(t) else "long inversion time"
        elif _T1.search(t):
            post = bool(s.contrast.strip()) or bool(_POST.search(t))
            s.label, s.reason = ("t1ce", "contrast agent recorded" if s.contrast.strip() else "post-contrast keyword") if post else ("t1", "T1 keyword")
        elif _T2.search(t) or ((s.te or 0) > 70 and (s.tr or 0) > 2000 and not s.ti):
            s.label, s.reason = "t2", "T2 keyword" if _T2.search(t) else "long TE and TR"
        elif (s.te or 99) < 20 and (s.tr or 9999) < 1000:
            s.label, s.reason = "t1", "short TE and TR"
        else:
            s.excluded = "sequence type not recognised"
    # two unlabelled T1 series and no post-contrast one: the later is assumed post-contrast
    t1s = [s for s in series if s.label == "t1"]
    if len(t1s) >= 2 and not any(s.label == "t1ce" for s in series):
        later = max(t1s, key=lambda s: (s.acq_time, s.number))
        later.label, later.reason = "t1ce", "assumed post-contrast (later of two T1 series)"
        notes.append(f"Two T1 series without a contrast label; series {later.number} (later) was assumed to be post-contrast. Check this.")

    def score(s: Series):
        return ("DERIVED" not in s.image_type.upper(), s.n * min(s.rows, s.cols), -(s.thickness or 5))
    chosen = {}
    for lab in LABELS:
        cands = [s for s in series if s.label == lab]
        if cands:
            chosen[lab] = max(cands, key=score)
    return chosen, notes


# ------------------------------------------------------------------ image handling
def _read(series: Series, root: str) -> "sitk.Image":
    r = sitk.ImageSeriesReader()
    names = r.GetGDCMSeriesFileNames(root, series.uid, False, True)
    if not names:
        names = sorted(series.files)
    if len(names) == 1:
        img = sitk.ReadImage(names[0])
    else:
        r.SetFileNames(names)
        img = r.Execute()
    if img.GetDimension() != 3:
        raise ValueError(f"Series {series.number} is not a 3D volume")
    return sitk.Cast(img, sitk.sitkFloat32)


def _grid(ref: "sitk.Image", spacing: float = 1.0) -> "sitk.Image":
    """Axis-aligned (identity direction) 1 mm grid that covers the reference image."""
    sz = ref.GetSize()
    corners = [ref.TransformContinuousIndexToPhysicalPoint((i * (sz[0] - 1), j * (sz[1] - 1), k * (sz[2] - 1)))
               for i in (0, 1) for j in (0, 1) for k in (0, 1)]
    lo, hi = np.min(corners, 0), np.max(corners, 0)
    size = [int(np.ceil((hi[d] - lo[d]) / spacing)) + 1 for d in range(3)]
    g = sitk.Image(size, sitk.sitkFloat32)
    g.SetSpacing([spacing] * 3)
    g.SetOrigin([float(v) for v in lo])
    g.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
    return g


def _register(fixed: "sitk.Image", moving: "sitk.Image") -> "sitk.Transform":
    """Rigid, mutual-information registration. Starts from scanner geometry (sequences from one session are close already)."""
    init = sitk.CenteredTransformInitializer(fixed, moving, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY)
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(32)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.05, seed=1)
    R.SetInterpolator(sitk.sitkLinear)
    R.SetOptimizerAsRegularStepGradientDescent(1.0, 1e-3, 120, relaxationFactor=0.6)
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetShrinkFactorsPerLevel([4, 2])
    R.SetSmoothingSigmasPerLevel([2, 1])
    R.SetInitialTransform(init, inPlace=False)
    try:
        before = R.MetricEvaluate(fixed, moving)
        tx = R.Execute(fixed, moving)
        after = R.GetMetricValue()
        return tx if after <= before else sitk.Transform(3, sitk.sitkIdentity)
    except Exception:
        return sitk.Transform(3, sitk.sitkIdentity)


def _largest(m):
    from scipy import ndimage as ndi
    lab, n = ndi.label(m)
    return (lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)) if n else m


def _skull_strip_approx(a: np.ndarray, spacing: float = 1.0) -> np.ndarray:
    """Approximate brain mask. The brain is separated from the scalp by dark bone and fluid, so after thresholding and a small
    erosion the brain is the largest connected piece. Falls back to a stronger erosion if the result is not brain-sized."""
    from scipy import ndimage as ndi
    from skimage.filters import threshold_otsu
    v = a[a > 0]
    if v.size < 1000:
        return a > 0
    th = threshold_otsu(np.clip(v, np.percentile(v, 1), np.percentile(v, 99.5)))
    ball = ndi.generate_binary_structure(3, 1)
    vox_ml = spacing ** 3 / 1000.0
    best = None
    for frac, r in ((0.5, 3), (0.5, 5), (0.35, 7)):
        tissue = ndi.binary_fill_holes(a > th * frac, structure=np.ones((3, 3, 3))) if r >= 7 else (a > th * frac)
        core = _largest(ndi.binary_erosion(tissue, ball, iterations=r))
        m = ndi.binary_fill_holes(ndi.binary_dilation(core, ball, iterations=r) & (a > 0))
        ml = m.sum() * vox_ml
        if 700 <= ml <= 2100:          # adult brain incl. tumour, roughly
            return m
        best = m if best is None or abs(ml - 1300) < abs(best.sum() * vox_ml - 1300) else best
    return best


def _skull_strip(img: "sitk.Image", tmpdir: str) -> Tuple[np.ndarray, str]:
    try:   # HD-BET (deep-learning skull stripping) when installed
        from HD_BET.hd_bet_prediction import get_hdbet_predictor, hdbet_predict  # type: ignore
        src, dst = os.path.join(tmpdir, "ref.nii.gz"), os.path.join(tmpdir, "ref_bet.nii.gz")
        sitk.WriteImage(img, src)
        predictor = get_hdbet_predictor(use_tta=False, device=__import__("torch").device("cpu"), verbose=False)
        hdbet_predict(src, dst, predictor, keep_brain_mask=True, compute_brain_extracted_image=False)
        m = sitk.GetArrayFromImage(sitk.ReadImage(dst.replace(".nii.gz", "_bet.nii.gz") if os.path.exists(dst.replace(".nii.gz", "_bet.nii.gz")) else dst)) > 0
        return m, "HD-BET"
    except Exception:
        return _skull_strip_approx(sitk.GetArrayFromImage(img)), "approximate (morphological)"


def prepare(root: str, chosen: Dict[str, Series], progress=None, target_axcodes: Tuple[str, str, str] = ("L", "P", "S")) -> Tuple[Dict[str, np.ndarray], np.ndarray, List[str]]:
    """Load, align and skull-strip the chosen series. Returns (volumes in NIfTI (x, y, z) order, affine, log)."""
    import nibabel as nib
    log = []
    say = progress or (lambda m: None)
    order = [m for m in ("t1ce", "flair", "t1", "t2") if m in chosen]
    if not order:
        raise ValueError("No usable FLAIR, T1, T1 with contrast or T2 series was found.")
    ref_key = order[0]
    say(f"Reading {LABELS[ref_key]} (reference)")
    ref = _read(chosen[ref_key], root)
    grid = _grid(ref)
    imgs = {ref_key: sitk.Resample(ref, grid, sitk.Transform(3, sitk.sitkIdentity), sitk.sitkLinear, 0.0)}
    for k in order[1:]:
        say(f"Reading and aligning {LABELS[k]}")
        mov = _read(chosen[k], root)
        tx = _register(imgs[ref_key], mov)
        imgs[k] = sitk.Resample(mov, grid, tx, sitk.sitkLinear, 0.0)
        log.append(f"{LABELS[k]} aligned to {LABELS[ref_key]} ({'registration refined' if tx.GetName() != 'IdentityTransform' else 'scanner geometry'}).")
    say("Removing the skull")
    strip_from = next(imgs[k] for k in ("t1", "t1ce", "flair", "t2") if k in imgs)
    mask, method = _skull_strip(strip_from, root)
    log.append(f"Skull removal: {method}.")
    tmp = tempfile.mkdtemp(prefix="ns_nii_", dir=root)
    vols, affine = {}, np.eye(4)
    for k, im in imgs.items():
        a = sitk.GetArrayFromImage(im)
        a[~mask] = 0
        out = sitk.GetImageFromArray(a.astype(np.float32))
        out.CopyInformation(im)
        p = os.path.join(tmp, f"{k}.nii.gz")
        sitk.WriteImage(out, p)
        n = nib.load(p)
        cur = nib.orientations.io_orientation(n.affine)
        tgt = nib.orientations.axcodes2ornt(tuple(target_axcodes))
        tr = nib.orientations.ornt_transform(cur, tgt)
        n = n.as_reoriented(tr)   # same axis order and directions as the training data
        vols[k] = np.asarray(n.get_fdata(dtype=np.float32))
        affine = n.affine
    log.append(f"Oriented as {''.join(target_axcodes)} to match the training data.")
    return vols, affine, log


def cleanup(root: Optional[str]) -> None:
    if root and os.path.isdir(root):
        shutil.rmtree(root, ignore_errors=True)
