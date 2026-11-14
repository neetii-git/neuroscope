# 3-minute demo script

1. **Open the app** (`streamlit run app/app.py`). Point at the title line: research prototype, human-in-the-loop.
2. **Load bundled demo case** (a held-out *test* patient the model never saw). Show: volumes in mL,
   case uncertainty, inference time on CPU, the review-flag banner, quality status.
3. **Slice viewer**: scrub the slider. Left = MRI, middle = 3 tumour regions, right = entropy map — bright rim
   on the boundaries is where the model disagrees with itself.
4. **3D view**: rotate the tumour.
5. **Remove a sequence**: upload the same case *without T1ce*. Segmentation still produced; enhancing-tumour
   volume drops and uncertainty rises → flag may flip to "needs review". That is the point of the project.
6. **Model card expander**: the 15-subset table, uncertainty-vs-Dice scatter, risk–coverage curve.
   "The threshold was chosen on validation data and frozen; here is how it performs on the test set."
7. **Download report** → open the markdown.

Viva one-liners
- Why 2D? Compute (1 h on a free GPU) and the contribution is the reliability layer, which is backbone-agnostic.
- Why modality dropout? Matches deployment exactly: a missing sequence *is* a zero channel. No synthetic images.
- How do you know uncertainty means anything? Failure-detection AUROC and the risk–coverage curve on held-out cases.
- Data leakage? Split is by patient, seeded, saved in `cache/split.json`; threshold chosen on val, never test.
- What failed? See `corruptions.csv` and the worst subsets — single T1 is weak for enhancing tumour, as expected.
