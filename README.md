# NeuroScope

**Modality-robust, uncertainty-aware brain tumour segmentation with a validated human-review gate.**
Research & decision-support prototype — not a medical device.

Clinical MRI exams are often incomplete (a sequence skipped for time, ruined by motion) and acquired on
scanners unlike the ones research datasets come from. Most published tumour-segmentation models assume all
four sequences are present and silently degrade when they are not. NeuroScope is built around that failure
mode:

1. **One model, any subset of sequences.** A 2D multimodal U-Net trained with *modality dropout* so it segments
   from whichever of T1 / T1ce / T2 / FLAIR are available. Evaluated on **all 15 modality combinations**.
2. **It knows when it is unsure.** Monte Carlo dropout gives voxel-wise predictive entropy and a case-level
   uncertainty score. The score is tested for whether it actually predicts segmentation failure
   (failure-detection AUROC, risk–coverage / AURC, Spearman ρ).
3. **A review threshold chosen on validation data, frozen for test.** Cases above it are flagged
   *needs human review*. No hand-picked confidence numbers.
4. **Input quality gate.** Missing modalities, SNR, saturation, field-of-view and intensity-tail checks run before
   inference and are reported with the result.
5. **Robustness measured, not claimed.** Dice and flag rate under added noise, contrast change and resolution loss.
6. **An application, not a notebook.** Streamlit app: upload any subset of NIfTI sequences → segmentation overlay,
   entropy map, 3D tumour view, volumes in mL, review flag, downloadable report and mask. Runs on CPU.

## Pipeline

```
raw NIfTI (4 seq + mask) ──► preprocess ──► uint8 cache + patient-level split (70/10/20)
                               │  brain-mask z-score, clip [-4,6], 192² crop, brain z-range
                               ▼
                      SliceDataset (axial slices, modality dropout p=0.5, flips, gain jitter)
                               ▼
                      UNet2D (4→3 regions WT/TC/ET, Dropout2d in deep blocks)   loss = Dice + BCE
                               ▼
      evaluate.py: val → choose threshold ─► test (Dice/IoU/HD95 + 95% CI, MC uncertainty, flag perf)
                                           ─► 15 subsets ─► corruptions ─► figures
                               ▼
                      app/app.py: quality gate → preprocess → MC inference → viewer / report
```

Regions follow BraTS: WT = whole tumour, TC = tumour core, ET = enhancing tumour (hierarchy ET ⊆ TC ⊆ WT enforced).

## Repository

```
neuroscope/
  data/brats.py        case discovery, label remap (4→3), region conversion
  data/preprocess.py   normalisation, cropping, quantisation, uncrop (shared by train/eval/app)
  data/split.py        seeded patient-level split
  data/dataset.py      SliceDataset w/ modality dropout; ALL_SUBSETS (15)
  models/unet2d.py     U-Net + MC-dropout switch
  models/losses.py     Dice + BCE
  eval/metrics.py      Dice, IoU, HD95, sensitivity/specificity, bootstrap CI
  eval/reliability.py  failure AUROC, risk–coverage/AURC, Youden threshold, flag performance
  quality/iqa.py       no-reference input quality checks
  inference/predictor.py  volume inference with MC dropout, entropy, case score, volumes
scripts/preprocess.py | train.py | evaluate.py | make_synthetic.py
app/app.py            Streamlit application
notebooks/kaggle_train_eval.ipynb   end-to-end GPU run on Kaggle
tests/test_pipeline.py              9 unit tests on synthetic data
configs/default.yaml
```

## Quick start

```bash
pip install -r requirements.txt
python -m pytest -q tests                      # 9 tests, ~2 s, no data needed

# smoke-test the whole pipeline on synthetic MRI (CPU, ~1 min)
python scripts/make_synthetic.py --out /tmp/synth --n 8 --small
python scripts/preprocess.py --raw /tmp/synth --cache /tmp/synth_cache
python scripts/train.py --cache /tmp/synth_cache --out /tmp/run --smoke
python scripts/evaluate.py --cache /tmp/synth_cache --ckpt /tmp/run/best.pt --out /tmp/run/results --mc 2 --mc-subsets 1
```

### Real data (BraTS 2020, Kaggle GPU)
Open `notebooks/kaggle_train_eval.ipynb` on Kaggle with the dataset
`awsaf49/brats20-dataset-training-validation` attached and a GPU enabled; run top to bottom (~1 h).
Download `neuroscope_artifacts.zip` from the notebook output and unzip it in the repo root so you have
`runs/base/best.pt`, `runs/base/results/` and `demo_case/`.

### App
```bash
streamlit run app/app.py -- --ckpt runs/base/best.pt --results runs/base/results
```
Upload any subset of the four sequences (optionally the ground-truth mask to see Dice live), or click
*Load bundled demo case*. The sidebar MC-sample slider trades speed for uncertainty quality (6 ≈ 10–20 s on CPU).

## What the evaluation produces (`runs/base/results/`)

| file | content |
|---|---|
| `summary.json` | test Dice/IoU/HD95 per region with bootstrap 95% CI, reliability (failure AUROC, AURC, ρ), flag performance |
| `threshold.json` | review threshold, rule (Youden's J on val), failure definition |
| `subsets.csv` | per-subset Dice, uncertainty, flag rate for all 15 modality combinations |
| `corruptions.csv` | Dice / uncertainty / flag rate under noise σ∈{0.1,0.3,0.6}, contrast γ∈{0.7,1.4}, resolution ÷{2,4} |
| `test_per_case.csv` | every test case: metrics, uncertainty, inference time, volumes |
| `fig_*.png` | subset bars, uncertainty-vs-Dice scatter, risk–coverage curve |

Fill in the numbers from your run here once it finishes.


### Using the app

```bash
python -m streamlit run app/app.py
```

The interface has two modes, switched in the sidebar:

- **Patient / first-time user** — plain-language explanations, a guided 3-step flow, and a clear verdict
  ("confident" vs "needs a specialist's review"). No jargon, no knobs.
- **Clinician / researcher** — full detail: per-region Dice, case uncertainty vs the validated threshold,
  input-quality checks, MC-sample control, ground-truth upload, and the complete model card with the
  15-subset table and reliability plots.

The validated test-set numbers (Dice, failure-detection AUROC, review-flag sensitivity/specificity, GPU
inference time) are shown on the landing page, read live from `runs/base/results/summary.json`.
The bundled example patient is cached after its first run, so repeat demos are instant.

## Design decisions (short)

- **2D not 3D.** A 3D U-Net at BraTS resolution needs a day of 16 GB-GPU time per model; the goal here is the
  *reliability* machinery, which is architecture-agnostic. The predictor interface takes a volume, so a 3D
  backbone can be dropped in later. Slice-wise prediction is stated as a limitation everywhere.
- **Modality dropout over synthesis.** Zeroing channels during training is simple, needs no second network and
  is exactly the input the app produces when a sequence is missing — no fabricated images.
- **Region (sigmoid) outputs** rather than 4-class softmax: they optimise the reported BraTS regions directly and
  give one calibrated probability per region for entropy.
- **Case uncertainty = mean entropy in a tumour ROI** rather than over the whole brain, so millions of
  confidently-background voxels don't mask disagreement around the lesion.
- **Threshold on validation, frozen for test.** Youden's J on failure (WT Dice < 0.75). Reported flag
  sensitivity/specificity on test is therefore honest.
- **uint8 cache.** z-scores clipped to [-4, 6] quantised at ~0.04 σ; all 369 cases fit in ~10 GB and memory-map
  instantly, which is what makes a 1-hour Kaggle run possible.

## Limitations

- Trained and tested on BraTS 2020 adult glioma only; no external dataset, no other tumour types or scanners.
- 2D slice-wise inference; no inter-slice context.
- MC dropout captures epistemic uncertainty approximately; no ensembles.
- Quality checks are heuristics and do not certify diagnostic suitability.
- Not clinically validated. Outputs require expert review.

## Future work

3D / nnU-Net backbone · external validation on BraTS-Africa (low-resource scanners) · deep-ensemble baseline ·
longitudinal volume tracking · ONNX export · FastAPI service + audit log.

## Data

BraTS 2020 (Menze et al. 2015; Bakas et al. 2017, 2018), de-identified public research data, via Kaggle
`awsaf49/brats20-dataset-training-validation`.
