# NeuroScope: brain tumour segmentation with a validated self-check under missing MRI sequences

**Technical report** · Neeti Soni, B.Tech CSE (Data Science), SRM Institute of Science and Technology · November 2026
Code: <https://github.com/neetii-git/neuroscope> · Live demo: <https://neuroscopeapp.streamlit.app>

> Every number in this report comes from a file in this repository (paths in Section 9) and can be regenerated with the commands listed there.

## Abstract

Automatic glioma segmentation from MRI is mature on curated benchmarks, but two things block routine use: hospitals often lack one or more of the four standard MRI sequences, and a model gives no warning when its outline is wrong. NeuroScope is a 2D U-Net trained with *modality dropout* so it accepts any subset of T1, T1ce, T2 and FLAIR, and uses Monte Carlo (MC) dropout to produce a per-case uncertainty score that sends doubtful results to a specialist. On 74 held-out BraTS 2020 patients it reaches whole-tumour Dice 0.899 (95% CI 0.879 to 0.914). The main contribution is evidence that the self-check keeps working when sequences are missing: across all 15 sequence combinations (1,110 patient-conditions, 208 failures), the uncertainty score separates failed from acceptable outlines with AUROC 0.954 (patient-bootstrap 95% CI 0.936 to 0.971), beating simple baselines such as counting missing sequences (0.706) or predicted tumour size (0.824). With a threshold fixed on the validation set, the flag catches 191 of 208 failures while 82% of acceptable cases pass without review. On 60 external BraTS-Africa patients (different continent, mostly lower-field scanners, never seen in training), whole-tumour Dice is 0.876 (0.834 to 0.907) and the unchanged threshold flags 226 of 232 failures across all sequence combinations (AUROC 0.964). We also report what did *not* help: Monte Carlo dropout and test-time augmentation detect failures only slightly better than a single forward pass on the internal set (AUROC 0.954 and 0.954 vs 0.946), with the gap widening on external data (0.964 and 0.967 vs 0.940); temperature scaling did not help; and a 2D SegResNet trained with the same recipe matches our U-Net (Dice 0.903 vs 0.899), so the contribution is the training and validation recipe, not the network.

## 1. Problem and contribution

**Clinical problem.** Glioma volume and its change over time drive treatment decisions, but manual 3D delineation takes a radiologist tens of minutes per scan. Deep networks trained on BraTS reach expert-level overlap, yet in practice (a) the full four-sequence protocol is not always acquired, and (b) the network fails silently on some cases.

**Research questions.**
- **RQ1** Can one network segment gliomas from any subset of the four sequences without a large loss at full input?
- **RQ2** Does a cheap uncertainty score flag failed outlines, and does it keep doing so when sequences are missing?
- **RQ3** Is the uncertainty calibrated, and which uncertainty method works best for this purpose (single pass, test-time augmentation, MC dropout, temperature scaling)?
- **RQ4** Do accuracy and the self-check transfer to scans from a different population and scanner fleet (BraTS-Africa)?

**Contribution.** Missing-modality segmentation methods (HeMIS, U-HVED, RFNet, mmFormer) report Dice per sequence combination but do not evaluate whether the model can detect its own failures. Uncertainty benchmarks for brain tumours (QU-BraTS) assume all four sequences are present. In the work we reviewed, we did not find an evaluation of failure detection *across* missing-sequence conditions. NeuroScope's contribution is that evaluation: a single lightweight model plus a review flag, with failure detection measured on every sequence combination, with patient-level confidence intervals, against trivial and established uncertainty baselines, and on an external cohort. The clinical tooling (3D viewer, RANO-style measurement, clinician correction, DICOM import, report) is engineering built on top and is labelled by validation status in Section 6.

## 2. Related work

- **Segmentation.** U-Net [Ronneberger 2015] and its variants dominate BraTS [Menze 2015; Bakas 2017, 2018]. SegResNet won BraTS 2018 [Myronenko 2018]; nnU-Net won BraTS 2020 with test-set Dice 88.95 / 85.06 / 82.03 (WT/TC/ET) [Isensee 2020, 2021]. Attention gates are a common U-Net extension [Oktay 2018].
- **Missing sequences.** HeMIS learns a shared latent space over available modalities [Havaei 2016]; U-HVED uses a multimodal variational encoder [Dorent 2019]; RFNet [Ding 2021] and mmFormer [Zhang 2022] fuse region-aware or transformer features. All report Dice for each of the 15 combinations; none evaluates failure detection.
- **Uncertainty and quality control.** MC dropout [Gal and Ghahramani 2016], deep ensembles [Lakshminarayanan 2017] and test-time augmentation [Wang 2019] are standard estimators. Jungo and Reyes [2019] showed that voxel uncertainty in brain-tumour segmentation is often poorly calibrated. QU-BraTS benchmarked uncertainty maps from 14 teams on BraTS 2020 with all sequences present [Mehta 2022]. Selective prediction and the risk-coverage curve come from Geifman and El-Yaniv [2017]; temperature scaling from Guo et al. [2017].
- **External validity.** BraTS-Africa provides expert-labelled pre-operative glioma MRI from Sub-Saharan Africa, mostly lower-field and lower-resolution than BraTS [Adewole 2023].

## 3. Data and evaluation protocol

| Item | Setting |
|---|---|
| Training data | BraTS 2020 training set, 369 adult glioma patients, four sequences, expert labels (necrotic core, edema, enhancing tumour) |
| Split | Patient-level, seed 42: 258 train / 37 validation / 74 test (`neuroscope/data/split.py`). Slices of one patient never cross sets. |
| Reproducibility check | Re-running preprocessing on Kaggle regenerated exactly the same 74 test patients as the published results (`test split identical to the published test set: True`). |
| Regions | Whole tumour (WT), tumour core (TC), enhancing tumour (ET), BraTS convention |
| Metrics | Dice, HD95 (mm), sensitivity, specificity; 95% CIs by bootstrap over patients (2,000 resamples) |
| Failure definition | WT Dice < 0.75, fixed before testing; sensitivity analysis from 0.60 to 0.85 |
| Review threshold | Chosen on the validation set (Youden's J), then frozen and applied unchanged to test and external data |
| Missing-sequence test | The same 74 test patients evaluated with each of the 15 non-empty subsets of {T1, T1ce, T2, FLAIR}; missing channels set to zero |
| Statistics across subsets | Rows from one patient are correlated, so CIs resample **patients**, keeping all 15 conditions of each sampled patient |
| External cohort | BraTS-Africa 2023 training set (60 patients with labels; Kaggle copy `aiocta/brats2023-ssa-training-dataset`); never used for training or threshold selection |

## 4. Method

**Preprocessing.** Brain mask from non-zero voxels; per-sequence z-score inside the brain, clipped to [-4, 6]; 192 x 192 axial crop centred on the brain (`neuroscope/data/preprocess.py`). The same code runs in training, evaluation and the web app.

**Model.** 2D U-Net, four levels, base width 32, BatchNorm and ReLU, 7.8 M parameters; Dropout2d (p = 0.2) in the bottleneck and the two deepest decoder blocks; three sigmoid outputs (WT, TC, ET) with the hierarchy ET ⊆ TC ⊆ WT enforced after thresholding.

**Training.** Soft Dice + BCE, AdamW (lr 3e-4, weight decay 1e-4), OneCycle schedule, 20 epochs, batch 24, mixed precision, flips and intensity augmentation. **Modality dropout:** with probability 0.5 each training slice receives a random non-empty subset of sequences, the rest zeroed.

**Uncertainty.** At inference the dropout layers stay active and the volume is processed T = 8 times. The case score is the mean predictive entropy of the WT probability over a tumour region of interest (predicted WT dilated by 2 voxels, plus voxels with probability between 0.1 and 0.9). Cases at or above the frozen threshold (0.159) are flagged for specialist review.

**Uncertainty baselines** (same trained weights, `scripts/evidence.py`): single deterministic pass (entropy of the sigmoid output), the same with temperature scaling fitted on validation voxels, and test-time augmentation with 8 passes (4 flips x 2 intensity scales).

**Architecture baselines** (`scripts/train.py --arch`): Attention U-Net [Oktay 2018] and 2D SegResNet [Myronenko 2018] from MONAI, trained with the identical split, loss, schedule, epochs, augmentation and modality dropout, so that only the architecture differs.

## 5. Results

### 5.1 Segmentation, internal test set (74 patients, all four sequences, MC = 8)

| Region | Dice (95% CI) | HD95 mm (95% CI) |
|---|---|---|
| Whole tumour | 0.899 (0.879 to 0.914) | 11.0 (8.1 to 14.3) |
| Tumour core | 0.861 (0.827 to 0.890) | 8.5 (5.6 to 11.9) |
| Enhancing tumour | 0.774 (0.719 to 0.825) | 5.5 (3.3 to 8.8) |

For context only: nnU-Net, the BraTS 2020 winner, reported 0.890 / 0.851 / 0.820 on the official BraTS 2020 test set [Isensee 2020]. The test sets differ (ours is a held-out part of the public training data), so the figures are **not directly comparable**; the like-for-like comparison is in 5.2.

### 5.2 Architecture comparison (same split and training recipe)

All three networks were trained on Kaggle with the identical split, loss, 20 epochs, augmentation and modality dropout (`notebooks/kaggle_evidence.ipynb`). Test set, 74 patients, MC = 8 for full-sequence metrics, MC = 4 for the missing-sequence columns.

| Model | Parameters | Dice WT (95% CI) | Dice TC | Dice ET | HD95 WT (mm) | Dice WT, 1 seq | 2 seq | 3 seq |
|---|---|---|---|---|---|---|---|---|
| NeuroScope 2D U-Net | 7.8 M | 0.899 (0.879 to 0.914) | 0.861 | 0.774 | 11.0 | 0.727 | 0.830 | 0.873 |
| Attention U-Net [Oktay 2018] | 7.9 M | 0.892 (0.873 to 0.908) | 0.830 | 0.758 | 12.8 | 0.669 | 0.802 | 0.856 |
| SegResNet 2D [Myronenko 2018] | 6.3 M | 0.903 (0.883 to 0.918) | 0.869 | 0.773 | 9.3 | 0.758 | 0.842 | 0.880 |

SegResNet is slightly ahead on most metrics and Attention U-Net slightly behind, but every confidence interval overlaps. The architecture is therefore not the contribution: the recipe (modality dropout plus a validated review flag) works across three different networks, and SegResNet is a sensible drop-in upgrade for a future version. The baselines' full-sequence failure AUROCs (0.977 and 1.000) rest on only 3 to 5 failures each and are not informative, which is the same small-sample problem discussed in 5.4.

### 5.3 Robustness to missing sequences, and ablation of modality dropout

Mean whole-tumour Dice over all subsets with a given number of sequences (74 test patients):

| Sequences available | With modality dropout (ours) | Without modality dropout |
|---|---|---|
| 1 | 0.727 | 0.316 |
| 2 | 0.830 | 0.603 |
| 3 | 0.872 | 0.797 |
| 4 | 0.899 | 0.900 |

Modality dropout costs nothing at full input (0.899 vs 0.900) and prevents collapse when sequences are missing. Enhancing tumour depends on T1ce: with T1ce present, ET Dice stays between 0.72 and 0.77; without it, 0.03 to 0.33, which is expected because enhancement is defined on the contrast scan. Per-subset numbers: `hosting/data/runs/base/results/subsets.csv`.

### 5.4 Failure detection (RQ2)

**All four sequences.** Only 3 of 74 patients fail, so the AUROC of 0.953 has a wide interval (0.875 to 1.000) and the frozen flag caught 2 of the 3. This setting alone cannot support a strong claim, which is why we test under missing sequences.

**All 15 sequence combinations** (1,110 patient-conditions, 74 patients, 208 failures, MC = 4):

| Score used to rank cases | Failure AUROC (patient-bootstrap 95% CI) |
|---|---|
| Number of missing sequences (trivial baseline) | 0.706 (0.662 to 0.757) |
| Small predicted tumour volume (trivial baseline) | 0.824 (0.746 to 0.888) |
| **NeuroScope uncertainty** | **0.954 (0.936 to 0.971)** |
| NeuroScope uncertainty, within each sequence set (stratified) | 0.958 (0.929 to 0.980) |

The stratified AUROC shows the score is not simply detecting *which* sequences are missing: it ranks failures correctly among cases with the same input.

**Frozen review threshold** (chosen on validation with all sequences, never re-tuned):

| | Value |
|---|---|
| Failures flagged | 191 of 208; sensitivity 0.918 (0.852 to 0.975) |
| Acceptable cases passed without review | specificity 0.824 (0.792 to 0.854) |
| Share of cases sent for review | 31.5% |
| Mean WT Dice, accepted vs flagged | 0.895 vs 0.653 |
| Risk-coverage area (AURC; lower is better) | 0.096, against 0.181 for random order and 0.086 for a perfect oracle |
| Spearman correlation, uncertainty vs Dice | -0.885 (-0.911 to -0.848) |

By number of sequences available:

| Sequences | Cases | Failures | AUROC | Flagged | Failures caught | Dice of accepted |
|---|---|---|---|---|---|---|
| 1 | 296 | 112 | 0.945 | 61% | 109 | 0.880 |
| 2 | 444 | 69 | 0.946 | 29% | 62 | 0.892 |
| 3 | 296 | 24 | 0.957 | 12% | 18 | 0.900 |
| 4 | 74 | 3 | 0.958 | 8% | 2 | 0.914 |

The flag rate rises automatically as information is removed, and the Dice of the cases it lets through stays near 0.88 to 0.91 in every condition. The result holds for any failure cut-off between 0.60 and 0.85 (AUROC 0.954 to 0.965; `docs/evidence/failure_detection.json`).

### 5.5 Which uncertainty method? (RQ3)

Same trained weights, 74 test patients x 15 sequence sets (1,110 cases). Each method's review threshold was chosen on the validation set. Patient-bootstrap 95% CIs. Timing on one NVIDIA Tesla T4.

| Method | Passes | Dice WT, 4 seq | Failures | Failure AUROC (95% CI) | AURC | Flagged failures (specificity) | Seconds per scan |
|---|---|---|---|---|---|---|---|
| Single pass, sigmoid entropy | 1 | 0.899 | 209 | 0.946 (0.924 to 0.966) | 0.097 | 191 / 209 (0.82) | 0.37 |
| Single pass + temperature scaling | 1 | 0.899 | 209 | 0.932 (0.906 to 0.955) | 0.101 | not set | 0.37 |
| Test-time augmentation | 8 | 0.904 | 198 | 0.954 (0.931 to 0.973) | 0.093 | 167 / 198 (0.91) | 3.1 |
| **MC dropout (NeuroScope)** | 8 | 0.899 | 210 | **0.954 (0.935 to 0.971)** | 0.096 | 195 / 210 (0.81) | 3.4 |

On the external cohort (900 cases) the same comparison gives AUROC 0.940 (single pass), 0.924 (temperature scaling), 0.967 (TTA) and 0.964 (MC dropout).

**Honest reading.** Most of the failure signal is already in a single forward pass. The eight-sample methods add about 0.01 AUROC internally and about 0.025 under distribution shift, at roughly nine times the compute, and the confidence intervals overlap. MC dropout and TTA perform equally well; TTA also gives slightly better Dice and calibration (5.6), so combining the two is a natural next step. Temperature scaling, fitted on validation voxels, slightly *lowered* failure detection. NeuroScope keeps MC dropout because it adds the per-voxel uncertainty map shown to the clinician, but the claim we make is about the validated flag, not about MC dropout being uniquely good.

### 5.6 Calibration (RQ3)

Reliability diagrams over all brain voxels: `docs/evidence/fig_calibration_test.png`, `fig_calibration_external.png`. ECE uses 15 equal-width bins.

| Region (internal test, all sequences) | ECE single pass | ECE MC dropout | ECE TTA | ECE temperature-scaled |
|---|---|---|---|---|
| Whole tumour | 0.0092 | 0.0089 | 0.0072 | 0.0247 |
| Tumour core | 0.0062 | 0.0059 | 0.0049 | 0.0581 |
| Enhancing tumour | 0.0037 | 0.0037 | 0.0032 | 0.0091 |

Overall ECE is low because most brain voxels are confidently background. That hides the clinically relevant part, so we also measured ECE only over voxels with predicted probability above 1/15 (the "uncertain zone" around the tumour):

| Uncertain-zone ECE, internal test, all sequences | Single pass | MC dropout | TTA | Temperature-scaled |
|---|---|---|---|---|
| Whole tumour | 0.052 | 0.050 | 0.040 | 0.025 |
| Tumour core | 0.084 | 0.080 | 0.066 | 0.073 |
| Enhancing tumour | 0.163 | 0.159 | 0.143 | 0.107 |

**Finding.** Inside the uncertain zone the raw model is *not* well calibrated: voxels predicted at 0.1 are tumour about 35 to 40% of the time, and voxels predicted at 0.9 only about 55 to 70% of the time (the reliability curve is flattened). MC dropout barely changes this; TTA improves it a little. Temperature scaling (fitted temperatures 2.5, 3.9 and 2.3 for WT, TC and ET) halves the uncertain-zone error for the whole tumour but worsens calibration of the confident background, so overall ECE gets worse. A likely cause is that the temperature was fitted on a tumour-enriched voxel sample; region-aware calibration is left as future work. **Consequence for the app:** the uncertainty map is shown as a relative measure of doubt, and probabilities are never presented to users as calibrated percentages.

### 5.7 External test: BraTS-Africa (RQ4)

60 BraTS-Africa 2023 patients with expert labels (BraTS 2023 label convention mapped to BraTS 2020), processed with the unchanged pipeline, model and frozen review threshold.

| | Internal test (BraTS 2020, 74) | External (BraTS-Africa, 60) |
|---|---|---|
| Dice WT, all sequences | 0.899 (0.879 to 0.914) | 0.876 (0.834 to 0.907) |
| Dice TC | 0.861 | 0.744 (0.675 to 0.806) |
| Dice ET | 0.774 | 0.735 (0.681 to 0.783) |
| Failures, all 15 sequence sets | 210 of 1,110 | 232 of 900 |
| Failure AUROC, all sets (MC dropout) | 0.954 (0.935 to 0.971) | 0.964 (0.947 to 0.980) |
| Frozen flag: failures caught | 195 of 210 | 226 of 232 |
| Frozen flag: specificity | 0.81 | 0.79 |
| Cases sent for review | 33% | 41% |
| Dice WT of accepted cases | 0.896 | 0.903 |

External Dice by number of sequences available (WT / TC / ET): 1 sequence 0.684 / 0.462 / 0.244; 2 sequences 0.800 / 0.607 / 0.439; 3 sequences 0.852 / 0.682 / 0.585; 4 sequences 0.876 / 0.744 / 0.735.

**Reading.** Whole-tumour segmentation transfers well (a drop of 0.02), tumour core drops more (0.12), consistent with the lower resolution and more variable contrast of these scans. The important result for the clinical claim is that the self-check transfers: with a threshold chosen on BraTS 2020 validation data it catches 97% of external failures, it sends more cases for review on the harder data (41% vs 33%), and the cases it accepts are as good as on internal data (Dice 0.903). This is one external population; it does not establish performance on raw scans from Indian hospitals.

### 5.8 Robustness to image degradation (74 test patients, all sequences)

| Corruption | WT Dice | Flag rate |
|---|---|---|
| None | 0.899 | 8% |
| Gaussian noise sd 0.1 / 0.3 / 0.6 | 0.897 / 0.857 / 0.621 | 7% / 14% / 46% |
| Contrast (gamma 0.7 / 1.4) | 0.893 / 0.881 | 8% / 11% |
| In-plane resolution / 2 / 4 | 0.889 / 0.857 | 11% / 23% |

The flag rate grows with the severity of the degradation, as intended.

### 5.9 Runtime

Measured on one NVIDIA Tesla T4 (Kaggle), PyTorch 2.11, mixed precision, batch of 32 slices; times cover the network passes for one preprocessed scan.

| Setting | Seconds per scan |
|---|---|
| Single pass | 0.37 |
| MC dropout x8 (default) | 3.4 |
| Test-time augmentation x8 | 3.1 |
| Free CPU server of the public demo (MC x4, includes pre- and post-processing) | about 45 |

Manual delineation of the three regions is commonly described as taking tens of minutes per scan; we did not time radiologists ourselves.

## 6. What is validated and what is a demonstration

| Feature | Status | Evidence |
|---|---|---|
| Tumour segmentation (WT, TC, ET) | Validated, internal test | 5.1, 5.2 |
| Any subset of sequences | Validated, internal test | 5.3 |
| Review flag / self-check | Validated, internal test, 15 conditions | 5.4, 5.5 |
| Calibration of probabilities | Measured: well calibrated overall, miscalibrated near the tumour boundary; uncertainty shown as relative doubt only | 5.6 |
| Generalisation to other scanners and populations | Tested on one external cohort (BraTS-Africa, 60 patients) | 5.7 |
| Volumes in mL | Follow from segmentation | 5.1 |
| RANO-style bidimensional measurement | Implemented to the RANO definition; **not** compared against radiologist measurements | Demonstration |
| Follow-up comparison | Works on any two scans, but shown with a **simulated** earlier scan because BraTS has one time point per patient; not validated on real repeat scans | Demonstration |
| Clinician correction (reject a region) | Engineering feature, recomputes all outputs | Demonstration |
| Hospital DICOM import | Tested on a synthetic DICOM study (series detection, registration, skull stripping); not tested on real hospital exports | Demonstration |
| Radiology-style report | Generated from the outputs above | Demonstration |

## 7. Limitations and threats to validity

1. **Single training source.** All training data are BraTS 2020: pre-processed, skull-stripped, co-registered, 1 mm isotropic, adult pre-operative glioma. Raw hospital scans must pass through the DICOM pipeline, which has not been clinically validated.
2. **Internal test set.** Our test patients come from the same BraTS distribution as training. The BraTS-Africa experiment addresses distribution shift for one external population only.
3. **Failures created by removing sequences.** Most of the 208 failures arise because sequences were removed. This is a realistic clinical situation, but failures from other causes (motion, unusual anatomy, post-operative change) are under-represented.
4. **2D model.** Slices are processed independently; a 3D model would likely improve boundary consistency. The comparison in 5.2 is among 2D networks trained with our recipe, not against the state of the art.
5. **No clinical reader study.** We have not measured whether the flag changes radiologist workload or accuracy in practice.
6. **Longitudinal tracking** is not validated (BraTS has no repeat scans). LUMIERE, a public longitudinal glioblastoma dataset with expert RANO ratings, is the planned next step.
7. NeuroScope is a research prototype and not a medical device.

## 8. Reproducibility

```bash
python scripts/preprocess.py --raw <BraTS2020 dir> --cache cache            # patient-level split, seed 42
python scripts/train.py --cache cache --out runs/base --epochs 20            # main model
python scripts/evaluate.py --cache cache --ckpt runs/base/best.pt --out runs/base/results
python analysis/failure_detection.py hosting/data/runs/base/results docs/evidence
python scripts/evidence.py --cache cache --ckpt hosting/data/runs/base/best.pt --ssa-root <BraTS-Africa dir> --out evidence
python scripts/train.py --cache cache --out runs/segresnet --arch segresnet   # baselines: segresnet, attention_unet
python analysis/evidence_report.py <unzipped evidence_artifacts.zip from the Kaggle notebook> docs/evidence
```

Kaggle notebooks: `notebooks/kaggle_train_eval.ipynb` (main model), `notebooks/kaggle_ablation.ipynb` (ablations), `notebooks/kaggle_evidence.ipynb` (Sections 5.2, 5.5 to 5.7, 5.9). Trained weights: `hosting/data/runs/base/best.pt`.

## 9. Result files

| File | Contents |
|---|---|
| `hosting/data/runs/base/results/summary.json` | Internal test metrics with CIs |
| `hosting/data/runs/base/results/test_per_case.csv`, `subsets_per_case.csv` | Per-patient and per-condition results |
| `hosting/data/runs/base/results/threshold.json` | Review threshold and how it was chosen |
| `app/ablation.json` | Modality-dropout and MC = 1 ablations |
| `docs/evidence/failure_detection.json`, `*.csv` | Section 5.4 |
| `docs/evidence/evidence.json`, `uncertainty_methods.csv`, `calibration.csv`, `architectures.csv`, `fig_*.png` | Sections 5.2, 5.5 to 5.7, 5.9 |
| `docs/evidence/raw/` | Raw Kaggle outputs behind those sections: per-case results, calibration bins, fitted temperatures, GPU environment, baseline summaries and training curves |

## References

- Adewole M, et al. The Brain Tumor Segmentation (BraTS) Challenge 2023: Glioma Segmentation in Sub-Saharan Africa Patient Population (BraTS-Africa). arXiv:2305.19369, 2023.
- Bakas S, et al. Advancing The Cancer Genome Atlas glioma MRI collections with expert segmentation labels and radiomic features. Scientific Data 4, 170117, 2017.
- Bakas S, et al. Identifying the best machine learning algorithms for brain tumor segmentation, progression assessment, and overall survival prediction in the BRATS challenge. arXiv:1811.02629, 2018.
- Ding Y, Yu X, Yang Y. RFNet: Region-aware fusion network for incomplete multi-modal brain tumor segmentation. ICCV 2021.
- Dorent R, Joutard S, Modat M, Ourselin S, Vercauteren T. Hetero-modal variational encoder-decoder for joint modality completion and segmentation. MICCAI 2019.
- Gal Y, Ghahramani Z. Dropout as a Bayesian approximation: representing model uncertainty in deep learning. ICML 2016.
- Geifman Y, El-Yaniv R. Selective classification for deep neural networks. NeurIPS 2017.
- Guo C, Pleiss G, Sun Y, Weinberger KQ. On calibration of modern neural networks. ICML 2017.
- Havaei M, Guizard N, Chapados N, Bengio Y. HeMIS: Hetero-modal image segmentation. MICCAI 2016.
- Isensee F, Jäger PF, Full PM, Vollmuth P, Maier-Hein KH. nnU-Net for brain tumor segmentation. BrainLes 2020 (arXiv:2011.00848).
- Isensee F, Jaeger PF, Kohl SAA, Petersen J, Maier-Hein KH. nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature Methods 18, 203-211, 2021.
- Jungo A, Reyes M. Assessing reliability and challenges of uncertainty estimations for medical image segmentation. MICCAI 2019.
- Lakshminarayanan B, Pritzel A, Blundell C. Simple and scalable predictive uncertainty estimation using deep ensembles. NeurIPS 2017.
- Mehta R, et al. QU-BraTS: MICCAI BraTS 2020 challenge on quantifying uncertainty in brain tumor segmentation. Journal of Machine Learning for Biomedical Imaging (MELBA), 2022:026.
- Menze BH, et al. The multimodal brain tumor image segmentation benchmark (BRATS). IEEE Transactions on Medical Imaging 34(10), 1993-2024, 2015.
- Myronenko A. 3D MRI brain tumor segmentation using autoencoder regularization. BrainLes 2018.
- Oktay O, et al. Attention U-Net: learning where to look for the pancreas. MIDL 2018.
- Ronneberger O, Fischer P, Brox T. U-Net: convolutional networks for biomedical image segmentation. MICCAI 2015.
- Wang G, Li W, Aertsen M, Deprest J, Ourselin S, Vercauteren T. Aleatoric uncertainty estimation with test-time augmentation for medical image segmentation with convolutional neural networks. Neurocomputing 338, 34-45, 2019.
- Wen PY, et al. Updated response assessment criteria for high-grade gliomas: Response Assessment in Neuro-Oncology working group. Journal of Clinical Oncology 28(11), 1963-1972, 2010.
- Zhang Y, et al. mmFormer: multimodal medical transformer for incomplete multimodal learning of brain tumor segmentation. MICCAI 2022.
