# CSBA-MRI: spatially resolved cervical spine age from routine MRI

Code for "A spatially resolved MRI clock of cervical spine aging localizes degenerative disease and relates to frailty".

DenseAgeNet is a 3D residual encoder–decoder. It takes a level-normalized sagittal T2-weighted cervical MRI (C2–T1) and estimates age and log-variance at every voxel. The voxel estimates are pooled by precision weighting into whole-spine, tissue and level ages, which are then calibrated. The spine age gap is the estimated minus the chronological age.

Trained weights will be released upon publication. No imaging, clinical or proteomic data are included.

## Installation

```bash
pip install -r requirements.txt
```

Segmentation uses [TotalSpineSeg](https://github.com/neuropoly/totalspineseg), which is installed separately. Cord compression measures were computed with the [Spinal Cord Toolbox](https://spinalcordtoolbox.com) and are read as inputs.

## Configuration

Paths are set with environment variables:

| Variable | Contents | Default |
|---|---|---|
| `CSBA_RAW` | NIfTI images and TotalSpineSeg label maps | `data/raw` |
| `CSBA_DATA` | Dataset metadata and clinical and proteomic tables | `data` |
| `CSBA_ROOT` | Working directory for outputs (`npz/`, `models/`, `preds/`, `tables/`, `figures/`, `logs/`) | `work` |
| `CSBA_FONTS` | Optional folder with Arial TTF files for figures | — |

## Pipeline

Scripts are in `csba/` and run in numerical order.

| Step | Script | Purpose |
|---|---|---|
| Cohort | `c01_cohort.py` | Assemble image list, labels and roles across data sources |
| Preprocessing | `c02_extract.py [main\|jitter\|thick\|uniform\|n4]` | Cord-centerline straightening, C2–T1 level normalization (192 × 96 × 96, 0.6 mm), N4, intensity standardization |
| Splits | `c03a_splits.py` | Internal test set and 5-fold cross-fitting (stratified by dataset × age band, seed 42) |
| Training | `c03_train.py --fold K [--variant main] [--gpu 0]` | Train DenseAgeNet (`model.py`) for one fold |
| Inference | `c04_infer.py --fold K [--maps] [--model_dir DIR]` | Voxel-wise age maps and pooled region ages |
| Calibration | `c05_calibrate.py TAG` | Quadratic calibration fitted on out-of-fold healthy predictions |
| Analyses | `c06`–`c25`, `c28` | Accuracy, pre-specified contrasts, validation, level and within-person analyses, marrow, harmonization (ComBat), sensitivity analyses, proteomic enrichment |
| Figures and tables | `c11`, `c16`, `c18`, `c14`, `c17` | Manuscript figures and supplementary tables |

`run_all.sh` reruns calibration and all downstream analyses for the primary model. `jobrunner.py` queues training and inference jobs across GPUs.

### Applying a trained model to new scans

1. Segment each sagittal T2-weighted scan with TotalSpineSeg.
2. Run `c02_extract.py` to produce the level-normalized volume.
3. Run `c04_infer.py --fold K --model_dir <weights>` for each fold, average the fold predictions and apply the calibration from `c05_calibrate.py`.

## Data

Public datasets:
- CSpineSeg
- Spine Generic
- OpenNeuro [ds005616](https://doi.org/10.18112/openneuro.ds005616.v1.1.4)

Clinical data from Johns Hopkins University and the Medical College of Wisconsin cannot be shared publicly because of patient privacy.
