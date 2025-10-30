# MALDI-Kleb-AI

## Overview

This repository contains a comprehensive machine learning pipeline for predicting antimicrobial resistance (AMR) phenotypes from MALDI-TOF mass spectrometry data across multiple collection sites. Data used within this pipeline are MALDI-TOF spectra of *Klebsiella pneumoniae* and were collected in 3 Italian clinical centres. The framework implements cross-dataset evaluation strategies and batch effect correction methods to assess model generalizability and site-specific performance.

The MALDI-TOF spectra are processed using [MaldiAMRKit](https://github.com/EttoreRocchi/MaldiAMRKit). The batch-effect correction is performed with [combatlearn](https://github.com/EttoreRocchi/combatlearn) to avoid data leakage in the machine learning pipeline.

## Structure

```
.
├── src/
│   ├── prepare_maldiset.py               # Data preparation and pseudogel visualization
│   ├── cross_datasets_framework.py       # Cross-dataset generalization analysis
│   ├── aggregated_datasets_framework.py  # ComBat-corrected cross-validation
│   └── batch_visualization.py            # Batch effect visualization (PCA/UMAP)
├── results/                              # Analysis outputs (generated)
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/EttoreRocchi/MALDI-Kleb-AI.git
cd MALDI-Kleb-AI/
```

and install required packages using pip:

```bash
pip install -r requirements
```

## Usage

### Data preparation

Automated processing of MALDI-TOF spectra with metadata integration using [MaldiAMRKit](https://github.com/EttoreRocchi/MaldiAMRKit).

**Input requirements:**
- `spectra_dir`: Directory containing individual spectrum files (`.txt`, `.csv`)
- `metadata.csv`: CSV file with columns:
  - Sample identifiers (matching spectrum filenames)
  - Antibiotic susceptibility results (S/I/R format)
  - `City` column indicating collection site

**Command:**
```bash
python src/prepare_maldiset.py \
    --spectra_dir </path/to/data/> \
    --metadata </path/to/metadata> \
    --antibiotics Meropenem Amikacin \
    --other City \
    --output_dir ./data/dfs/
```

**Outputs:**
- `data_bin_3.csv`: Feature matrix (samples × mass-to-charge bins) with target labels
- `metadata_bin_3.csv`: Associated metadata including batch information
- `pseudogel_Meropenem.png`: Visualization of spectral intensities by resistance to Meropenem
- `pseudogel_Amikacin.png`: Visualization of spectral intensities by resistance to Amikacin

### Cross-dataset generalization analysis

Evaluate model generalization by training on individual sites and testing on all sites.

**Input requirements:**
- `data_bin_3.csv`: Feature matrix from preparation step
- `metadata_bin_3.csv`: Batch metadata with `City` column

**Command:**
```bash
python src/cross_datasets_framework.py \
    --data ./data/dfs/data_bin_3.csv \
    --batches ./data/dfs/metadata_bin_3.csv \
    --targets Meropenem Amikacin \
    --out_dir ./results/cross_datasets
```

**Outputs** (per target antibiotic):
```
results/cross_datasets/
└── Meropenem/
    ├── logistic_regression_f1.png          # Heatmap: F1-weighted scores
    ├── logistic_regression_f1.csv          # Numeric matrix
    ├── logistic_regression_auroc.png       # Heatmap: AUROC scores
    ├── logistic_regression_auroc.csv
    ├── logistic_regression_balacc.png      # Heatmap: Balanced accuracy
    ├── logistic_regression_balacc.csv
    ├── logistic_regression_mcc.png         # Heatmap: Matthews correlation coefficient
    ├── logistic_regression_mcc.csv
    ├── [similar files for random_forest, xgboost, mlp]
    └── ...
```

Each heatmap shows train city (rows) × test city (columns) performance.

### Batch-corrected cross-validation

Perform stratified cross-validation with ComBat batch effect correction using [combatlearn](https://github.com/EttoreRocchi/combatlearn).

**Input requirements:**
- Same as cross-dataset analysis
- Models are saved and can be reloaded unless `--force-retrain` is specified

**Command:**
```bash
python src/aggregated_datasets_framework.py \
    --data ./data/dfs/data_bin_3.csv \
    --batches ./data/dfs/metadata_bin_3.csv \
    --targets Meropenem Amikacin \
    --out ./results/aggregated_datasets \
    [--force-retrain]
```

**Outputs** (per target):
```
results/aggregated_datasets/
├── metrics_Meropenem.csv                                         # Aggregate metrics (all models)
├── metrics_detailed_logistic_regression_Meropenem.csv            # Per-fold metrics
├── confmat_Logistic_Regression_Meropenem.png                     # Confusion matrix
├── confmat_Logistic_Regression_Meropenem.pdf
├── shap_beeswarm_Logistic_Regression_Meropenem.png               # Feature importance
├── shap_beeswarm_Logistic_Regression_Meropenem.pdf
├── perf_by_center_logistic_regression_Meropenem.csv              # Center-specific metrics
├── perf_boxplot_Logistic_Regression_Meropenem_Balanced_Acc.png
├── perf_boxplot_Logistic_Regression_Meropenem_MCC.png
├── perf_boxplot_Logistic_Regression_Meropenem_AUROC.png
├── saved_models/
│   └── Meropenem/
│       ├── Logistic_Regression_fold0.pkl
│       ├── Logistic_Regression_fold1.pkl
│       └── ...
└── [similar files for other models and targets]
```

### Batch effect visualization

Visualize batch effects before and after ComBat correction using dimensionality reduction.

**Command:**
```bash
python src/batch_visualization.py \
    --data ./data/dfs/data_bin_3.csv \
    --batches ./data/dfs/metadata_bin_3.csv \
    --out ./results/batches
```

**Outputs:**
```
results/batches/
├── pca_comparison.png                    # Side-by-side PCA (before/after)
├── pca_comparison.pdf
├── umap_comparison.png                   # Side-by-side UMAP (before/after)
└── umap_comparison.pdf
```

## Citation

The dataset is publicly available on Zenodo at: [MALDI-Kleb-AI](https://zenodo.org/records/17405072).

If you use the dataset and/or the pipeline, please consider citing:

```bibtex
```
