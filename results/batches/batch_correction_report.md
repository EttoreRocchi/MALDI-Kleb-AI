# Batch Effect Correction: Quantitative Assessment

**Analysis Date:** 2025-12-09 18:41:05

## I. Overview

- **Number of Samples:** 743
- **Number of Features:** 6000
- **Number of Batches:** 3

- **Batch Names:** Catania, Milan, Rome


## II. Batch Effect Removal

### 2.1 Batch Alignment in Feature Space

| Metric | Before | After |
|:-------|-------:|------:|
| Mean centroid distance | 47.1198 ± 9.4612 | 1.9386 ± 0.4434 |
| Variance CV (mean) | 0.5332 | 0.0489 |
| Levene's statistic (median) | 19.6284 | 1.3581 |
| Levene's statistic (mean) | 25.5877 | 2.5794 |
| Between/within variance ratio | 0.1279 | 0.0001 |

**Interpretation:** Lower values indicate better batch alignment and reduced batch-associated variance.

### 2.2 Batch Separability in Embedding Space

| Metric | Before | After |
|:-------|-------:|------:|
| Silhouette score (PCA space) | 0.1725 | -0.0724 |
| Silhouette score (UMAP space) | 0.3340 | -0.0715 |

**Interpretation:** Lower silhouette scores indicate better batch mixing.

## III. Structure Preservation

### 3.1 Local Structure (k-NN Preservation)

| k | Mean Overlap | Std Dev |
|--:|-------------:|--------:|
| 5 | 0.6600 | 0.2586 |
| 10 | 0.6444 | 0.2349 |
| 20 | 0.6142 | 0.2145 |
| 30 | 0.5976 | 0.1982 |

**Interpretation:** Higher overlap values indicate better preservation of local neighborhood structure.

### 3.2 Global Structure (Distance Correlation)

- **Spearman correlation:** 0.7374

**Interpretation:** Higher correlation (closer to 1.0) indicates better preservation of overall distance structure.

## IV. Distribution Shape Preservation

| Metric | Mean | Median |
|:-------|-----:|-------:|
| Skewness change | 0.6593 | 0.1682 |
| Kolmogorov-Smirnov statistic | 0.2332 | 0.2746 |

**Interpretation:** Lower values indicate better preservation of original feature distributions.


## V. Per-Batch Diagnostics
### 5.1 Per-Batch Alignment

| Batch | Before | After |
|:------|-------:|------:|
| Catania | 44.1078 | 1.8678 |
| Milan | 29.0461 | 0.9959 |
| Rome | 17.3026 | 0.4375 |

**Interpretation:** Distance of each batch centroid to the global mean.


### 5.2 Per-Batch Structure Preservation

| Batch | Mean Overlap | Std Dev |
|:------|-------------:|--------:|
| Catania | 0.7345 | 0.1445 |
| Milan | 0.8158 | 0.1023 |
| Rome | 0.8568 | 0.0849 |

**Interpretation:** k-NN preservation (k=10) within each batch.


### 5.3 Per-Batch Distribution Similarity

| Batch | KS Statistic |
|:------|-------------:|
| Catania | 0.3357 |
| Milan | 0.2653 |
| Rome | 0.2671 |

**Interpretation:** Kolmogorov-Smirnov statistic for distribution similarity within each batch.

