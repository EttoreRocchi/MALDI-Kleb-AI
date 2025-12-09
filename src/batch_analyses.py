"""
Batch effect visualization and quantitative assessment using PCA and UMAP

Usage:
------
python src/batch_visualization.py \
    --data ./data/dfs/data_bin_3.csv \
    --batches ./data/metadata.csv \
    --out ./results/batches \
    --per-batch
"""

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jinja2 import Template
from scipy import stats
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from umap import UMAP
from combatlearn import ComBat

warnings.filterwarnings("ignore")


SEED = 42
DPI = 300
FIGSIZE_SINGLE = (8, 6)
FIGSIZE_DOUBLE = (16, 6)
FIGSIZE_GRID = (16, 12)

# Colorblind-friendly palette for batches
BATCH_COLORS = [
    '#E69F00',  # Orange
    '#56B4E9',  # Sky blue
    '#009E73',  # Bluish green
]

# Analysis parameters
KNN_K_VALUES = [5, 10, 20, 30]
DISTANCE_CORR_SAMPLE_SIZE = 1000

# Publication-ready style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.titlesize': 18,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.linewidth': 1.5,
    'axes.grid': True,
    'grid.alpha': 0.3,
})


REPORT_TEMPLATE = """# Batch Effect Correction: Quantitative Assessment

**Analysis Date:** {{ timestamp }}

## I. Overview

- **Number of Samples:** {{ n_samples }}
- **Number of Features:** {{ n_features }}
- **Number of Batches:** {{ n_batches }}
{% if batch_names %}
- **Batch Names:** {{ batch_names|join(', ') }}
{% endif %}

## II. Batch Effect Removal

### 2.1 Batch Alignment in Feature Space

| Metric | Before | After |
|:-------|-------:|------:|
| Mean centroid distance | {{ "%.4f"|format(metrics.alignment.orig.mean_centroid_dist) }} ± {{ "%.4f"|format(metrics.alignment.orig.mean_centroid_dist_std) }} | {{ "%.4f"|format(metrics.alignment.corr.mean_centroid_dist) }} ± {{ "%.4f"|format(metrics.alignment.corr.mean_centroid_dist_std) }} |
| Variance CV (mean) | {{ "%.4f"|format(metrics.alignment.orig.variance_cv_mean) }} | {{ "%.4f"|format(metrics.alignment.corr.variance_cv_mean) }} |
{%- if metrics.alignment.orig.levene_statistic_median is not none %}
| Levene's statistic (median) | {{ "%.4f"|format(metrics.alignment.orig.levene_statistic_median) }} | {{ "%.4f"|format(metrics.alignment.corr.levene_statistic_median) }} |
| Levene's statistic (mean) | {{ "%.4f"|format(metrics.alignment.orig.levene_statistic_mean) }} | {{ "%.4f"|format(metrics.alignment.corr.levene_statistic_mean) }} |
{%- endif %}
| Between/within variance ratio | {{ "%.4f"|format(metrics.variance.orig.between_within_ratio_mean) }} | {{ "%.4f"|format(metrics.variance.corr.between_within_ratio_mean) }} |

**Interpretation:** Lower values indicate better batch alignment and reduced batch-associated variance.

### 2.2 Batch Separability in Embedding Space

| Metric | Before | After |
|:-------|-------:|------:|
{%- if metrics.variance.orig.silhouette_score_pca is not none %}
| Silhouette score (PCA space) | {{ "%.4f"|format(metrics.variance.orig.silhouette_score_pca) }} | {{ "%.4f"|format(metrics.variance.corr.silhouette_score_pca) }} |
{%- endif %}
{%- if metrics.variance.orig.silhouette_score_umap is not none %}
| Silhouette score (UMAP space) | {{ "%.4f"|format(metrics.variance.orig.silhouette_score_umap) }} | {{ "%.4f"|format(metrics.variance.corr.silhouette_score_umap) }} |
{%- endif %}

**Interpretation:** Lower silhouette scores indicate better batch mixing.

## III. Structure Preservation

### 3.1 Local Structure (k-NN Preservation)

| k | Mean Overlap | Std Dev |
|--:|-------------:|--------:|
{%- for k in knn_k_values %}
| {{ k }} | {{ "%.4f"|format(metrics.structure['knn_k%d_mean'|format(k)]) }} | {{ "%.4f"|format(metrics.structure['knn_k%d_std'|format(k)]) }} |
{%- endfor %}

**Interpretation:** Higher overlap values indicate better preservation of local neighborhood structure.

### 3.2 Global Structure (Distance Correlation)

- **Spearman correlation:** {{ "%.4f"|format(metrics.structure.distance_correlation) }}

**Interpretation:** Higher correlation (closer to 1.0) indicates better preservation of overall distance structure.

## IV. Distribution Shape Preservation

| Metric | Mean | Median |
|:-------|-----:|-------:|
| Skewness change | {{ "%.4f"|format(metrics.distribution.skewness_change_mean) }} | {{ "%.4f"|format(metrics.distribution.skewness_change_median) }} |
{%- if metrics.distribution.ks_statistic_mean is not none %}
| Kolmogorov-Smirnov statistic | {{ "%.4f"|format(metrics.distribution.ks_statistic_mean) }} | {{ "%.4f"|format(metrics.distribution.ks_statistic_median) }} |
{%- endif %}

**Interpretation:** Lower values indicate better preservation of original feature distributions.
{% if per_batch_metrics and (per_batch_metrics.alignment or per_batch_metrics.knn or per_batch_metrics.distribution) %}

## V. Per-Batch Diagnostics

{%- if per_batch_metrics.alignment %}
### 5.1 Per-Batch Alignment

| Batch | Before | After |
|:------|-------:|------:|
{%- for batch_id, data in per_batch_metrics.alignment.items() %}
| {{ data.name }} | {{ "%.4f"|format(data.orig) }} | {{ "%.4f"|format(data.corr) }} |
{%- endfor %}

**Interpretation:** Distance of each batch centroid to the global mean.

{% endif -%}
{%- if per_batch_metrics.knn %}
### 5.2 Per-Batch Structure Preservation

| Batch | Mean Overlap | Std Dev |
|:------|-------------:|--------:|
{%- for batch_id, data in per_batch_metrics.knn.items() %}
| {{ data.name }} | {{ "%.4f"|format(data.mean) }} | {{ "%.4f"|format(data.std) }} |
{%- endfor %}

**Interpretation:** k-NN preservation (k=10) within each batch.

{% endif -%}
{%- if per_batch_metrics.distribution %}
### 5.3 Per-Batch Distribution Similarity

| Batch | KS Statistic |
|:------|-------------:|
{%- for batch_id, data in per_batch_metrics.distribution.items() %}
| {{ data.name }} | {{ "%.4f"|format(data.ks_statistic) }} |
{%- endfor %}

**Interpretation:** Kolmogorov-Smirnov statistic for distribution similarity within each batch.

{% endif -%}
{% endif %}
"""


class EmbeddingComputer:
    """Compute PCA and UMAP embeddings."""

    def __init__(self, random_state: int = SEED):
        """
        Initialize EmbeddingComputer.

        Parameters
        ----------
        random_state : int
            Random seed for reproducibility
        """
        self.random_state = random_state

    def compute_pca(self, X: np.ndarray, n_components: int = 2) -> Tuple[np.ndarray, PCA]:
        """
        Compute PCA embedding.

        Parameters
        ----------
        X : np.ndarray
            Data matrix (samples x features)
        n_components : int
            Number of principal components

        Returns
        -------
        embedding : np.ndarray
            PCA embedding (samples x n_components)
        model : PCA
            Fitted PCA model
        """
        pca = PCA(n_components=n_components, random_state=self.random_state)
        embedding = pca.fit_transform(X)
        return embedding, pca

    def compute_umap(
        self,
        X: np.ndarray,
        n_components: int = 2,
        n_neighbors: int = 30,
        min_dist: float = 0.3
    ) -> Tuple[np.ndarray, UMAP]:
        """
        Compute UMAP embedding.

        Parameters
        ----------
        X : np.ndarray
            Data matrix (samples x features)
        n_components : int
            Number of UMAP components
        n_neighbors : int
            Number of neighbors for UMAP
        min_dist : float
            Minimum distance for UMAP

        Returns
        -------
        embedding : np.ndarray
            UMAP embedding (samples x n_components)
        model : UMAP
            Fitted UMAP model
        """
        umap = UMAP(
            n_components=n_components,
            random_state=self.random_state,
            n_neighbors=n_neighbors,
            min_dist=min_dist
        )
        embedding = umap.fit_transform(X)
        return embedding, umap


class BatchMetrics:
    """
    Compute comprehensive batch effect metrics.

    Metrics are organized into four categories:
    1. Alignment: batch mean/variance alignment
    2. Variance: inter-batch vs within-batch variance
    3. Structure: k-NN preservation and distance correlation
    4. Distribution: distribution shape preservation (skewness, KS)
    """

    def __init__(
        self,
        X_orig_scaled: np.ndarray,
        X_corr_scaled: np.ndarray,
        batches: np.ndarray,
        X_orig_unscaled: Optional[np.ndarray] = None,
        X_corr_unscaled: Optional[np.ndarray] = None,
        X_pca_orig: Optional[np.ndarray] = None,
        X_pca_corr: Optional[np.ndarray] = None,
        X_umap_orig: Optional[np.ndarray] = None,
        X_umap_corr: Optional[np.ndarray] = None,
        batch_names: Optional[Dict] = None
    ):
        """
        Initialize BatchMetrics.

        Parameters
        ----------
        X_orig_scaled : np.ndarray
            Standardized original data (samples x features)
        X_corr_scaled : np.ndarray
            Standardized corrected data (samples x features)
        batches : np.ndarray
            Batch labels for each sample
        X_orig_unscaled : np.ndarray, optional
            Unscaled original data (samples x features) - used for KS tests
        X_corr_unscaled : np.ndarray, optional
            Unscaled corrected data (samples x features) - used for KS tests
        X_pca_orig : np.ndarray, optional
            PCA embedding of original data
        X_pca_corr : np.ndarray, optional
            PCA embedding of corrected data
        X_umap_orig : np.ndarray, optional
            UMAP embedding of original data
        X_umap_corr : np.ndarray, optional
            UMAP embedding of corrected data
        batch_names : dict, optional
            Mapping from batch ID to batch name
        """
        self.X_orig_scaled = X_orig_scaled
        self.X_corr_scaled = X_corr_scaled
        self.batches = batches
        self.unique_batches = np.unique(batches)

        # Unscaled data (for KS tests)
        self.X_orig_unscaled = X_orig_unscaled if X_orig_unscaled is not None else X_orig_scaled
        self.X_corr_unscaled = X_corr_unscaled if X_corr_unscaled is not None else X_corr_scaled

        # Embeddings (optional)
        self.X_pca_orig = X_pca_orig
        self.X_pca_corr = X_pca_corr
        self.X_umap_orig = X_umap_orig
        self.X_umap_corr = X_umap_corr

        # Batch names
        self.batch_names = batch_names or {b: f"Batch {b}" for b in self.unique_batches}

    def compute_alignment(self, per_batch: bool = False) -> Dict:
        """
        Compute batch alignment metrics (means and variances).

        Parameters
        ----------
        per_batch : bool
            Whether to compute per-batch metrics

        Returns
        -------
        dict
            Alignment metrics for original and corrected data
        """
        orig_stats = self._compute_batch_statistics(self.X_orig_scaled)
        corr_stats = self._compute_batch_statistics(self.X_corr_scaled)

        result = {
            'orig': orig_stats,
            'corr': corr_stats
        }

        # Per-batch analysis
        if per_batch:
            result['per_batch'] = self._compute_per_batch_centroids()

        return result

    def _compute_batch_statistics(self, X: np.ndarray) -> Dict:
        """Compute batch statistics for means and variances."""
        batch_means = []
        batch_vars = []

        for batch in self.unique_batches:
            mask = self.batches == batch
            batch_means.append(np.mean(X[mask], axis=0))
            batch_vars.append(np.var(X[mask], axis=0))

        batch_means = np.array(batch_means)
        batch_vars = np.array(batch_vars)

        # Mean centroid distances
        centroid_distances = pdist(batch_means)

        # Variance alignment (CV across batches)
        cv_vars = np.std(batch_vars, axis=0) / (np.mean(batch_vars, axis=0) + 1e-10)

        # Levene's test for variance homogeneity
        levene_stats = []
        for feat_idx in range(X.shape[1]):
            groups = [X[self.batches == b, feat_idx] for b in self.unique_batches]
            try:
                stat, _ = stats.levene(*groups)
                levene_stats.append(stat)
            except:
                levene_stats.append(np.nan)

        levene_stats = np.array([s for s in levene_stats if not np.isnan(s)])

        return {
            'mean_centroid_dist': np.mean(centroid_distances),
            'mean_centroid_dist_std': np.std(centroid_distances),
            'variance_cv_mean': np.mean(cv_vars),
            'levene_statistic_median': np.median(levene_stats) if len(levene_stats) > 0 else None,
            'levene_statistic_mean': np.mean(levene_stats) if len(levene_stats) > 0 else None,
        }

    def _compute_per_batch_centroids(self) -> Dict:
        """Compute per-batch centroid distances from global mean."""
        global_mean_orig = np.mean(self.X_orig_scaled, axis=0)
        global_mean_corr = np.mean(self.X_corr_scaled, axis=0)

        per_batch_metrics = {}

        for batch in self.unique_batches:
            mask = self.batches == batch

            # Batch centroids
            batch_mean_orig = np.mean(self.X_orig_scaled[mask], axis=0)
            batch_mean_corr = np.mean(self.X_corr_scaled[mask], axis=0)

            # Distance to global mean
            dist_orig = np.linalg.norm(batch_mean_orig - global_mean_orig)
            dist_corr = np.linalg.norm(batch_mean_corr - global_mean_corr)

            per_batch_metrics[batch] = {
                'name': self.batch_names[batch],
                'orig': dist_orig,
                'corr': dist_corr
            }

        return per_batch_metrics

    def compute_interbatch_variance(self, per_batch: bool = False) -> Dict:
        """
        Compute between-batch vs within-batch variance metrics.

        Parameters
        ----------
        per_batch : bool
            Whether to compute per-batch metrics (not used for this metric)

        Returns
        -------
        dict
            Variance metrics for original and corrected data
        """
        orig_var = self._compute_batch_variance_ratio(self.X_orig_scaled)
        corr_var = self._compute_batch_variance_ratio(self.X_corr_scaled)

        # Silhouette scores
        orig_var['silhouette_score_pca'] = self._compute_silhouette(self.X_pca_orig) if self.X_pca_orig is not None else None
        corr_var['silhouette_score_pca'] = self._compute_silhouette(self.X_pca_corr) if self.X_pca_corr is not None else None
        orig_var['silhouette_score_umap'] = self._compute_silhouette(self.X_umap_orig) if self.X_umap_orig is not None else None
        corr_var['silhouette_score_umap'] = self._compute_silhouette(self.X_umap_corr) if self.X_umap_corr is not None else None

        result = {
            'orig': orig_var,
            'corr': corr_var
        }

        return result

    def _compute_batch_variance_ratio(self, X: np.ndarray) -> Dict:
        """Compute between-batch vs within-batch variance ratio."""
        grand_mean = np.mean(X, axis=0)
        total_var = np.var(X, axis=0)

        n_total = len(X)

        # Between-batch variance
        between_var = np.zeros(X.shape[1])
        for batch in self.unique_batches:
            mask = self.batches == batch
            n_batch = np.sum(mask)
            batch_mean = np.mean(X[mask], axis=0)
            between_var += (n_batch / n_total) * (batch_mean - grand_mean) ** 2

        # Within-batch variance
        within_var = total_var - between_var

        # Variance ratio (higher = more batch effect)
        var_ratio = between_var / (within_var + 1e-10)

        return {
            'between_within_ratio_mean': np.mean(var_ratio)
        }

    def _compute_silhouette(self, X_embedded: np.ndarray) -> Optional[float]:
        """Compute silhouette score using batch labels."""
        if len(self.unique_batches) < 2 or X_embedded is None:
            return None

        try:
            return silhouette_score(X_embedded, self.batches, metric='euclidean')
        except:
            return None

    def compute_structure(
        self,
        k_values: List[int] = KNN_K_VALUES,
        sample_size: int = DISTANCE_CORR_SAMPLE_SIZE,
        per_batch: bool = False
    ) -> Dict:
        """
        Compute structure preservation metrics (k-NN, distance correlation).

        Parameters
        ----------
        k_values : list
            List of k values for k-NN analysis
        sample_size : int
            Sample size for distance correlation
        per_batch : bool
            Whether to compute per-batch metrics

        Returns
        -------
        dict
            Structure preservation metrics
        """
        # k-NN preservation
        knn_metrics = self._compute_knn_preservation(k_values)

        # Distance correlation
        dist_corr = self._compute_distance_correlation(sample_size)

        result = {**knn_metrics, **dist_corr}

        # Per-batch k-NN preservation (use k=10)
        if per_batch:
            result['per_batch_knn'] = self._compute_per_batch_knn(k=10)

        return result

    def _compute_knn_preservation(self, k_values: List[int]) -> Dict:
        """Measure how well k-NN structure is preserved for multiple k values."""
        results = {}

        for k in k_values:
            # Find k nearest neighbors in original space
            nbrs_orig = NearestNeighbors(n_neighbors=k+1).fit(self.X_orig_scaled)
            _, indices_orig = nbrs_orig.kneighbors(self.X_orig_scaled)

            # Find k nearest neighbors in corrected space
            nbrs_corr = NearestNeighbors(n_neighbors=k+1).fit(self.X_corr_scaled)
            _, indices_corr = nbrs_corr.kneighbors(self.X_corr_scaled)

            # Compute overlap (exclude self)
            overlaps = []
            for i in range(len(self.X_orig_scaled)):
                neighbors_orig = set(indices_orig[i, 1:])
                neighbors_corr = set(indices_corr[i, 1:])
                overlap = len(neighbors_orig & neighbors_corr) / k
                overlaps.append(overlap)

            results[f'knn_k{k}_mean'] = np.mean(overlaps)
            results[f'knn_k{k}_std'] = np.std(overlaps)

        return results

    def _compute_distance_correlation(self, sample_size: int) -> Dict:
        """Compute correlation between pairwise distance matrices."""
        # Subsample for efficiency
        if len(self.X_orig_scaled) > sample_size:
            idx = np.random.choice(len(self.X_orig_scaled), sample_size, replace=False)
            X_orig_sub = self.X_orig_scaled[idx]
            X_corr_sub = self.X_corr_scaled[idx]
        else:
            X_orig_sub = self.X_orig_scaled
            X_corr_sub = self.X_corr_scaled

        # Compute pairwise distances
        dist_orig = pdist(X_orig_sub)
        dist_corr = pdist(X_corr_sub)

        # Spearman correlation
        corr, _ = stats.spearmanr(dist_orig, dist_corr)

        return {
            'distance_correlation': corr
        }

    def _compute_per_batch_knn(self, k: int = 10) -> Dict:
        """Compute per-batch k-NN preservation."""
        per_batch_metrics = {}

        for batch in self.unique_batches:
            mask = self.batches == batch
            n_batch = np.sum(mask)

            # Need enough samples for k-NN
            if n_batch <= k:
                continue

            # Extract batch data
            X_orig_batch = self.X_orig_scaled[mask]
            X_corr_batch = self.X_corr_scaled[mask]

            # Find k nearest neighbors
            nbrs_orig = NearestNeighbors(n_neighbors=k+1).fit(X_orig_batch)
            _, indices_orig = nbrs_orig.kneighbors(X_orig_batch)

            nbrs_corr = NearestNeighbors(n_neighbors=k+1).fit(X_corr_batch)
            _, indices_corr = nbrs_corr.kneighbors(X_corr_batch)

            # Compute overlaps
            overlaps = []
            for i in range(len(X_orig_batch)):
                neighbors_orig = set(indices_orig[i, 1:])
                neighbors_corr = set(indices_corr[i, 1:])
                overlap = len(neighbors_orig & neighbors_corr) / k
                overlaps.append(overlap)

            per_batch_metrics[batch] = {
                'name': self.batch_names[batch],
                'mean': np.mean(overlaps),
                'std': np.std(overlaps)
            }

        return per_batch_metrics

    def compute_distribution_shape(self, per_batch: bool = False) -> Dict:
        """
        Compute distribution shape preservation metrics (skewness, KS).

        Parameters
        ----------
        per_batch : bool
            Whether to compute per-batch metrics

        Returns
        -------
        dict
            Distribution shape metrics
        """
        # Global metrics - skewness on scaled data
        skew_orig = stats.skew(self.X_orig_scaled, axis=0)
        skew_corr = stats.skew(self.X_corr_scaled, axis=0)

        skewness_changes = np.abs(skew_corr - skew_orig)

        # KS test for distribution similarity - use unscaled data
        ks_stats = []
        for i in range(self.X_orig_unscaled.shape[1]):
            try:
                ks_stat, _ = stats.ks_2samp(
                    self.X_orig_unscaled[:, i],
                    self.X_corr_unscaled[:, i]
                )
                ks_stats.append(ks_stat)
            except:
                pass

        result = {
            'skewness_change_mean': np.mean(skewness_changes),
            'skewness_change_median': np.median(skewness_changes),
            'ks_statistic_mean': np.mean(ks_stats) if len(ks_stats) > 0 else None,
            'ks_statistic_median': np.median(ks_stats) if len(ks_stats) > 0 else None
        }

        # Per-batch metrics
        if per_batch:
            result['per_batch'] = self._compute_per_batch_distribution()

        return result

    def _compute_per_batch_distribution(self) -> Dict:
        """Compute per-batch distribution changes."""
        per_batch_metrics = {}

        for batch in self.unique_batches:
            mask = self.batches == batch

            X_orig_batch_scaled = self.X_orig_scaled[mask]
            X_corr_batch_scaled = self.X_corr_scaled[mask]

            # Use unscaled data for KS tests
            X_orig_batch_unscaled = self.X_orig_unscaled[mask]
            X_corr_batch_unscaled = self.X_corr_unscaled[mask]

            # Skewness changes (on scaled data)
            skew_orig = stats.skew(X_orig_batch_scaled, axis=0)
            skew_corr = stats.skew(X_corr_batch_scaled, axis=0)
            skewness_change = np.mean(np.abs(skew_corr - skew_orig))

            # KS statistics (on unscaled data)
            ks_stats = []
            for i in range(X_orig_batch_unscaled.shape[1]):
                try:
                    ks_stat, _ = stats.ks_2samp(X_orig_batch_unscaled[:, i], X_corr_batch_unscaled[:, i])
                    ks_stats.append(ks_stat)
                except:
                    pass

            per_batch_metrics[batch] = {
                'name': self.batch_names[batch],
                'skewness_change': skewness_change,
                'ks_statistic': np.mean(ks_stats) if len(ks_stats) > 0 else np.nan
            }

        return per_batch_metrics

    def compute_all(self, per_batch: bool = False) -> Dict:
        """
        Compute all metrics.

        Parameters
        ----------
        per_batch : bool
            Whether to compute per-batch metrics

        Returns
        -------
        dict
            All computed metrics
        """
        alignment = self.compute_alignment(per_batch=per_batch)
        variance = self.compute_interbatch_variance(per_batch=per_batch)
        structure = self.compute_structure(per_batch=per_batch)
        distribution = self.compute_distribution_shape(per_batch=per_batch)

        metrics = {
            'alignment': alignment,
            'variance': variance,
            'structure': structure,
            'distribution': distribution
        }

        # Extract per-batch metrics if requested
        per_batch_metrics = None
        if per_batch:
            per_batch_metrics = {
                'alignment': alignment.get('per_batch'),
                'knn': structure.get('per_batch_knn'),
                'distribution': distribution.get('per_batch')
            }

        return metrics, per_batch_metrics


def save_figure(fig: plt.Figure, path: Path, formats: List[str] = ['png', 'pdf'], dpi: int = DPI):
    """
    Save figure in multiple formats.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save
    path : Path
        Base path for saving (without extension)
    formats : list
        List of formats to save
    dpi : int
        Resolution for raster formats
    """
    base_path = path.with_suffix('')
    for fmt in formats:
        save_path = f"{base_path}.{fmt}"
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight', format=fmt)


def plot_2d_projection(
    X: np.ndarray,
    batches: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    ax: plt.Axes,
    batch_names: Optional[Dict] = None
):
    """
    Create a single 2D projection plot.

    Parameters
    ----------
    X : np.ndarray
        2D embedding (samples x 2)
    batches : np.ndarray
        Batch labels
    title : str
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    ax : matplotlib.axes.Axes
        Axes to plot on
    batch_names : dict, optional
        Mapping from batch ID to batch name
    """
    unique_batches = np.unique(batches)

    if batch_names is None:
        batch_names = {b: f"Batch {b}" for b in unique_batches}

    # Plot each batch
    for i, batch in enumerate(unique_batches):
        mask = batches == batch
        ax.scatter(
            X[mask, 0], X[mask, 1],
            c=BATCH_COLORS[i % len(BATCH_COLORS)],
            label=batch_names[batch],
            alpha=0.7, s=50, edgecolors='black', linewidth=0.5
        )

    ax.set_xlabel(xlabel, fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=10)
    ax.legend(
        bbox_to_anchor=(1.05, 1), loc='upper left',
        frameon=True, fancybox=True, shadow=True
    )

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)


def add_explained_variance(ax: plt.Axes, pca: PCA):
    """
    Add explained variance annotation to PCA plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to annotate
    pca : PCA
        Fitted PCA model
    """
    var_exp = pca.explained_variance_ratio_
    ax.text(
        0.02, 0.98,
        f'PC1: {var_exp[0]:.1%}\nPC2: {var_exp[1]:.1%}',
        transform=ax.transAxes, fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )


def create_comparison_plot(
    X_orig: np.ndarray,
    X_corr: np.ndarray,
    batches: np.ndarray,
    plot_type: str,
    pca_orig: Optional[PCA] = None,
    pca_corr: Optional[PCA] = None,
    batch_names: Optional[Dict] = None
) -> plt.Figure:
    """
    Create side-by-side comparison plot (PCA or UMAP).

    Parameters
    ----------
    X_orig : np.ndarray
        Original embedding
    X_corr : np.ndarray
        Corrected embedding
    batches : np.ndarray
        Batch labels
    plot_type : str
        Either 'PCA' or 'UMAP'
    pca_orig : PCA, optional
        Fitted PCA model for original data (for variance annotation)
    pca_corr : PCA, optional
        Fitted PCA model for corrected data (for variance annotation)
    batch_names : dict, optional
        Mapping from batch ID to batch name

    Returns
    -------
    fig : matplotlib.figure.Figure
        Created figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_DOUBLE, dpi=DPI)

    # Labels
    if plot_type.upper() == 'PCA':
        xlabel, ylabel = 'PC1', 'PC2'
    elif plot_type.upper() == 'UMAP':
        xlabel, ylabel = 'UMAP1', 'UMAP2'
    else:
        xlabel, ylabel = f'{plot_type}1', f'{plot_type}2'

    # Plot original
    plot_2d_projection(
        X_orig, batches,
        f'{plot_type} - Original Data\n(Before ComBat)',
        xlabel, ylabel, ax1, batch_names
    )
    if plot_type.upper() == 'PCA' and pca_orig is not None:
        add_explained_variance(ax1, pca_orig)

    # Plot corrected
    plot_2d_projection(
        X_corr, batches,
        f'{plot_type} - ComBat Corrected\n(After ComBat)',
        xlabel, ylabel, ax2, batch_names
    )
    if plot_type.upper() == 'PCA' and pca_corr is not None:
        add_explained_variance(ax2, pca_corr)

    plt.suptitle(
        f'{plot_type} Analysis: Batch Effect Correction',
        fontsize=18, fontweight='bold'
    )
    plt.tight_layout()

    return fig


class ReportGenerator:
    """Generate markdown reports using Jinja2 templates."""

    def __init__(self, template_str: str = REPORT_TEMPLATE):
        """
        Initialize ReportGenerator.

        Parameters
        ----------
        template_str : str
            Jinja2 template string
        """
        self.template = Template(template_str)

    def generate(
        self,
        metrics: Dict,
        per_batch_metrics: Optional[Dict],
        n_samples: int,
        n_features: int,
        n_batches: int,
        batch_names: Optional[List[str]] = None,
        knn_k_values: List[int] = KNN_K_VALUES
    ) -> str:
        """
        Generate markdown report from metrics.

        Parameters
        ----------
        metrics : dict
            Computed metrics dictionary
        per_batch_metrics : dict, optional
            Per-batch metrics dictionary
        n_samples : int
            Number of samples
        n_features : int
            Number of features
        n_batches : int
            Number of batches
        batch_names : list, optional
            List of batch names
        knn_k_values : list
            List of k values used for k-NN analysis

        Returns
        -------
        str
            Generated markdown report
        """
        timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')

        context = {
            'timestamp': timestamp,
            'n_samples': n_samples,
            'n_features': n_features,
            'n_batches': n_batches,
            'batch_names': batch_names,
            'metrics': metrics,
            'per_batch_metrics': per_batch_metrics,
            'knn_k_values': knn_k_values
        }

        return self.template.render(**context)


def visualize_batch_effects(
    data_path: str,
    batch_path: str,
    out_dir: str = 'batch_viz',
    per_batch: bool = False
):
    """
    Main function to visualize batch effects before and after ComBat.

    Parameters
    ----------
    data_path : str
        Path to data CSV file
    batch_path : str
        Path to batch information CSV file
    out_dir : str
        Output directory for results
    per_batch : bool
        Whether to compute per-batch metrics
    """
    # Create output directory
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading data")
    df = pd.read_csv(data_path, index_col=0)
    batch_df = pd.read_csv(batch_path, index_col=0)

    # Extract batch information
    if 'City' in batch_df.columns:
        batches = batch_df['City']
        batch_names = {b: b for b in np.unique(batches)}
    else:
        batches = batch_df.iloc[:, 0]
        batch_names = {b: f"Batch {b}" for b in np.unique(batches)}

    # Prepare features (exclude drug columns)
    columns_to_drop = []
    if 'Meropenem' in df.columns:
        columns_to_drop.append('Meropenem')
    if 'Amikacin' in df.columns:
        columns_to_drop.append('Amikacin')

    X = df.drop(columns=columns_to_drop)

    # Ensure batches align with X
    batches = batches.loc[X.index]

    print(f"\nData shape: {X.shape}")
    print(f"Number of batches: {len(np.unique(batches))}")
    print(f"Batches: {list(batch_names.values())}")
    if per_batch:
        print("Per-batch analysis: ENABLED")

    # Apply ComBat correction
    print("\nApplying ComBat correction")
    combat = ComBat(batch=batches)
    X_corrected = combat.fit_transform(X)

    # Convert to numpy arrays
    X_original = X.values
    X_corrected = X_corrected.values if hasattr(X_corrected, 'values') else X_corrected

    # Standardize data once for all metrics
    print("\nPreparing data for analysis:")
    scaler_orig = StandardScaler()
    scaler_corr = StandardScaler()
    X_orig_scaled = scaler_orig.fit_transform(X_original)
    X_corr_scaled = scaler_corr.fit_transform(X_corrected)

    # Compute embeddings
    print("  - Computing embeddings")
    embedder = EmbeddingComputer(random_state=SEED)

    X_pca_orig, pca_orig = embedder.compute_pca(X_orig_scaled)
    X_pca_corr, pca_corr = embedder.compute_pca(X_corr_scaled)

    X_umap_orig, umap_orig = embedder.compute_umap(X_orig_scaled)
    X_umap_corr, umap_corr = embedder.compute_umap(X_corr_scaled)

    # Compute metrics
    print("  - Computing metrics")
    batch_metrics = BatchMetrics(
        X_orig_scaled=X_orig_scaled,
        X_corr_scaled=X_corr_scaled,
        batches=batches.values,
        X_orig_unscaled=X_original,
        X_corr_unscaled=X_corrected,
        X_pca_orig=X_pca_orig,
        X_pca_corr=X_pca_corr,
        X_umap_orig=X_umap_orig,
        X_umap_corr=X_umap_corr,
        batch_names=batch_names
    )

    metrics, per_batch_metrics = batch_metrics.compute_all(per_batch=per_batch)

    # Create visualizations
    print("\nCreating visualizations:")

    # PCA
    print("  - PCA plots")
    fig_pca = create_comparison_plot(
        X_pca_orig, X_pca_corr, batches.values,
        plot_type='PCA',
        pca_orig=pca_orig,
        pca_corr=pca_corr,
        batch_names=batch_names
    )
    save_figure(fig_pca, out_path / 'pca_comparison')
    plt.close(fig_pca)

    # UMAP
    print("  - UMAP plots")
    fig_umap = create_comparison_plot(
        X_umap_orig, X_umap_corr, batches.values,
        plot_type='UMAP',
        batch_names=batch_names
    )
    save_figure(fig_umap, out_path / 'umap_comparison')
    plt.close(fig_umap)

    # Generate report
    print("  - Quantitative assessment report")
    report_gen = ReportGenerator()
    report_text = report_gen.generate(
        metrics=metrics,
        per_batch_metrics=per_batch_metrics,
        n_samples=len(X_original),
        n_features=X_original.shape[1],
        n_batches=len(np.unique(batches)),
        batch_names=list(batch_names.values()) if batch_names else None,
        knn_k_values=KNN_K_VALUES
    )

    # Save report
    report_path = out_path / "batch_correction_report.md"
    with open(report_path, 'w') as f:
        f.write(report_text)

    print(f"\nAnalysis complete! Results saved to: {out_path}")
    print(f"  - Visualizations: pca_comparison.png/pdf, umap_comparison.png/pdf")
    print(f"  - Quantitative report: batch_correction_report.md\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize and quantify batch effects using PCA, UMAP, and statistical metrics"
    )
    parser.add_argument("--data", required=True, help="Path to dataset CSV")
    parser.add_argument("--batches", required=True, help="Path to batch information CSV")
    parser.add_argument("--out", default="batches", help="Output directory")
    parser.add_argument(
        "--per-batch",
        action="store_true",
        help="Enable per-batch analysis (computes metrics for each batch individually)"
    )

    args = parser.parse_args()

    visualize_batch_effects(
        data_path=args.data,
        batch_path=args.batches,
        out_dir=args.out,
        per_batch=args.per_batch
    )
