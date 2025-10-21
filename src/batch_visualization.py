"""
Batch effect visualization using PCA and UMAP

Usage:
------
python src/batch_visualization.py \
    --data ./data/dfs/data_bin_3.csv \
    --batches ./data/metadata.csv \
    --out ./results/batches
"""

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from umap import UMAP
from combatlearn import ComBat

warnings.filterwarnings("ignore")

SEED = 42
DPI = 300
FIGSIZE_SINGLE = (8, 6)
FIGSIZE_DOUBLE = (16, 6)
FIGSIZE_GRID = (16, 12)

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

# Colorblind-friendly palette for batches
BATCH_COLORS = [
    '#E69F00',  # Orange
    '#56B4E9',  # Sky blue
    '#009E73',  # Bluish green
]

def save_figure(fig, path: Path, formats=['png', 'pdf'], dpi=DPI):
    """Save figure in multiple formats."""
    base_path = path.with_suffix('')
    for fmt in formats:
        save_path = f"{base_path}.{fmt}"
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight', format=fmt)


def plot_2d_projection(X, batches, title, xlabel, ylabel, ax, batch_names=None):
    """Create a single 2D projection plot."""
    unique_batches = np.unique(batches)
    
    if batch_names is None:
        batch_names = {b: f"Batch {b}" for b in unique_batches}
    
    # Plot each batch
    for i, batch in enumerate(unique_batches):
        mask = batches == batch
        ax.scatter(X[mask, 0], X[mask, 1], 
                  c=BATCH_COLORS[i % len(BATCH_COLORS)],
                  label=batch_names[batch],
                  alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel(xlabel, fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=10)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
              frameon=True, fancybox=True, shadow=True)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)


def add_explained_variance(ax, pca):
    """Add explained variance to PCA plot."""
    var_exp = pca.explained_variance_ratio_
    ax.text(0.02, 0.98, 
            f'PC1: {var_exp[0]:.1%}\nPC2: {var_exp[1]:.1%}',
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


def create_pca_comparison(X_original, X_corrected, batches, batch_names, out_dir):
    """Create side-by-side PCA comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_DOUBLE, dpi=DPI)
    
    # Standardize data
    scaler1 = StandardScaler()
    scaler2 = StandardScaler()
    X_orig_scaled = scaler1.fit_transform(X_original)
    X_corr_scaled = scaler2.fit_transform(X_corrected)
    
    # PCA
    pca_orig = PCA(n_components=2, random_state=SEED)
    pca_corr = PCA(n_components=2, random_state=SEED)
    
    X_pca_orig = pca_orig.fit_transform(X_orig_scaled)
    X_pca_corr = pca_corr.fit_transform(X_corr_scaled)
    
    # Plot original
    plot_2d_projection(X_pca_orig, batches, 
                      'PCA - Original Data\n(Before ComBat)',
                      'PC1', 'PC2', ax1, batch_names)
    add_explained_variance(ax1, pca_orig)
    
    # Plot corrected
    plot_2d_projection(X_pca_corr, batches,
                      'PCA - ComBat Corrected\n(After ComBat)',
                      'PC1', 'PC2', ax2, batch_names)
    add_explained_variance(ax2, pca_corr)
    
    plt.suptitle('PCA Analysis: Batch Effect Correction', fontsize=18, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, out_dir / 'pca_comparison')
    plt.close()


def create_umap_comparison(X_original, X_corrected, batches, batch_names, out_dir):
    """Create side-by-side UMAP comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_DOUBLE, dpi=DPI)
    
    # Standardize data
    scaler1 = StandardScaler()
    scaler2 = StandardScaler()
    X_orig_scaled = scaler1.fit_transform(X_original)
    X_corr_scaled = scaler2.fit_transform(X_corrected)
    
    # UMAP
    umap_orig = UMAP(n_components=2, random_state=SEED, n_neighbors=30, min_dist=0.3)
    umap_corr = UMAP(n_components=2, random_state=SEED, n_neighbors=30, min_dist=0.3)
    
    X_umap_orig = umap_orig.fit_transform(X_orig_scaled)
    X_umap_corr = umap_corr.fit_transform(X_corr_scaled)
    
    # Plot original
    plot_2d_projection(X_umap_orig, batches,
                      'UMAP - Original Data\n(Before ComBat)',
                      'UMAP1', 'UMAP2', ax1, batch_names)
    
    # Plot corrected
    plot_2d_projection(X_umap_corr, batches,
                      'UMAP - ComBat Corrected\n(After ComBat)',
                      'UMAP1', 'UMAP2', ax2, batch_names)
    
    plt.suptitle('UMAP Analysis: Batch Effect Correction', fontsize=18, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, out_dir / 'umap_comparison')
    plt.close()


def visualize_batch_effects(data_path, batch_path, out_dir='batch_viz'):
    """Main function to visualize batch effects before and after ComBat."""
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
    
    # Prepare features (exclude both drug columns)
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
    
    # Apply ComBat correction
    print("\nApplying ComBat correction")
    combat = ComBat(batch=batches)
    X_corrected = combat.fit_transform(X)
    
    # Convert to numpy arrays
    X_original = X.values
    X_corrected = X_corrected.values if hasattr(X_corrected, 'values') else X_corrected
    
    # Create visualizations
    print("\nCreating visualizations:")
    
    # PCA
    print("  - PCA")
    create_pca_comparison(X_original, X_corrected, batches, batch_names, out_path)
    
    # UMAP
    print("  - UMAP")
    create_umap_comparison(X_original, X_corrected, batches, batch_names, out_path)
    
    print(f"\nVisualization complete! Results saved to: {out_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize batch effects using PCA and UMAP"
    )
    parser.add_argument("--data", required=True, help="Path to dataset CSV")
    parser.add_argument("--batches", required=True, help="Path to batch information CSV")
    parser.add_argument("--out", default="batches", help="Output directory")
    
    args = parser.parse_args()
    
    visualize_batch_effects(args.data, args.batches, args.out)