"""
Cross-validated pipeline with ComBat batch correction

Usage
-----
python src/aggregated_datasets_framework.py \
    --data ./data/dfs/data_bin_3.csv \
    --batches ./data/dfs/metadata_bin_3.csv \
    --targets Meropenem Amikacin \
    --out ./results/aggregated_datasets
    [--force-retrain]
"""

import argparse
import pickle
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from xgboost import XGBClassifier
from combatlearn import ComBat

warnings.filterwarnings("ignore")

# Global settings
SEED = 42
N_SPLITS_OUTER = 10
N_SPLITS_INNER = 5
N_BOOTSTRAP = 10_000
DPI = 300
FIGSIZE_STANDARD = (8, 6)
FIGSIZE_SQUARE = (6, 6)

# Fixed center order for consistent plotting
CENTER_ORDER = ['Catania', 'Rome', 'Milan']

# Center-specific color mapping for consistent visualization
CENTER_COLORS = {
    'Catania': '#0173B2',  # Blue
    'Rome': '#DE8F05',     # Orange
    'Milan': '#029E73',    # Green
}

# Publication-ready style settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# Hyperparameter grids
param_grids: Dict[str, Dict[str, List]] = {
    "Random Forest": {
        "clf__n_estimators": [100, 200],
        "clf__criterion": ["gini", "entropy"],
        "clf__max_features": ["auto", "sqrt", "log2"],
        "clf__max_depth": [3, 5, 7],
    },
    "Logistic Regression": {
        "clf__C": [0.1, 1, 10, 100],
        "clf__penalty": ["l1", "l2"],
        "clf__solver": ["liblinear"],
    },
    "XGBoost": {
        "clf__booster": ["gbtree", "dart"],
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [3, 5, 7],
    },
    "MLP": {
        "clf__hidden_layer_sizes": [(256, 128), (128, 64)],
        "clf__alpha": [1e-4, 1e-5],
    },
}

def save_figure(fig, path: Path, formats=['png', 'pdf'], dpi=DPI):
    """Save figure in multiple formats with high quality."""
    base_path = path.with_suffix('')
    for fmt in formats:
        save_path = f"{base_path}.{fmt}"
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight', format=fmt)
        if fmt == 'pdf':
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight',
                       format=fmt, backend='pdf')


def bootstrap_ci(data: np.ndarray, n_bootstrap: int = N_BOOTSTRAP,
                 ci_percentile: float = 95, seed: int = SEED) -> tuple[float, float]:
    """
    Calculate nonparametric bootstrap confidence interval for the mean.

    Parameters
    ----------
    data : np.ndarray
        1D array of observed values
    n_bootstrap : int
        Number of bootstrap resamples (default: 10,000)
    ci_percentile : float
        Confidence interval percentile (default: 95 for 95% CI)
    seed : int
        Random seed for reproducibility

    Returns
    -------
    tuple[float, float]
        Lower and upper bounds of the confidence interval
    """
    rng = np.random.RandomState(seed)
    n = len(data)

    # Generate bootstrap resamples and compute means
    bootstrap_means = np.array([
        np.mean(rng.choice(data, size=n, replace=True))
        for _ in range(n_bootstrap)
    ])

    # Calculate percentiles for CI
    alpha = (100 - ci_percentile) / 2
    lower = np.percentile(bootstrap_means, alpha)
    upper = np.percentile(bootstrap_means, 100 - alpha)

    return lower, upper


def metric_dict(y_true, y_pred, y_prob):
    """Calculate comprehensive metrics for binary classification."""
    return {
        "Precision (0)": precision_score(y_true, y_pred, pos_label=0),
        "Precision (1)": precision_score(y_true, y_pred, pos_label=1),
        "Recall (0)": recall_score(y_true, y_pred, pos_label=0),
        "Recall (1)": recall_score(y_true, y_pred, pos_label=1),
        "F1 (0)": f1_score(y_true, y_pred, pos_label=0),
        "F1 (1)": f1_score(y_true, y_pred, pos_label=1),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "Balanced Acc": balanced_accuracy_score(y_true, y_pred),
        "AUROC": roc_auc_score(y_true, y_prob[:, 1]),
    }


def plot_conf_matrix(cm: np.ndarray, model_name: str, target: str, out_dir: Path):
    """Create confusion matrix plot."""
    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE, dpi=DPI)
    ax.grid(False)
    
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues', vmin=0.0, vmax=1.0)
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Normalized frequency', rotation=270, labelpad=20)
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = ax.text(j, i, f'{cm[i, j]:.3f}',
                          ha="center", va="center",
                          color="white" if cm[i, j] > thresh else "black",
                          fontsize=14, fontweight='bold')
    
    ax.set_title(f'Confusion Matrix - {model_name}\nTarget: {target}', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Predicted label', fontsize=14)
    ax.set_ylabel('True label', fontsize=14)
    
    ax.set_xticks(np.arange(cm.shape[1]))
    ax.set_yticks(np.arange(cm.shape[0]))
    ax.set_xticklabels(['Negative (0)', 'Positive (1)'])
    ax.set_yticklabels(['Negative (0)', 'Positive (1)'])
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    ax.set_xticks(np.arange(cm.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(cm.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", size=0)
    
    plt.tight_layout()
    save_figure(fig, out_dir / f"confmat_{model_name.replace(' ', '_')}_{target}")
    plt.close()


def plot_performance_boxplots(perf_df: pd.DataFrame, metric: str, model_name: str, 
                            target: str, out_dir: Path):
    """Create boxplots."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=DPI)
    
    # Use fixed center order for consistency across all plots
    available_centers = [center for center in CENTER_ORDER if center in perf_df['Centre'].unique()]
    
    # Create boxplot with consistent ordering
    box_data = [perf_df[perf_df['Centre'] == center][metric].values 
                for center in available_centers]
    
    box_plot = ax.boxplot(box_data,
                         labels=available_centers,
                         patch_artist=True,
                         notch=False,
                         showmeans=True,
                         meanprops=dict(marker='D', markerfacecolor='red', 
                                      markeredgecolor='darkred', markersize=8))
    
    # Apply consistent colors for each center
    for patch, center in zip(box_plot['boxes'], available_centers):
        patch.set_facecolor(CENTER_COLORS[center])
        patch.set_alpha(0.7)
    
    # Customize plot elements
    for whisker in box_plot['whiskers']:
        whisker.set(linewidth=1.2, linestyle='--', color='gray')
    for cap in box_plot['caps']:
        cap.set(linewidth=1.2, color='gray')
    for median in box_plot['medians']:
        median.set(linewidth=2, color='black')
    
    # Set appropriate y-axis limits and reference lines
    if metric == 'MCC':
        ax.set_ylim(-1.05, 1.05)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, 
                  label='No correlation')
        ax.legend(loc='lower left')
    else:
        ax.set_ylim(-0.05, 1.05)
        if metric == 'Balanced Acc':
            ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, 
                      label='Chance level')
            ax.legend(loc='lower left')
    
    # Styling
    ax.set_xlabel('Center', fontsize=14, fontweight='bold')
    ax.set_ylabel(metric, fontsize=14, fontweight='bold')
    ax.set_title(f'{metric} by Center - {model_name}\nTarget: {target}', 
                fontsize=16, fontweight='bold', pad=20)
    
    ax.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    if len(available_centers) > 5:
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    save_figure(fig, out_dir / f"perf_boxplot_{model_name.replace(' ', '_')}_{target}_{metric.replace(' ', '_')}")
    plt.close()


def get_model_path(out_dir: Path, model_name: str, target: str, fold: int) -> Path:
    """Generate path for saved model."""
    model_dir = out_dir / "saved_models" / target
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir / f"{model_name.replace(' ', '_')}_fold{fold}.pkl"


def save_model(model, path: Path):
    """Save model using pickle."""
    with open(path, 'wb') as f:
        pickle.dump(model, f)


def load_model(path: Path):
    """Load model using pickle."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def model_exists(path: Path) -> bool:
    """Check if model file exists."""
    return path.exists()


def _to_numpy(shap_vals):
    """Convert SHAP values to numpy array for positive class."""
    if hasattr(shap_vals, "values"):
        shap_vals = shap_vals.values
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    if shap_vals.ndim == 3:
        shap_vals = shap_vals[:, :, 1]
    return np.asarray(shap_vals)


def compute_shap(best_model: Pipeline, model_name: str, X_train, X_test,
                 *, nsamples_kernel: int = 100, background_size: int = 100):
    """Compute SHAP values."""
    preproc = Pipeline(best_model.steps[:-1])
    X_train_trans = preproc.transform(X_train)
    X_test_trans = preproc.transform(X_test)
    clf = best_model.named_steps["clf"]

    if model_name in ("Random Forest", "XGBoost"):
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_test_trans)
    elif model_name == "Logistic Regression":
        background = X_train_trans
        if hasattr(background, "values"):
            background = background.values
        explainer = shap.LinearExplainer(clf, background)
        shap_values = explainer.shap_values(X_test_trans)
    elif model_name == "MLP":
        background = X_train_trans
        if hasattr(background, "values"):
            background = background.values
        background_sample = background[: min(background_size, background.shape[0])]
        explainer = shap.KernelExplainer(clf.predict_proba, background_sample)
        shap_values = explainer.shap_values(X_test_trans, nsamples=nsamples_kernel, silent=True)
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    shap_values = _to_numpy(shap_values)
    return shap_values


def plot_shap_beeswarm(shap_values: np.ndarray, X_values: pd.DataFrame, 
                      model_name: str, target: str, out_dir: Path):
    """Create SHAP beeswarm plot."""
    fig, ax = plt.subplots(figsize=(10, 8), dpi=DPI)
    
    shap.summary_plot(shap_values, X_values, show=False, max_display=20,
                     plot_size=(10, 8), color_bar_label='Feature value')
    
    ax = plt.gca()
    ax.set_xlabel('SHAP value (impact on model output)', fontsize=14, fontweight='bold')
    ax.set_title(f'SHAP Feature Importance - {model_name}\nTarget: {target}', 
                fontsize=16, fontweight='bold', pad=20)
    
    ax.grid(True, axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    save_figure(fig, out_dir / f"shap_beeswarm_{model_name.replace(' ', '_')}_{target}")
    plt.close()


def crossval_pipeline(
    df: pd.DataFrame,
    targets: list[str],
    batch_df: pd.DataFrame | pd.Series,
    out_dir: Path,
    convert_si: bool = True,
    force_retrain: bool = False,
):
    """Main cross-validation pipeline with model persistence."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ensure a single Series with city labels aligned to df
    if isinstance(batch_df, pd.DataFrame):
        if "City" not in batch_df.columns:
            raise ValueError("`batch_df` must contain a 'City' column.")
        groups = batch_df["City"].astype("category")
    else:
        groups = batch_df.astype("category")
    if not groups.index.equals(df.index):
        raise ValueError("Index mismatch between `df` and `batch_df`. Align them first.")

    X_full = df.drop(columns=targets)

    for tgt in targets:
        print(f"\n{'='*60}")
        print(f"Processing target: {tgt}")
        print(f"{'='*60}")
        
        y_raw = df[tgt]
        if convert_si:
            y = y_raw.map({"S": 0, "I": 0, "R": 1})
        else:
            y = y_raw.copy()
        y = y.dropna()
        X = X_full.loc[y.index]
        g = groups.loc[y.index]

        if y.nunique() < 2 or y.value_counts().min() < 20:
            print(f"Skipping {tgt}: insufficient class balance.")
            continue

        # Composite label for dual-stratification
        strata = y.astype(str) + "__" + g.astype(str)

        models = {
            "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED),
            "MLP": MLPClassifier(random_state=SEED, early_stopping=True, learning_rate="adaptive"),
            "Random Forest": RandomForestClassifier(class_weight="balanced", random_state=SEED),
            "XGBoost": XGBClassifier(eval_metric="logloss", random_state=SEED),
        }

        metrics_all: dict[str, list[dict]] = {m: [] for m in models}
        confmat_sum: dict[str, np.ndarray] = {m: np.zeros((2, 2)) for m in models}

        # SHAP accumulation
        shap_vals_all: dict[str, list[np.ndarray]] = {m: [] for m in models}
        X_test_all: dict[str, list[pd.DataFrame]] = {m: [] for m in models}

        outer_cv = StratifiedKFold(n_splits=N_SPLITS_OUTER, shuffle=True, random_state=SEED)
        
        for fold_idx, (train_idx, test_idx) in enumerate(
            tqdm(outer_cv.split(X, strata), total=N_SPLITS_OUTER, desc=f"{tgt} – outer CV")
        ):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            strata_train = strata.iloc[train_idx]

            inner_cv = StratifiedKFold(n_splits=N_SPLITS_INNER, shuffle=True, random_state=SEED)

            for model_name, base_clf in models.items():
                model_path = get_model_path(out_dir, model_name, tgt, fold_idx)
                
                # Check if model exists and we're not forcing retrain
                if model_exists(model_path) and not force_retrain:
                    print(f"  Loading existing model: {model_name} (fold {fold_idx})")
                    best_est = load_model(model_path)
                else:
                    print(f"  Training model: {model_name} (fold {fold_idx})")
                    pipe = Pipeline([
                        ("combat", ComBat(batch=g)),
                        ("scaler", StandardScaler().set_output(transform="pandas")),
                        ("clf", base_clf),
                    ])
                    grid = param_grids.get(model_name, {})

                    gs = GridSearchCV(
                        pipe,
                        param_grid=grid,
                        scoring="roc_auc",
                        cv=inner_cv.split(X_train, strata_train),
                        n_jobs=-1,
                    )
                    gs.fit(X_train, y_train)
                    best_est = gs.best_estimator_
                    
                    # Save the model
                    save_model(best_est, model_path)

                # Make predictions
                y_pred = best_est.predict(X_test)
                y_prob = best_est.predict_proba(X_test)

                y_pred_ser = pd.Series(y_pred, index=y_test.index)
                y_prob_ser = pd.DataFrame(y_prob, index=y_test.index, columns=[0, 1])
                confmat_sum[model_name] += confusion_matrix(y_test, y_pred, normalize="true")
                metrics_all[model_name].append(metric_dict(y_test, y_pred, y_prob))

                # SHAP values
                shap_vals = compute_shap(best_est, model_name, X_train, X_test)
                shap_vals_all[model_name].append(shap_vals)
                X_test_all[model_name].append(X_test)

                # Per-center metrics
                per_center = []
                for center, idx_center in (g.iloc[test_idx]).groupby(g.iloc[test_idx], observed=True).groups.items():
                    y_c = y_test.loc[idx_center]
                    y_p = y_pred_ser.loc[idx_center].values
                    y_pb = y_prob_ser.loc[idx_center].values
                    m = metric_dict(y_c, y_p, y_pb)
                    m.update({"Centre": center, "Fold": fold_idx})
                    per_center.append(m)
                df_cent = pd.DataFrame(per_center)
                centre_file = out_dir / f"perf_by_center_{model_name.replace(' ', '_')}_{tgt}.csv"
                header = not centre_file.exists()
                df_cent.to_csv(centre_file, mode="a", index=False, header=header)

        # Aggregate results for all models
        for model_name, fold_metrics in metrics_all.items():
            df_metrics = pd.DataFrame(fold_metrics)
            mean = df_metrics.mean()
            std = df_metrics.std()

            # Calculate bootstrap 95% CI for each metric
            ci_lower = []
            ci_upper = []
            for col in df_metrics.columns:
                lower, upper = bootstrap_ci(df_metrics[col].values)
                ci_lower.append(lower)
                ci_upper.append(upper)

            summary_data = {
                'Metric': df_metrics.columns,
                'Mean': mean.values,
                'Std': std.values,
                'CI_Lower': ci_lower,
                'CI_Upper': ci_upper,
                'Mean [95% CI]': [f"{m:.3f} [{l:.3f}, {u:.3f}]"
                                  for m, l, u in zip(mean, ci_lower, ci_upper)]
            }
            summary_df = pd.DataFrame(summary_data)

            # Save detailed metrics
            summary_df.to_csv(out_dir / f"metrics_detailed_{model_name.replace(' ', '_')}_{tgt}.csv", index=False)

            # Confusion matrix plot
            cm_mean = confmat_sum[model_name] / N_SPLITS_OUTER
            plot_conf_matrix(cm_mean, model_name, tgt, out_dir)

            # SHAP beeswarm plot
            sv_all = np.vstack(shap_vals_all[model_name])
            X_all = pd.concat(X_test_all[model_name], axis=0)
            plot_shap_beeswarm(sv_all, X_all, model_name, tgt, out_dir)

            # Per-center performance boxplots
            perf_by_center = pd.read_csv(out_dir / f"perf_by_center_{model_name.replace(' ', '_')}_{tgt}.csv")
            
            # Create boxplots for each metric
            for metric in ["Balanced Acc", "MCC", "AUROC"]:
                plot_performance_boxplots(perf_by_center, metric, model_name, tgt, out_dir)

        # Save aggregate metrics for all models
        all_summaries = {}
        for model_name, fold_metrics in metrics_all.items():
            df_metrics = pd.DataFrame(fold_metrics)
            mean = df_metrics.mean()

            # Calculate bootstrap 95% CI for each metric
            ci_formatted = []
            for col in df_metrics.columns:
                lower, upper = bootstrap_ci(df_metrics[col].values)
                ci_formatted.append(f"{mean[col]:.3f} [{lower:.3f}, {upper:.3f}]")

            all_summaries[model_name] = ci_formatted

        metrics_df = pd.DataFrame(all_summaries, index=df_metrics.columns).T
        metrics_df.to_csv(out_dir / f"metrics_{tgt}.csv")
        
        print(f"\nCompleted processing for target '{tgt}'")
        print(f"Results saved to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cross-validated ComBat pipeline + SHAP with enhanced visualizations"
    )
    parser.add_argument("--data", required=True, help="CSV with features + targets")
    parser.add_argument("--batches", required=True, help="CSV with 'City' column")
    parser.add_argument("--targets", nargs="+", required=True, help="Target column names")
    parser.add_argument("--out", default="results/aggregated_datasets", help="Output directory")
    parser.add_argument("--force-retrain", action="store_true", 
                       help="Force retraining even if saved models exist")
    args = parser.parse_args()

    df_data = pd.read_csv(args.data, index_col=0)
    df_batches = pd.read_csv(args.batches, index_col=0)
    output_path = Path(args.out)

    crossval_pipeline(
        df_data, 
        args.targets, 
        df_batches, 
        output_path,
        force_retrain=args.force_retrain
    )