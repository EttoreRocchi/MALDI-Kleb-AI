"""
Cross-dataset learning framework across multiple sites

Usage:
------
python src/cross_datasets_framework.py \
    --data ./data/dfs/data_bin_3.csv \
    --batches ./data/dfs/metadata_bin_3.csv \
    --targets Meropenem Amikacin \
    --out_dir ./results/cross_datasets
"""

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from xgboost import XGBClassifier

SEED = 42
N_SPLITS = 10
INNER_SPLITS = 5

DPI = 300
FIGSIZE_STANDARD = (8, 6)
FIGSIZE_SQUARE = (6, 6)

plt.style.use("seaborn-v0_8-white")
plt.rcParams.update({
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 18,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.grid": False,
})

plt.switch_backend("Agg")
warnings.filterwarnings("ignore")


def compute_metrics(y_true, y_pred, y_prob):
    """Return evaluation metric values for a single fold."""
    return {
        "F1": f1_score(y_true, y_pred, average="weighted"),
        "AUROC": roc_auc_score(y_true, y_prob[:, 1]),
        "BalancedAcc": balanced_accuracy_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def save_heatmap(matrix: pd.DataFrame, title: str, out_path: Path):
    """Save heatmap."""
    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
    ax.grid(False)

    sns.heatmap(
        matrix,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        vmin=0.0 if 'Matthews' in title else 0.5,
        vmax=1.0,
        square=True,
        cbar_kws={"label": title.split(" - ")[0]},
        ax=ax
    )
    
    ax.set_title(title)
    ax.set_xlabel("Test city")
    ax.set_ylabel("Train city")
    
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def evaluate_single_target(
    df_data: pd.DataFrame,
    cities: pd.Series,
    target: str,
    other_targets: List[str],
    out_dir: Path,
):
    print(f"\n=== Processing target: {target} ===")
    target_out = out_dir / target
    target_out.mkdir(parents=True, exist_ok=True)

    # Build city-specific datasets
    city_dfs: List[pd.DataFrame] = []
    city_names: List[str] = []

    for city in cities.cat.categories:
        subset = df_data.loc[cities == city].copy()
        subset = subset[subset[target].notna()]
        subset.drop_duplicates(inplace=True)

        if subset.shape[0] < N_SPLITS:
            print(f"Skipping city '{city}' for {target}: not enough samples.")
            continue

        city_dfs.append(subset)
        city_names.append(city)

    if len(city_dfs) < 2:
        print(f"Not enough cities with data for {target}; skipping.")
        return

    # Prepare features and labels for each city
    X_list, y_list, outer_splits = [], [], []
    sskf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    for df in city_dfs:
        X = df.drop(columns=other_targets + [target])
        y = df[target].map({"S": 0, "I": 0, "R": 1})  # Map to binary
        y = y.dropna()
        X = X.loc[y.index]

        X_list.append(X)
        y_list.append(y)
        outer_splits.append(list(sskf.split(X, y)))

    # Define classifiers
    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=SEED
        ),
        "MLP": MLPClassifier(
            random_state=SEED, early_stopping=True, learning_rate="adaptive"
        ),
        "Random Forest": RandomForestClassifier(
            class_weight="balanced", random_state=SEED
        ),
        "XGBoost": XGBClassifier(eval_metric="logloss", random_state=SEED),
    }

    # Define hyperparameter grids for each model
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

    # Store results for each model and city pair
    results: Dict[str, Dict[Tuple[str, str], List[Dict[str, float]]]] = {
        m: {} for m in models
    }

    # Run outer cross-validation loop
    for fold_idx in tqdm(range(N_SPLITS), desc=f"Outer folds ({target})"):
        # Train on each city
        for i_city, (X_train_df, y_train_ser) in enumerate(zip(X_list, y_list)):
            train_idx, _ = outer_splits[i_city][fold_idx]
            X_train = X_train_df.iloc[train_idx]
            y_train = y_train_ser.iloc[train_idx]

            preproc = StandardScaler().set_output(transform="pandas")
            inner_cv = StratifiedKFold(
                n_splits=INNER_SPLITS, shuffle=True, random_state=SEED
            )

            # Hyperparameter tuning
            for model_name, base_clf in models.items():
                pipe = Pipeline([("prep", preproc), ("clf", base_clf)])
                gs = GridSearchCV(
                    pipe,
                    param_grid=param_grids[model_name],
                    cv=inner_cv,
                    scoring="roc_auc",
                    n_jobs=-1,
                )
                gs.fit(X_train, y_train)
                best_model = gs.best_estimator_

                # Evaluate on all cities
                for j_city, (X_test_df, y_test_ser) in enumerate(zip(X_list, y_list)):
                    _, test_idx = outer_splits[j_city][fold_idx]
                    X_test = X_test_df.iloc[test_idx]
                    y_test = y_test_ser.iloc[test_idx]

                    y_pred = best_model.predict(X_test)
                    y_prob = best_model.predict_proba(X_test)
                    metrics = compute_metrics(y_test, y_pred, y_prob)

                    key = (city_names[i_city], city_names[j_city])
                    results[model_name].setdefault(key, []).append(metrics)

    # Aggregate results and create visualizations
    for model_name, res in results.items():
        # Initialize metric matrices
        tmpl = pd.DataFrame(index=city_names, columns=city_names, dtype=float)
        f1_mat, auc_mat, bal_mat, mcc_mat = (
            tmpl.copy(),
            tmpl.copy(),
            tmpl.copy(),
            tmpl.copy(),
        )

        # Fill matrices with mean metrics across folds
        for (train_city, test_city), fold_metrics in res.items():
            dfm = pd.DataFrame(fold_metrics)
            f1_mat.loc[train_city, test_city] = dfm["F1"].mean()
            auc_mat.loc[train_city, test_city] = dfm["AUROC"].mean()
            bal_mat.loc[train_city, test_city] = dfm["BalancedAcc"].mean()
            mcc_mat.loc[train_city, test_city] = dfm["MCC"].mean()

        # Save heatmaps as PNG images
        save_heatmap(
            f1_mat, f"F1 weighted - {model_name}", target_out / f"{model_name.lower().replace(' ', '_')}_f1.png"
        )
        save_heatmap(
            auc_mat, f"AUROC - {model_name}", target_out / f"{model_name.lower().replace(' ', '_')}_auroc.png"
        )
        save_heatmap(
            bal_mat, f"Balanced accuracy - {model_name}", target_out / f"{model_name.lower().replace(' ', '_')}_balacc.png"
        )
        save_heatmap(
            mcc_mat, f"Matthews Corr. Coef. - {model_name}", target_out / f"{model_name.lower().replace(' ', '_')}_mcc.png"
        )

        # Save raw matrices as CSVs
        f1_mat.to_csv(target_out / f"{model_name.lower().replace(' ', '_')}_f1.csv")
        auc_mat.to_csv(target_out / f"{model_name.lower().replace(' ', '_')}_auroc.csv")
        bal_mat.to_csv(target_out / f"{model_name.lower().replace(' ', '_')}_balacc.csv")
        mcc_mat.to_csv(target_out / f"{model_name.lower().replace(' ', '_')}_mcc.csv")

    print(f"Finished target {target}. Files saved to '{target_out}'.")


def main(data_path: str, batches_path: str, targets: List[str], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data files
    df_data = pd.read_csv(data_path, index_col=0)
    df_batches = pd.read_csv(batches_path, index_col=0)

    # Validate input data
    if "City" not in df_batches.columns:
        raise ValueError("The batches file must contain a 'City' column.")
    if not df_batches.index.equals(df_data.index):
        raise ValueError("Indices of data and batches do not match.")

    cities = df_batches["City"].astype("category")

    # Check that all target columns exist
    missing = [t for t in targets if t not in df_data.columns]
    if missing:
        raise ValueError(f"Missing target columns in data: {missing}")

    # Process each target phenotype
    for tgt in targets:
        others = [o for o in targets if o != tgt]
        evaluate_single_target(df_data, cities, tgt, others, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cross-dataset evaluation across cities for multiple phenotypic targets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", required=True, help="CSV with features + phenotype columns")
    parser.add_argument("--batches", required=True, help="CSV with 'City' column")
    parser.add_argument("--targets", nargs="+", required=True, help="Target column names")
    parser.add_argument("--out_dir", default="results_cross_dataset", help="Output directory")

    args = parser.parse_args()

    main(args.data, args.batches, args.targets, Path(args.out_dir))