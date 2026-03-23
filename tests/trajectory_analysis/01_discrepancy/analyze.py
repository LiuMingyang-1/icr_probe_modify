#!/usr/bin/env python3
"""Method 1: Early/Middle/Late Discrepancy Modeling.

Split 27 layers into 3 stages, extract stage-level features,
and test if stage discrepancies can distinguish hallucinated vs correct samples.
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.data_loader import load_data, get_sample_trajectories, EARLY_LAYERS, MIDDLE_LAYERS, LATE_LAYERS
from shared.metrics import evaluate, cross_validate, print_results
from shared.visualization import plot_trajectory_ribbon, plot_roc, plot_multi_roc, plot_feature_violin

FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def extract_features(sample_trajs: np.ndarray) -> tuple:
    """Extract 7 discrepancy features from sample trajectories.

    Args:
        sample_trajs: [n_samples, 27]

    Returns:
        features: [n_samples, 7]
        feature_names: list of 7 names
    """
    mean_early = sample_trajs[:, EARLY_LAYERS].mean(axis=1)
    mean_mid = sample_trajs[:, MIDDLE_LAYERS].mean(axis=1)
    mean_late = sample_trajs[:, LATE_LAYERS].mean(axis=1)

    diff_mid_early = mean_mid - mean_early
    diff_late_mid = mean_late - mean_mid
    diff_late_early = mean_late - mean_early

    # Linear regression slope across all 27 layers
    layers = np.arange(27)
    layer_mean = layers.mean()
    layer_var = ((layers - layer_mean) ** 2).sum()
    slopes = ((sample_trajs * (layers[None, :] - layer_mean)).sum(axis=1)) / layer_var

    features = np.column_stack([
        mean_early, mean_mid, mean_late,
        diff_mid_early, diff_late_mid, diff_late_early,
        slopes,
    ])

    feature_names = [
        "mean_early", "mean_mid", "mean_late",
        "diff_mid_early", "diff_late_mid", "diff_late_early",
        "slope",
    ]

    return features, feature_names


def main():
    print("=" * 60)
    print("  Method 1: Discrepancy Modeling (Early/Middle/Late)")
    print("=" * 60)

    # Load data
    trajs, labels, _ = load_data()
    sample_trajs = get_sample_trajectories(trajs)
    print(f"Samples: {len(labels)} (hallu={labels.sum()}, correct={len(labels)-labels.sum()})")

    # Extract features
    features, feature_names = extract_features(sample_trajs)

    # ============================================================
    # 1. Per-feature AUROC (univariate)
    # ============================================================
    print("\n--- Per-feature AUROC ---")
    feature_aurocs = {}
    for i, name in enumerate(feature_names):
        auroc = roc_auc_score(labels, features[:, i])
        # If AUROC < 0.5, flip direction (lower value = hallucinated)
        if auroc < 0.5:
            auroc = 1 - auroc
        feature_aurocs[name] = auroc
        print(f"  {name:20s}: {auroc:.4f}")

    best_feat = max(feature_aurocs, key=feature_aurocs.get)
    print(f"\n  Best single feature: {best_feat} (AUROC={feature_aurocs[best_feat]:.4f})")

    # ============================================================
    # 2. Multi-feature classification (5-fold CV)
    # ============================================================
    print("\n--- Logistic Regression (5-fold CV) ---")
    lr_results = cross_validate(features, labels, LogisticRegression, max_iter=1000)
    print_results(lr_results, "LogisticRegression")

    print("--- Random Forest (5-fold CV) ---")
    rf_results = cross_validate(
        features, labels, RandomForestClassifier,
        n_estimators=100, random_state=42,
    )
    print_results(rf_results, "RandomForest")

    # ============================================================
    # 3. Baseline: raw 27-layer vector with LR
    # ============================================================
    print("--- Baseline: Raw 27-layer vector + LR ---")
    baseline_results = cross_validate(sample_trajs, labels, LogisticRegression, max_iter=1000)
    print_results(baseline_results, "Baseline (Raw 27-dim LR)")

    # ============================================================
    # Visualizations
    # ============================================================
    print("\n--- Generating visualizations ---")

    hallu_mask = labels == 1
    correct_mask = labels == 0

    # 1. Trajectory ribbon plot
    plot_trajectory_ribbon(
        sample_trajs[hallu_mask], sample_trajs[correct_mask],
        FIGURES_DIR / "trajectory_comparison.png",
        title="ICR Trajectory: Hallucinated vs Correct",
    )

    # 2. Feature violin plots
    feat_dict = {name: features[:, i] for i, name in enumerate(feature_names)}
    plot_feature_violin(
        feat_dict, labels,
        FIGURES_DIR / "feature_violins.png",
        title="Discrepancy Features by Label",
    )

    # 3. ROC curves: best feature + LR + RF
    # Train on full data for ROC visualization
    lr = LogisticRegression(max_iter=1000).fit(features, labels)
    lr_probs = lr.predict_proba(features)[:, 1]
    rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(features, labels)
    rf_probs = rf.predict_proba(features)[:, 1]

    best_feat_idx = feature_names.index(best_feat)
    best_feat_vals = features[:, best_feat_idx]
    if roc_auc_score(labels, best_feat_vals) < 0.5:
        best_feat_vals = -best_feat_vals

    plot_multi_roc(
        labels,
        {
            f"Best Feature ({best_feat})": best_feat_vals,
            "LR (7 features)": lr_probs,
            "RF (7 features)": rf_probs,
        },
        FIGURES_DIR / "roc_comparison.png",
        title="ROC: Discrepancy Features",
    )

    # 4. Scatter: diff_mid_early vs diff_late_mid
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        features[correct_mask, 3], features[correct_mask, 4],
        alpha=0.2, s=10, c="blue", label="Correct",
    )
    ax.scatter(
        features[hallu_mask, 3], features[hallu_mask, 4],
        alpha=0.2, s=10, c="red", label="Hallucinated",
    )
    ax.set_xlabel("diff_mid_early")
    ax.set_ylabel("diff_late_mid")
    ax.set_title("Stage Discrepancy Scatter")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "scatter_discrepancy.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / 'scatter_discrepancy.png'}")

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Best single feature:    {best_feat} (AUROC={feature_aurocs[best_feat]:.4f})")
    print(f"  LR (7 features):        AUROC={lr_results['AUROC_mean']:.4f} ± {lr_results['AUROC_std']:.4f}")
    print(f"  RF (7 features):        AUROC={rf_results['AUROC_mean']:.4f} ± {rf_results['AUROC_std']:.4f}")
    print(f"  Baseline (27-dim LR):   AUROC={baseline_results['AUROC_mean']:.4f} ± {baseline_results['AUROC_std']:.4f}")
    print()

    return {
        "best_single_feature": {best_feat: feature_aurocs[best_feat]},
        "LR_7feat": lr_results,
        "RF_7feat": rf_results,
        "Baseline_27dim_LR": baseline_results,
    }


if __name__ == "__main__":
    main()
