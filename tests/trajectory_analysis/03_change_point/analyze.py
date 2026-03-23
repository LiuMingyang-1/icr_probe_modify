#!/usr/bin/env python3
"""Method 3: Cross-Layer Change Point Detection.

Detect abrupt transitions in token ICR trajectories across layers.
Analyze whether hallucinated samples have different change point patterns.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.data_loader import load_data, get_sample_trajectories
from shared.metrics import evaluate, cross_validate, print_results
from shared.visualization import plot_multi_roc, plot_feature_violin

FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def detect_change_points(traj: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Detect change points in a 1D trajectory using threshold on first-order diff.

    Args:
        traj: [27] ICR trajectory
        sigma: threshold = mean(|d1|) + sigma * std(|d1|)

    Returns:
        boolean array [26] indicating change points
    """
    d1 = np.diff(traj)
    abs_d1 = np.abs(d1)
    threshold = abs_d1.mean() + sigma * abs_d1.std()
    return abs_d1 > threshold


def extract_features(sample_trajs: np.ndarray) -> tuple:
    """Extract 8 change-point features per sample.

    Args:
        sample_trajs: [n_samples, 27]

    Returns:
        features: [n_samples, 8]
        feature_names: list of names
    """
    n_samples = sample_trajs.shape[0]
    features = np.zeros((n_samples, 8))

    for i, traj in enumerate(sample_trajs):
        d1 = np.diff(traj)           # [26]
        d2 = np.diff(d1)             # [25]
        abs_d1 = np.abs(d1)
        abs_d2 = np.abs(d2)

        # Change points (1st and 2nd order)
        cp1 = detect_change_points(traj, sigma=2.0)
        cp2_threshold = abs_d2.mean() + 2.0 * abs_d2.std()
        cp2 = abs_d2 > cp2_threshold

        features[i, 0] = cp1.sum()                           # num change points (1st order)
        features[i, 1] = cp2.sum()                           # num change points (2nd order)
        features[i, 2] = np.argmax(cp1) if cp1.any() else -1 # first change point location
        features[i, 3] = np.argmax(abs_d1)                   # location of largest change
        features[i, 4] = abs_d1.max()                        # magnitude of largest change
        features[i, 5] = abs_d1.mean()                       # mean change magnitude
        features[i, 6] = d1.var()                            # trajectory roughness
        features[i, 7] = abs_d2.max()                        # max 2nd order sharpness

    feature_names = [
        "n_cp_d1", "n_cp_d2", "first_cp_loc", "max_change_loc",
        "max_change_mag", "mean_change_mag", "roughness", "max_sharpness",
    ]

    return features, feature_names


def main():
    print("=" * 60)
    print("  Method 3: Change Point Detection")
    print("=" * 60)

    # Load data
    trajs, labels, _ = load_data()
    sample_trajs = get_sample_trajectories(trajs)
    print(f"Samples: {len(labels)} (hallu={labels.sum()}, correct={len(labels)-labels.sum()})")

    # Extract features
    features, feature_names = extract_features(sample_trajs)

    hallu_mask = labels == 1
    correct_mask = labels == 0

    # ============================================================
    # 1. Per-feature AUROC
    # ============================================================
    print("\n--- Per-feature AUROC ---")
    feature_aurocs = {}
    for i, name in enumerate(feature_names):
        try:
            auroc = roc_auc_score(labels, features[:, i])
            if auroc < 0.5:
                auroc = 1 - auroc
        except ValueError:
            auroc = 0.5
        feature_aurocs[name] = auroc
        print(f"  {name:20s}: {auroc:.4f}")

    best_feat = max(feature_aurocs, key=feature_aurocs.get)
    print(f"\n  Best single feature: {best_feat} (AUROC={feature_aurocs[best_feat]:.4f})")

    # ============================================================
    # 2. Multi-feature classification
    # ============================================================
    print("\n--- Logistic Regression (5-fold CV) ---")
    lr_results = cross_validate(features, labels, LogisticRegression, max_iter=1000)
    print_results(lr_results, "LogisticRegression (CP features)")

    print("--- Random Forest (5-fold CV) ---")
    rf_results = cross_validate(
        features, labels, RandomForestClassifier,
        n_estimators=100, random_state=42,
    )
    print_results(rf_results, "RandomForest (CP features)")

    # ============================================================
    # Visualizations
    # ============================================================
    print("\n--- Generating visualizations ---")

    # 1. Example trajectories with change points
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    np.random.seed(42)
    for row, (mask, title_prefix, color) in enumerate([
        (hallu_mask, "Hallucinated", "red"),
        (correct_mask, "Correct", "blue"),
    ]):
        indices = np.where(mask)[0]
        chosen = np.random.choice(indices, 3, replace=False)
        for col, idx in enumerate(chosen):
            ax = axes[row, col]
            traj = sample_trajs[idx]
            cp = detect_change_points(traj)
            layers = np.arange(27)
            ax.plot(layers, traj, color=color, linewidth=2)
            cp_layers = np.where(cp)[0]
            for cl in cp_layers:
                ax.axvline(x=cl + 0.5, color="orange", linestyle="--", alpha=0.7)
            ax.set_title(f"{title_prefix} #{idx}")
            ax.set_xlabel("Layer")
            ax.set_ylabel("ICR Score")
            ax.grid(True, alpha=0.3)

    fig.suptitle("Example Trajectories with Change Points (orange dashed)", fontsize=14)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "example_trajectories.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / 'example_trajectories.png'}")

    # 2. Change point frequency heatmap by layer position
    cp_freq_hallu = np.zeros(26)
    cp_freq_correct = np.zeros(26)
    for i in range(len(sample_trajs)):
        cp = detect_change_points(sample_trajs[i])
        if labels[i] == 1:
            cp_freq_hallu += cp
        else:
            cp_freq_correct += cp

    cp_freq_hallu /= hallu_mask.sum()
    cp_freq_correct /= correct_mask.sum()

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(26)
    width = 0.35
    ax.bar(x - width/2, cp_freq_hallu, width, label="Hallucinated", color="red", alpha=0.7)
    ax.bar(x + width/2, cp_freq_correct, width, label="Correct", color="blue", alpha=0.7)
    ax.set_xlabel("Layer Transition")
    ax.set_ylabel("Change Point Frequency")
    ax.set_title("Change Point Frequency by Layer Position")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "cp_frequency_by_layer.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / 'cp_frequency_by_layer.png'}")

    # 3. First-order difference distribution
    d1_hallu = np.diff(sample_trajs[hallu_mask], axis=1).flatten()
    d1_correct = np.diff(sample_trajs[correct_mask], axis=1).flatten()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(d1_correct, bins=100, alpha=0.5, label="Correct", color="blue", density=True)
    ax.hist(d1_hallu, bins=100, alpha=0.5, label="Hallucinated", color="red", density=True)
    ax.set_xlabel("First-order Difference")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Layer-to-Layer ICR Changes")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "d1_distribution.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / 'd1_distribution.png'}")

    # 4. Feature violin plots
    feat_dict = {name: features[:, i] for i, name in enumerate(feature_names)}
    plot_feature_violin(
        feat_dict, labels,
        FIGURES_DIR / "feature_violins.png",
        title="Change Point Features by Label",
    )

    # 5. ROC curves
    lr = LogisticRegression(max_iter=1000).fit(features, labels)
    lr_probs = lr.predict_proba(features)[:, 1]
    rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(features, labels)
    rf_probs = rf.predict_proba(features)[:, 1]

    plot_multi_roc(
        labels,
        {"LR (CP features)": lr_probs, "RF (CP features)": rf_probs},
        FIGURES_DIR / "roc_comparison.png",
        title="ROC: Change Point Features",
    )

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Best single feature:    {best_feat} (AUROC={feature_aurocs[best_feat]:.4f})")
    print(f"  LR (CP features):      AUROC={lr_results['AUROC_mean']:.4f} ± {lr_results['AUROC_std']:.4f}")
    print(f"  RF (CP features):      AUROC={rf_results['AUROC_mean']:.4f} ± {rf_results['AUROC_std']:.4f}")
    print()

    return {
        "best_feature": {best_feat: feature_aurocs[best_feat]},
        "LR_CP": lr_results,
        "RF_CP": rf_results,
    }


if __name__ == "__main__":
    main()
