"""Shared visualization utilities for trajectory analysis."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def set_style():
    """Set publication-quality plot style."""
    plt.rcParams.update({
        "figure.dpi": 150,
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "figure.figsize": (10, 6),
    })


def plot_trajectory_ribbon(hallu_trajs, correct_trajs, save_path, title="Layer-wise ICR Trajectory"):
    """Plot mean ± std ribbon for hallucinated vs correct trajectories.

    Args:
        hallu_trajs: array [n_hallu, 27]
        correct_trajs: array [n_correct, 27]
    """
    set_style()
    fig, ax = plt.subplots(figsize=(12, 6))
    layers = np.arange(hallu_trajs.shape[1])

    for trajs, label, color in [
        (hallu_trajs, "Hallucinated", "red"),
        (correct_trajs, "Correct", "blue"),
    ]:
        mean = trajs.mean(axis=0)
        std = trajs.std(axis=0)
        ax.plot(layers, mean, color=color, label=label, linewidth=2)
        ax.fill_between(layers, mean - std, mean + std, color=color, alpha=0.15)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean ICR Score")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_roc(y_true, y_prob, label, save_path, title="ROC Curve"):
    """Plot ROC curve."""
    set_style()
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auroc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(fpr, tpr, linewidth=2, label=f"{label} (AUROC={auroc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_multi_roc(y_true, predictions_dict, save_path, title="ROC Comparison"):
    """Plot multiple ROC curves on one figure.

    Args:
        predictions_dict: {name: y_prob_array}
    """
    set_style()
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(predictions_dict)))

    for (name, y_prob), color in zip(predictions_dict.items(), colors):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auroc = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=2, label=f"{name} ({auroc:.4f})", color=color)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_feature_violin(features_dict, labels, save_path, title="Feature Distribution by Label"):
    """Plot violin plots for features split by label.

    Args:
        features_dict: {feature_name: array [n_samples]}
        labels: array [n_samples]
    """
    set_style()
    n_features = len(features_dict)
    fig, axes = plt.subplots(1, n_features, figsize=(4 * n_features, 5))
    if n_features == 1:
        axes = [axes]

    hallu_mask = labels == 1
    correct_mask = labels == 0

    for ax, (name, values) in zip(axes, features_dict.items()):
        data = [values[correct_mask], values[hallu_mask]]
        parts = ax.violinplot(data, positions=[0, 1], showmeans=True, showmedians=True)
        parts["cmeans"].set_color("black")
        parts["cmedians"].set_color("gray")
        for i, color in enumerate(["blue", "red"]):
            parts["bodies"][i].set_facecolor(color)
            parts["bodies"][i].set_alpha(0.3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Correct", "Hallucinated"])
        ax.set_title(name)
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_bar_comparison(methods_results, save_path, title="Method Comparison"):
    """Plot grouped bar chart comparing methods across metrics.

    Args:
        methods_results: {method_name: {metric: value}}
    """
    set_style()
    metrics = ["AUROC", "AUPRC", "F1"]
    methods = list(methods_results.keys())
    n_methods = len(methods)
    n_metrics = len(metrics)

    x = np.arange(n_metrics)
    width = 0.8 / n_methods
    colors = plt.cm.tab10(np.linspace(0, 1, n_methods))

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (method, results) in enumerate(methods_results.items()):
        values = [results.get(f"{m}_mean", results.get(m, 0)) for m in metrics]
        errors = [results.get(f"{m}_std", 0) for m in metrics]
        offset = (i - n_methods / 2 + 0.5) * width
        ax.bar(x + offset, values, width, yerr=errors, label=method, color=colors[i], alpha=0.8, capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.set_ylim(0.4, 1.0)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")
