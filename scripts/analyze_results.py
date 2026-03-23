#!/usr/bin/env python3
"""Analyze ICR score results: AUROC, layer-wise stats, and visualization."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve


def load_results(path: Path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def main():
    parser = argparse.ArgumentParser(description="Analyze ICR score results.")
    parser.add_argument("--input", type=str, required=True, help="Path to output JSONL")
    parser.add_argument("--output_dir", type=str, default="./analysis", help="Where to save figures")
    args = parser.parse_args()

    records = load_results(Path(args.input))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = []
    layer_means = []  # per-sample mean ICR score for each layer
    sample_means = []  # per-sample overall mean ICR score

    for rec in records:
        label = rec.get("label")
        if label is None:
            continue
        labels.append(int(label))
        scores = np.array(rec["icr_scores"])  # shape: [n_layers, n_tokens]
        layer_means.append(scores.mean(axis=1))  # [n_layers]
        sample_means.append(scores.mean())

    labels = np.array(labels)
    layer_means = np.array(layer_means)  # [n_samples, n_layers]
    sample_means = np.array(sample_means)
    n_layers = layer_means.shape[1]

    print(f"Samples: {len(labels)} (hallucinated={labels.sum()}, correct={len(labels)-labels.sum()})")
    print(f"Layers: {n_layers}")
    print()

    # ============================================================
    # 1. Overall AUROC (mean ICR score as predictor)
    # ============================================================
    auroc_overall = roc_auc_score(labels, sample_means)
    print(f"Overall AUROC (mean ICR): {auroc_overall:.4f}")
    print()

    # ============================================================
    # 2. Per-layer AUROC
    # ============================================================
    layer_aurocs = []
    print("Per-layer AUROC:")
    for l in range(n_layers):
        try:
            auc = roc_auc_score(labels, layer_means[:, l])
        except ValueError:
            auc = 0.5
        layer_aurocs.append(auc)
    layer_aurocs = np.array(layer_aurocs)

    best_layer = np.argmax(layer_aurocs)
    for l in range(n_layers):
        marker = " <-- best" if l == best_layer else ""
        print(f"  Layer {l:2d}: {layer_aurocs[l]:.4f}{marker}")

    print(f"\nBest layer: {best_layer} (AUROC={layer_aurocs[best_layer]:.4f})")

    # ============================================================
    # 3. Mean ICR score by label
    # ============================================================
    hallucinated_mask = labels == 1
    correct_mask = labels == 0

    print(f"\nMean ICR score (hallucinated): {sample_means[hallucinated_mask].mean():.6f}")
    print(f"Mean ICR score (correct):      {sample_means[correct_mask].mean():.6f}")

    # ============================================================
    # Figure 1: Per-layer AUROC
    # ============================================================
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(range(n_layers), layer_aurocs, marker="o", markersize=4, linewidth=2)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Layer", fontsize=14)
    ax.set_ylabel("AUROC", fontsize=14)
    ax.set_title("Per-layer AUROC for Hallucination Detection", fontsize=16)
    ax.set_xticks(range(n_layers))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "layer_auroc.png", dpi=150)
    print(f"\nSaved: {out_dir / 'layer_auroc.png'}")

    # ============================================================
    # Figure 2: Layer-wise mean ICR score (hallucinated vs correct)
    # ============================================================
    fig, ax = plt.subplots(figsize=(12, 6))
    hallu_layer_mean = layer_means[hallucinated_mask].mean(axis=0)
    correct_layer_mean = layer_means[correct_mask].mean(axis=0)
    ax.plot(range(n_layers), hallu_layer_mean, marker="o", markersize=4, label="Hallucinated", color="red")
    ax.plot(range(n_layers), correct_layer_mean, marker="s", markersize=4, label="Correct", color="blue")
    ax.set_xlabel("Layer", fontsize=14)
    ax.set_ylabel("Mean ICR Score", fontsize=14)
    ax.set_title("Layer-wise ICR Score: Hallucinated vs Correct", fontsize=16)
    ax.set_xticks(range(n_layers))
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "layer_icr_comparison.png", dpi=150)
    print(f"Saved: {out_dir / 'layer_icr_comparison.png'}")

    # ============================================================
    # Figure 3: ROC curve (best layer)
    # ============================================================
    fpr, tpr, _ = roc_curve(labels, layer_means[:, best_layer])
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(fpr, tpr, linewidth=2, label=f"Layer {best_layer} (AUROC={layer_aurocs[best_layer]:.4f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=14)
    ax.set_ylabel("True Positive Rate", fontsize=14)
    ax.set_title("ROC Curve (Best Layer)", fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "roc_best_layer.png", dpi=150)
    print(f"Saved: {out_dir / 'roc_best_layer.png'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
