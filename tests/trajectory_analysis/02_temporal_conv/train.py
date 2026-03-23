#!/usr/bin/env python3
"""Method 2: Temporal Convolution Over Layers.

Train CNN models that treat 27-layer ICR trajectories as 1D signals.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.data_loader import load_data, get_sample_trajectories
from shared.metrics import evaluate, print_results, RANDOM_SEED
from shared.visualization import plot_multi_roc

from models import BaselineMLP, TemporalCNN, MultiScaleCNN

FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_EPOCHS = 100
BATCH_SIZE = 64
LR = 1e-3
PATIENCE = 10
N_SPLITS = 5


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        pred = model(X_batch).squeeze()
        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_model(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    criterion = nn.BCELoss()
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        pred = model(X_batch).squeeze()
        total_loss += criterion(pred, y_batch).item() * len(y_batch)
        all_preds.append(pred.cpu().numpy())
        all_labels.append(y_batch.cpu().numpy())
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    loss = total_loss / len(loader.dataset)
    return loss, preds, labels


def train_model(model_class, X_train, y_train, X_val, y_val, model_name="Model"):
    """Train a model and return val predictions + training history."""
    train_ds = TensorDataset(
        torch.FloatTensor(X_train), torch.FloatTensor(y_train)
    )
    val_ds = TensorDataset(
        torch.FloatTensor(X_val), torch.FloatTensor(y_val)
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = model_class().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float("inf")
    best_preds = None
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(N_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_preds, val_labels = evaluate_model(model, val_loader)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_preds = val_preds.copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    return best_preds, val_labels, history


def main():
    print("=" * 60)
    print("  Method 2: Temporal Convolution Over Layers")
    print("=" * 60)

    # Load data
    trajs, labels, _ = load_data()
    X = get_sample_trajectories(trajs)
    y = labels.astype(np.float32)
    print(f"Samples: {len(y)} (hallu={int(y.sum())}, correct={int(len(y)-y.sum())})")
    print(f"Device: {DEVICE}")

    # Models to compare
    model_classes = {
        "BaselineMLP": BaselineMLP,
        "TemporalCNN": TemporalCNN,
        "MultiScaleCNN": MultiScaleCNN,
    }

    # Cross-validation
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    all_results = {}
    all_histories = {}
    all_val_preds = {}  # For ROC plotting

    for model_name, model_class in model_classes.items():
        print(f"\n--- {model_name} ({N_SPLITS}-fold CV) ---")
        fold_metrics = []
        fold_histories = []
        full_preds = np.zeros(len(y))
        full_labels = np.zeros(len(y))

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            torch.manual_seed(RANDOM_SEED + fold)
            preds, val_labels, history = train_model(
                model_class, X_train, y_train, X_val, y_val, model_name
            )
            metrics = evaluate(val_labels, preds)
            fold_metrics.append(metrics)
            fold_histories.append(history)

            full_preds[val_idx] = preds
            full_labels[val_idx] = val_labels

            print(f"  Fold {fold+1}: AUROC={metrics['AUROC']:.4f}, F1={metrics['F1']:.4f}")

        # Aggregate results
        result = {}
        for key in fold_metrics[0]:
            if key == "Threshold":
                continue
            vals = [m[key] for m in fold_metrics]
            result[f"{key}_mean"] = np.mean(vals)
            result[f"{key}_std"] = np.std(vals)

        all_results[model_name] = result
        all_histories[model_name] = fold_histories
        all_val_preds[model_name] = (full_labels, full_preds)
        print_results(result, model_name)

    # ============================================================
    # Visualizations
    # ============================================================
    print("--- Generating visualizations ---")

    # 1. Training curves (last fold for each model)
    fig, axes = plt.subplots(1, len(model_classes), figsize=(5 * len(model_classes), 4))
    for ax, (name, histories) in zip(axes, all_histories.items()):
        h = histories[-1]  # last fold
        ax.plot(h["train_loss"], label="Train", color="blue")
        ax.plot(h["val_loss"], label="Val", color="red")
        ax.set_title(name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle("Training Curves (Last Fold)", fontsize=14)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "training_curves.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / 'training_curves.png'}")

    # 2. ROC comparison
    roc_dict = {}
    for name, (lbl, pred) in all_val_preds.items():
        roc_dict[name] = pred
    plot_multi_roc(
        all_val_preds[list(all_val_preds.keys())[0]][0],
        roc_dict,
        FIGURES_DIR / "roc_comparison.png",
        title="ROC: Temporal Conv Models",
    )

    # 3. Bar chart comparison
    metrics_names = ["AUROC", "AUPRC", "F1"]
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metrics_names))
    width = 0.25
    colors = ["steelblue", "coral", "mediumseagreen"]

    for i, (name, result) in enumerate(all_results.items()):
        vals = [result[f"{m}_mean"] for m in metrics_names]
        errs = [result[f"{m}_std"] for m in metrics_names]
        ax.bar(x + i * width, vals, width, yerr=errs, label=name, color=colors[i], alpha=0.8, capsize=3)

    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics_names)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.legend()
    ax.set_ylim(0.4, 1.0)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "model_comparison.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / 'model_comparison.png'}")

    # 4. Visualize conv filters (TemporalCNN)
    model = TemporalCNN().to(DEVICE)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, conv) in zip(axes, [
        ("kernel=3", model.conv3), ("kernel=5", model.conv5), ("kernel=7", model.conv7)
    ]):
        weights = conv.weight.detach().cpu().numpy()[:, 0, :]  # [n_filters, kernel_size]
        ax.imshow(weights, aspect="auto", cmap="RdBu_r")
        ax.set_title(f"Conv1d {name} (init)")
        ax.set_xlabel("Kernel Position")
        ax.set_ylabel("Filter Index")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "conv_filters_init.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / 'conv_filters_init.png'}")

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for name, result in all_results.items():
        print(f"  {name:20s}: AUROC={result['AUROC_mean']:.4f} ± {result['AUROC_std']:.4f}, "
              f"F1={result['F1_mean']:.4f} ± {result['F1_std']:.4f}")
    print()

    return all_results


if __name__ == "__main__":
    main()
