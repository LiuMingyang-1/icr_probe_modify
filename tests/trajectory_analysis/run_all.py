#!/usr/bin/env python3
"""Run all 4 trajectory analysis methods and generate comparison.

Can also be run with --summary-only to just generate the comparison
from previously collected results (hardcoded below).
"""

import sys
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

FIGURES_DIR = Path(__file__).parent
RESULTS_FILE = Path(__file__).parent / "results_summary.txt"


def get_precomputed_results():
    """Return results from previous runs for quick summary generation."""
    return {
        # Method 1: Discrepancy Modeling
        "Discrepancy_LR_7feat": {
            "AUROC_mean": 0.9213, "AUROC_std": 0.0074,
            "AUPRC_mean": 0.9148, "AUPRC_std": 0.0098,
            "F1_mean": 0.8547, "F1_std": 0.0103,
        },
        "Discrepancy_RF_7feat": {
            "AUROC_mean": 0.9237, "AUROC_std": 0.0070,
            "AUPRC_mean": 0.9147, "AUPRC_std": 0.0080,
            "F1_mean": 0.8612, "F1_std": 0.0070,
        },
        "Baseline_27dim_LR": {
            "AUROC_mean": 0.9697, "AUROC_std": 0.0035,
            "AUPRC_mean": 0.9697, "AUPRC_std": 0.0046,
            "F1_mean": 0.9230, "F1_std": 0.0052,
        },
        # Method 3: Change Point Detection
        "ChangePoint_LR": {
            "AUROC_mean": 0.8114, "AUROC_std": 0.0131,
            "AUPRC_mean": 0.7497, "AUPRC_std": 0.0174,
            "F1_mean": 0.7765, "F1_std": 0.0060,
        },
        "ChangePoint_RF": {
            "AUROC_mean": 0.8984, "AUROC_std": 0.0072,
            "AUPRC_mean": 0.8843, "AUPRC_std": 0.0100,
            "F1_mean": 0.8369, "F1_std": 0.0071,
        },
        # Method 2: Temporal Conv
        "TempConv_BaselineMLP": {
            "AUROC_mean": 0.9862, "AUROC_std": 0.0033,
            "AUPRC_mean": 0.9875, "AUPRC_std": 0.0037,
            "F1_mean": 0.9545, "F1_std": 0.0041,
        },
        "TempConv_TemporalCNN": {
            "AUROC_mean": 0.9717, "AUROC_std": 0.0031,
            "AUPRC_mean": 0.9729, "AUPRC_std": 0.0037,
            "F1_mean": 0.9302, "F1_std": 0.0056,
        },
        "TempConv_MultiScaleCNN": {
            "AUROC_mean": 0.9836, "AUROC_std": 0.0033,
            "AUPRC_mean": 0.9848, "AUPRC_std": 0.0040,
            "F1_mean": 0.9486, "F1_std": 0.0021,
        },
        # Method 4: Trajectory Encoder
        "TrajEnc_GRU": {
            "AUROC_mean": 0.9696, "AUROC_std": 0.0043,
            "AUPRC_mean": 0.9692, "AUPRC_std": 0.0064,
            "F1_mean": 0.9245, "F1_std": 0.0054,
        },
        "TrajEnc_Transformer": {
            "AUROC_mean": 0.9764, "AUROC_std": 0.0056,
            "AUPRC_mean": 0.9769, "AUPRC_std": 0.0061,
            "F1_mean": 0.9316, "F1_std": 0.0085,
        },
        "TrajEnc_Deep1DCNN": {
            "AUROC_mean": 0.9810, "AUROC_std": 0.0040,
            "AUPRC_mean": 0.9827, "AUPRC_std": 0.0047,
            "F1_mean": 0.9440, "F1_std": 0.0038,
        },
    }


def run_all():
    """Run all 4 methods from scratch."""
    all_results = {}

    print("\n" + "#" * 70)
    print("# METHOD 1: DISCREPANCY MODELING")
    print("#" * 70)
    m1 = __import__("01_discrepancy.analyze", fromlist=["main"])
    results_1 = m1.main()
    all_results["Discrepancy_LR_7feat"] = results_1["LR_7feat"]
    all_results["Discrepancy_RF_7feat"] = results_1["RF_7feat"]
    all_results["Baseline_27dim_LR"] = results_1["Baseline_27dim_LR"]

    print("\n" + "#" * 70)
    print("# METHOD 3: CHANGE POINT DETECTION")
    print("#" * 70)
    m3 = __import__("03_change_point.analyze", fromlist=["main"])
    results_3 = m3.main()
    all_results["ChangePoint_LR"] = results_3["LR_CP"]
    all_results["ChangePoint_RF"] = results_3["RF_CP"]

    print("\n" + "#" * 70)
    print("# METHOD 2: TEMPORAL CONV")
    print("#" * 70)
    m2 = __import__("02_temporal_conv.train", fromlist=["main"])
    results_2 = m2.main()
    for name, res in results_2.items():
        all_results[f"TempConv_{name}"] = res

    print("\n" + "#" * 70)
    print("# METHOD 4: TRAJECTORY ENCODER")
    print("#" * 70)
    m4 = __import__("04_trajectory_encoder.train", fromlist=["main"])
    results_4 = m4.main()
    for name, res in results_4.items():
        all_results[f"TrajEnc_{name}"] = res

    return all_results


def print_and_save(all_results):
    """Print comparison table and save to file."""
    print("\n" + "=" * 70)
    print("  FINAL COMPARISON (All Methods, 5-fold CV)")
    print("=" * 70)

    header = f"{'Method':30s} {'AUROC':>14s} {'AUPRC':>14s} {'F1':>14s}"
    print(header)
    print("-" * len(header))

    lines = [header, "-" * len(header)]
    for name, result in sorted(all_results.items(), key=lambda x: x[1]["AUROC_mean"], reverse=True):
        auroc = f"{result['AUROC_mean']:.4f}±{result['AUROC_std']:.4f}"
        auprc = f"{result['AUPRC_mean']:.4f}±{result['AUPRC_std']:.4f}"
        f1 = f"{result['F1_mean']:.4f}±{result['F1_std']:.4f}"
        line = f"{name:30s} {auroc:>14s} {auprc:>14s} {f1:>14s}"
        print(line)
        lines.append(line)

    with open(RESULTS_FILE, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nResults saved to: {RESULTS_FILE}")


def generate_comparison_chart(all_results):
    """Generate grouped bar chart."""
    from shared.visualization import plot_bar_comparison
    plot_bar_comparison(all_results, FIGURES_DIR / "final_comparison.png", title="All Methods Comparison")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-only", action="store_true",
                        help="Use precomputed results instead of re-running")
    args = parser.parse_args()

    print("=" * 70)
    print("  Trajectory Analysis - All Methods Comparison")
    print("=" * 70)

    if args.summary_only:
        all_results = get_precomputed_results()
    else:
        all_results = run_all()

    print_and_save(all_results)
    generate_comparison_chart(all_results)
    print("\nDone!")


if __name__ == "__main__":
    main()
