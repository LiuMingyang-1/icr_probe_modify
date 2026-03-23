"""Shared data loading and preprocessing for trajectory analysis."""

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np

# Default data path (relative to project root)
DEFAULT_DATA_PATH = Path(__file__).resolve().parents[3] / "outputs" / "icr_halu_eval_random_qwen2.5.jsonl"

# Layer configuration
NUM_RAW_LAYERS = 28
USABLE_LAYERS = 27  # layer 27 is all-zeros, drop it

# Stage definitions (each 9 layers)
EARLY_LAYERS = slice(0, 9)    # 0-8
MIDDLE_LAYERS = slice(9, 18)  # 9-17
LATE_LAYERS = slice(18, 27)   # 18-26


def load_data(path: str = None) -> Tuple[List[np.ndarray], np.ndarray, List[str]]:
    """Load ICR score data from JSONL file.

    Returns:
        trajectories: list of arrays, each [27, n_tokens_i] (layer 27 dropped)
        labels: array [n_samples], 0=correct, 1=hallucinated
        responses: list of response strings
    """
    data_path = Path(path) if path else DEFAULT_DATA_PATH
    trajectories = []
    labels = []
    responses = []

    with data_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            label = rec.get("label")
            if label is None:
                continue
            scores = np.array(rec["icr_scores"])[:USABLE_LAYERS]  # [27, n_tokens]
            trajectories.append(scores)
            labels.append(int(label))
            responses.append(rec.get("response", ""))

    return trajectories, np.array(labels), responses


def get_sample_trajectories(trajectories: List[np.ndarray]) -> np.ndarray:
    """Aggregate token-level ICR into sample-level trajectories.

    For each sample, compute mean ICR across tokens at each layer.

    Returns:
        array [n_samples, 27]
    """
    return np.array([t.mean(axis=1) for t in trajectories])


def get_token_trajectories(
    trajectories: List[np.ndarray], labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Flatten to token-level trajectories with associated sample labels.

    Returns:
        token_trajs: array [n_total_tokens, 27]
        token_labels: array [n_total_tokens] (sample-level label for each token)
    """
    all_trajs = []
    all_labels = []
    for traj, label in zip(trajectories, labels):
        n_tokens = traj.shape[1]
        all_trajs.append(traj.T)  # [n_tokens, 27]
        all_labels.extend([label] * n_tokens)

    return np.vstack(all_trajs), np.array(all_labels)
