import sys
import unittest
from pathlib import Path

import numpy as np


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from spanlab.aggregation import aggregate_probabilities
from spanlab.evaluation import build_group_folds, evaluate_binary_predictions
from spanlab.representation import build_span_dataset_record
from spanlab.silver import assign_silver_label
from spanlab.spans import build_tokenizer_windows, map_char_span_to_token_span
from spanlab.visualization import build_prediction_index, build_token_level_scores, select_case_sample_id


class SpanLabTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sample_row = {
            "sample_id": "0:0",
            "source_sample_index": 0,
            "candidate_index": 0,
            "sample_label": 1,
            "response": "Paris was founded in 300.",
            "response_token_ids": [1, 2, 3, 4, 5],
            "response_offsets": [[0, 5], [6, 9], [10, 17], [18, 20], [21, 24]],
            "alignment_ok": True,
            "knowledge": "Paris is the capital of France and was founded long before 300.",
            "question": "When was Paris founded?",
            "icr_scores": [
                [0.1, 0.2, 0.3, 0.4, 0.5],
                [0.5, 0.4, 0.3, 0.2, 0.1],
            ]
            + [[0.2, 0.2, 0.2, 0.2, 0.2] for _ in range(25)],
        }

    def test_map_char_span_to_token_span(self) -> None:
        token_span = map_char_span_to_token_span(self.sample_row["response_offsets"], 6, 17)
        self.assertEqual(token_span, (1, 3))

    def test_build_tokenizer_windows(self) -> None:
        windows = build_tokenizer_windows(self.sample_row, window_sizes=[1, 2])
        self.assertEqual(len(windows), 9)
        self.assertEqual(windows[0]["span_text"], "Paris")
        self.assertEqual(windows[6]["span_text"], "was founded")

    def test_assign_silver_label_supported_number(self) -> None:
        span_row = {
            "span_id": "x",
            "sample_id": "0:0",
            "sample_label": 1,
            "route": "tokenizer_window",
            "span_type": "window",
            "token_start": 3,
            "token_end": 5,
            "char_start": 18,
            "char_end": 24,
            "token_char_start": 18,
            "token_char_end": 24,
            "span_len_tokens": 2,
            "span_text": "in 300",
        }
        labeled = assign_silver_label(span_row, self.sample_row)
        self.assertEqual(labeled["silver_label"], 0)
        self.assertGreaterEqual(labeled["silver_confidence"], 0.8)

    def test_build_span_dataset_record(self) -> None:
        span_row = {
            "span_id": "x",
            "sample_id": "0:0",
            "sample_label": 1,
            "route": "tokenizer_window",
            "span_type": "window",
            "token_start": 1,
            "token_end": 3,
            "char_start": 6,
            "char_end": 17,
            "token_char_start": 6,
            "token_char_end": 17,
            "span_len_tokens": 2,
            "span_text": "was founded",
            "silver_label": 1,
        }
        dataset_row = build_span_dataset_record(self.sample_row, span_row, pooling="mean")
        self.assertAlmostEqual(dataset_row["span_vector"][0], 0.25, places=6)
        self.assertAlmostEqual(dataset_row["span_vector"][1], 0.35, places=6)
        self.assertEqual(len(dataset_row["span_vector"]), 27)

    def test_metrics_and_folds(self) -> None:
        metrics = evaluate_binary_predictions([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
        self.assertAlmostEqual(metrics["AUROC"], 1.0, places=6)
        self.assertAlmostEqual(aggregate_probabilities([0.2, 0.8, 0.6], "topk_mean", top_k=2), 0.7, places=6)

        folds = build_group_folds(
            sample_ids=["a", "a", "b", "b", "c", "d"],
            sample_labels=[0, 0, 1, 1, 0, 1],
            n_splits=2,
            seed=42,
        )
        self.assertEqual(len(folds), 2)
        for train_samples, val_samples in folds:
            self.assertTrue(train_samples.isdisjoint(val_samples))
            self.assertEqual(train_samples | val_samples, {"a", "b", "c", "d"})

    def test_visualization_helpers(self) -> None:
        dataset_rows = [
            {
                "span_id": "a",
                "sample_id": "s1",
                "sample_label": 1,
                "token_start": 0,
                "token_end": 2,
                "silver_label": 1,
            },
            {
                "span_id": "b",
                "sample_id": "s1",
                "sample_label": 1,
                "token_start": 2,
                "token_end": 3,
                "silver_label": 0,
            },
            {
                "span_id": "c",
                "sample_id": "s2",
                "sample_label": 0,
                "token_start": 0,
                "token_end": 1,
                "silver_label": 0,
            },
        ]
        prediction_rows = [
            {"span_id": "a", "sample_id": "s1", "sample_label": 1, "probability": 0.9},
            {"span_id": "b", "sample_id": "s1", "sample_label": 1, "probability": 0.3},
            {"span_id": "c", "sample_id": "s2", "sample_label": 0, "probability": 0.8},
        ]
        pred_index = build_prediction_index(prediction_rows)
        token_scores, token_silver = build_token_level_scores(dataset_rows, pred_index, n_tokens=3)
        np.testing.assert_allclose(token_scores, np.array([0.9, 0.9, 0.3], dtype=np.float32))
        np.testing.assert_array_equal(token_silver, np.array([1, 1, -1], dtype=np.int32))
        self.assertEqual(select_case_sample_id(prediction_rows, selection="highest_hallucinated"), "s1")
        self.assertEqual(select_case_sample_id(prediction_rows, selection="highest_false_positive"), "s2")


if __name__ == "__main__":
    unittest.main()
