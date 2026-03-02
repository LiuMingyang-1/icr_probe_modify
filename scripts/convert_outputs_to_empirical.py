#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert outputs JSONL to empirical_study format (icr_score.pt + output_judge.jsonl)."
    )
    parser.add_argument("--input_jsonl", type=str, required=True, help="Path to outputs/*.jsonl")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save converted files")
    parser.add_argument(
        "--label_semantic",
        type=str,
        default="right_is_0",
        choices=["right_is_0", "right_is_1", "auto"],
        help=(
            "Meaning of input 'label'. right_is_0 means label=0 is correct/right, "
            "label=1 is hallucinated; right_is_1 is the opposite."
        ),
    )
    parser.add_argument(
        "--result_positive",
        type=str,
        default="correct",
        choices=["correct", "hallucinated"],
        help="Meaning of result_type=1 in output_judge.jsonl.",
    )
    return parser.parse_args()


def infer_correct_from_row(row: Dict[str, Any], label_semantic: str) -> Optional[bool]:
    label = row.get("label", None)
    response_type = row.get("response_type", None)

    if label is not None:
        if label_semantic == "right_is_0":
            return int(label) == 0
        if label_semantic == "right_is_1":
            return int(label) == 1

        # auto
        if response_type == "right":
            return int(label) == 0
        if response_type == "hallucinated":
            return int(label) == 1
        # fallback to the common convention in current outputs
        return int(label) == 0

    if response_type == "right":
        return True
    if response_type == "hallucinated":
        return False
    return None


def make_sample_id(row: Dict[str, Any], line_no: int) -> str:
    # Prefer explicit id if provided by upstream data.
    if row.get("id") is not None:
        base = str(row["id"])
    elif row.get("index") is not None:
        base = str(row["index"])
    else:
        base = f"line_{line_no}"

    # Keep candidate_index to avoid collisions when pairing=both.
    cand = row.get("candidate_index")
    if cand is not None:
        return f"{base}_{cand}"
    return base


def to_result_type(is_correct: bool, result_positive: str) -> int:
    if result_positive == "correct":
        return 1 if is_correct else 0
    return 1 if (not is_correct) else 0


def main() -> None:
    args = parse_args()
    in_path = Path(args.input_jsonl)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    icr_scores: Dict[str, Any] = {}
    judge_rows = []
    total = 0
    skipped = 0

    with in_path.open("r", encoding="utf-8") as fin:
        for line_no, line in enumerate(fin, start=1):
            text = line.strip()
            if not text:
                continue
            total += 1
            row = json.loads(text)

            if "icr_scores" not in row:
                skipped += 1
                continue

            is_correct = infer_correct_from_row(row, args.label_semantic)
            if is_correct is None:
                skipped += 1
                continue

            sample_id = make_sample_id(row, line_no)
            icr_scores[sample_id] = row["icr_scores"]
            judge_rows.append({"id": sample_id, "result_type": to_result_type(is_correct, args.result_positive)})

    torch.save(icr_scores, out_dir / "icr_score.pt")
    with (out_dir / "output_judge.jsonl").open("w", encoding="utf-8") as fout:
        for obj in judge_rows:
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Done. total_rows={total}, converted={len(judge_rows)}, skipped={skipped}")
    print(f"Saved: {out_dir / 'icr_score.pt'}")
    print(f"Saved: {out_dir / 'output_judge.jsonl'}")


if __name__ == "__main__":
    main()
