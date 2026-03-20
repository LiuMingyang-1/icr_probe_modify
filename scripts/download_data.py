#!/usr/bin/env python3
"""Download HaluEval (and optionally SQuAD 2.0) datasets to the local data/ directory."""

import argparse
import json
import os
from pathlib import Path


def download_halueval(data_dir: Path) -> None:
    """Download HaluEval qa_data.json from HuggingFace Hub."""
    from huggingface_hub import hf_hub_download

    out_dir = data_dir / "HaluEval"
    out_dir.mkdir(parents=True, exist_ok=True)

    dest = out_dir / "qa_data.json"
    if dest.exists():
        print(f"[HaluEval] Already exists: {dest}")
        return

    print("[HaluEval] Downloading qa_data.json from pminervini/HaluEval ...")
    local = hf_hub_download(
        repo_id="pminervini/HaluEval",
        filename="qa_data.json",
        repo_type="dataset",
        local_dir=str(out_dir),
    )
    print(f"[HaluEval] Saved to: {local}")


def download_squad2(data_dir: Path) -> None:
    """Download SQuAD 2.0 dev set from HuggingFace datasets."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("[SQuAD2] 'datasets' package not installed, skipping.")
        return

    out_dir = data_dir / "SQuAD2.0"
    out_dir.mkdir(parents=True, exist_ok=True)

    dest = out_dir / "dev-v2.0.json"
    if dest.exists():
        print(f"[SQuAD2] Already exists: {dest}")
        return

    print("[SQuAD2] Downloading SQuAD 2.0 validation split ...")
    ds = load_dataset("rajpurkar/squad_v2", split="validation")

    # Convert HuggingFace dataset to the original SQuAD JSON format
    data_dict: dict = {"version": "v2.0", "data": []}
    for row in ds:
        answers = row["answers"]
        answer_list = [
            {"text": t, "answer_start": s}
            for t, s in zip(answers["text"], answers["answer_start"])
        ]
        qa = {
            "id": row["id"],
            "question": row["question"],
            "answers": answer_list,
            "is_impossible": len(answer_list) == 0,
        }
        paragraph = {
            "context": row["context"],
            "qas": [qa],
        }
        data_dict["data"].append({"title": row.get("title", ""), "paragraphs": [paragraph]})

    with dest.open("w", encoding="utf-8") as f:
        json.dump(data_dict, f, ensure_ascii=False, indent=2)

    print(f"[SQuAD2] Saved to: {dest}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download datasets for ICR Probe experiments.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.environ.get("DATA_DIR", "./data"),
        help="Root directory to store datasets (default: ./data or $DATA_DIR)",
    )
    parser.add_argument("--halueval", action="store_true", default=True, help="Download HaluEval (default: on)")
    parser.add_argument("--no_halueval", dest="halueval", action="store_false")
    parser.add_argument("--squad2", action="store_true", default=False, help="Also download SQuAD 2.0 (default: off)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.halueval:
        download_halueval(data_dir)

    if args.squad2:
        download_squad2(data_dir)

    print("Done.")


if __name__ == "__main__":
    main()
