#!/usr/bin/env python3
import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.icr_score import ICRScore


TASK_CHOICES = ["auto", "squad2", "custom"]
PAIRING_CHOICES = ["auto", "both", "random", "right", "hallucinated", "single"]
NO_ANSWER_TEXT = "No answer."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute ICR scores for SQuAD2.0 samples.")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True, help="SQuAD2.0 file path")
    parser.add_argument("--output_path", type=str, required=True, help="Output JSONL path for ICR scores")

    parser.add_argument("--task", type=str, default="auto", choices=TASK_CHOICES)
    parser.add_argument(
        "--pairing",
        type=str,
        default="auto",
        choices=PAIRING_CHOICES,
        help="How to use right/hallucinated pairs. auto->random for squad2; both outputs two rows per sample.",
    )
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--prompt_key", type=str, default=None, help="Custom field name for prompt text")
    parser.add_argument("--response_key", type=str, default=None, help="Custom field name for response text")
    parser.add_argument("--label_key", type=str, default=None, help="Custom field name for hallucination label")
    parser.add_argument("--id_key", type=str, default=None, help="Custom field name for sample id")

    parser.add_argument("--split", type=str, default=None, help="Split key when JSON root is a dict")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_response_tokens", type=int, default=128)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2"],
        help="Attention backend used by the model. Use eager when output_attentions is required.",
    )

    parser.add_argument("--use_chat_template", action="store_true", default=True)
    parser.add_argument("--disable_chat_template", action="store_true")
    parser.add_argument("--system_prompt", type=str, default=None)

    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--top_p", type=float, default=0.1)
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "max", "min"])
    parser.add_argument("--attention_uniform", action="store_true")
    parser.add_argument("--hidden_uniform", action="store_true")
    parser.add_argument("--use_induction_head", action="store_true")
    parser.add_argument("--skew_threshold", type=float, default=0)
    parser.add_argument("--entropy_threshold", type=float, default=1e5)

    return parser.parse_args()


def load_records(path: Path, split: Optional[str]) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8")

    try:
        raw = json.loads(text)
        if isinstance(raw, list):
            return raw
        if isinstance(raw, dict):
            if split is not None:
                if split not in raw:
                    raise KeyError(f"split '{split}' not found in JSON keys: {list(raw.keys())}")
                if not isinstance(raw[split], list):
                    raise TypeError(f"JSON split '{split}' is not a list")
                return raw[split]
            for k in ("data", "train", "validation", "test", "dev"):
                if k in raw and isinstance(raw[k], list):
                    return raw[k]
            raise TypeError("JSON is a dict but no list-like split found; pass --split")
    except json.JSONDecodeError:
        pass

    records = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def flatten_squad2(path: Path) -> List[Dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if "data" not in raw or not isinstance(raw["data"], list):
        raise ValueError("SQuAD2.0 JSON must contain a 'data' list")

    rows: List[Dict[str, Any]] = []
    for article in raw["data"]:
        title = str(article.get("title", ""))
        for paragraph in article.get("paragraphs", []):
            context = str(paragraph.get("context", ""))
            for qa in paragraph.get("qas", []):
                answers = [a.get("text", "").strip() for a in qa.get("answers", []) if a.get("text", "").strip()]
                plausible = [
                    a.get("text", "").strip()
                    for a in qa.get("plausible_answers", [])
                    if a.get("text", "").strip()
                ]
                rows.append(
                    {
                        "id": str(qa["id"]),
                        "title": title,
                        "context": context,
                        "question": str(qa.get("question", "")),
                        "answers": answers,
                        "plausible_answers": plausible,
                        "is_impossible": bool(qa.get("is_impossible", False)),
                    }
                )
    if not rows:
        raise ValueError(f"No QA items found in {path}")
    return rows


def pick_first_key(record: Dict[str, Any], candidates: Iterable[str], explicit_key: Optional[str]) -> Optional[str]:
    if explicit_key is not None:
        if explicit_key not in record:
            raise KeyError(f"Key '{explicit_key}' not found in record keys: {list(record.keys())}")
        return explicit_key
    for key in candidates:
        if key in record:
            return key
    return None


def build_prompt(tokenizer, prompt: str, use_chat_template: bool, system_prompt: Optional[str]) -> str:
    if not use_chat_template or not hasattr(tokenizer, "apply_chat_template"):
        return prompt
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def dtype_from_name(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def tokenize_text(tokenizer, text: str, max_len: Optional[int] = None) -> torch.Tensor:
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    ids = encoded["input_ids"][0]
    if max_len is not None:
        ids = ids[:max_len]
    return ids


def infer_task(record: Dict[str, Any], specified: str) -> str:
    if specified != "auto":
        return specified
    keys = set(record.keys())
    if {"id", "context", "question", "answers", "is_impossible"}.issubset(keys):
        return "squad2"
    return "custom"


def normalize_binary_label(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value > 0)
    text = str(value).strip().lower()
    if text in {"yes", "y", "true", "1", "hallucinated", "hallucination"}:
        return 1
    if text in {"no", "n", "false", "0", "non-hallucinated", "non_hallucinated"}:
        return 0
    return None


def build_answer_pool(records: List[Dict[str, Any]]) -> List[str]:
    pool = set()
    for rec in records:
        for a in rec.get("answers", []):
            if a:
                pool.add(a)
        for a in rec.get("plausible_answers", []):
            if a:
                pool.add(a)
    pool.add(NO_ANSWER_TEXT)
    return list(pool)


def sample_wrong_answer(gold_answers: List[str], answer_pool: List[str], rng: random.Random) -> str:
    gold_set = set(gold_answers)
    candidates = [x for x in answer_pool if x not in gold_set]
    if not candidates:
        return NO_ANSWER_TEXT
    return rng.choice(candidates)


def make_squad2_candidates(
    record: Dict[str, Any],
    pairing: str,
    rng: random.Random,
    answer_pool: List[str],
) -> List[Dict[str, Any]]:
    if pairing in {"auto", "single"}:
        pairing = "random"

    prompt = (
        f"Context:\n{record['context']}\n\n"
        f"Question:\n{record['question']}\n\n"
        "Answer:"
    )
    is_impossible = bool(record.get("is_impossible", False))
    gold_answers = record.get("answers", [])
    plausible_answers = record.get("plausible_answers", [])

    if is_impossible:
        right = NO_ANSWER_TEXT
        hallu = plausible_answers[0] if plausible_answers else sample_wrong_answer([right], answer_pool, rng)
    else:
        if not gold_answers:
            return []
        right = gold_answers[0]
        hallu = plausible_answers[0] if plausible_answers else sample_wrong_answer(gold_answers, answer_pool, rng)

    both = [
        {"prompt": prompt, "response": right, "label": 0, "response_type": "right"},
        {"prompt": prompt, "response": hallu, "label": 1, "response_type": "hallucinated"},
    ]

    if pairing == "both":
        return both
    if pairing == "random":
        return [both[rng.randint(0, 1)]]
    if pairing == "right":
        return [both[0]]
    if pairing == "hallucinated":
        return [both[1]]
    raise ValueError(f"Pairing '{pairing}' is invalid for task 'squad2'")


def make_custom_candidate(record: Dict[str, Any], args: argparse.Namespace) -> List[Dict[str, Any]]:
    prompt_key = pick_first_key(record, ["question", "prompt", "query", "instruction", "input"], args.prompt_key)
    response_key = pick_first_key(record, ["answer", "response", "model_output", "output", "text"], args.response_key)
    label_key = pick_first_key(record, ["label", "hallucination", "is_hallucination", "binary_label"], args.label_key)

    if prompt_key is None or response_key is None:
        raise ValueError(
            "Custom mode requires detectable prompt/response keys. "
            "Please pass --prompt_key and --response_key explicitly."
        )

    label = normalize_binary_label(record.get(label_key)) if label_key is not None else None
    return [
        {
            "prompt": str(record[prompt_key]),
            "response": str(record[response_key]),
            "label": label,
            "response_type": response_key,
            "prompt_key": prompt_key,
            "response_key": response_key,
            "label_key": label_key,
        }
    ]


def collect_stepwise_cache(
    model,
    prompt_ids: torch.Tensor,
    response_ids: torch.Tensor,
    device: str,
) -> Tuple[List[Any], List[Any]]:
    hidden_states_steps: List[Any] = []
    attentions_steps: List[Any] = []

    with torch.no_grad():
        inputs = prompt_ids.unsqueeze(0).to(device)
        out = model(
            input_ids=inputs,
            output_hidden_states=True,
            output_attentions=True,
            use_cache=True,
            return_dict=True,
        )
        hidden_states_steps.append(out.hidden_states)
        attentions_steps.append(out.attentions)
        past_key_values = out.past_key_values

        for token_id in response_ids:
            next_input = token_id.view(1, 1).to(device)
            out = model(
                input_ids=next_input,
                past_key_values=past_key_values,
                output_hidden_states=True,
                output_attentions=True,
                use_cache=True,
                return_dict=True,
            )
            hidden_states_steps.append(out.hidden_states)
            attentions_steps.append(out.attentions)
            past_key_values = out.past_key_values

    return hidden_states_steps, attentions_steps


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    use_chat_template = args.use_chat_template and not args.disable_chat_template

    data_path = Path(args.data_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    device = args.device
    dtype = dtype_from_name(args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        device_map=None,
        attn_implementation=args.attn_implementation,
    ).to(device)
    model.eval()

    # SQuAD2.0 standard file is a dict with nested structure. Custom mode supports json/jsonl records.
    if args.task in {"auto", "squad2"}:
        records = flatten_squad2(data_path)
    else:
        records = load_records(data_path, args.split)

    if args.start_index:
        records = records[args.start_index :]
    if args.max_samples is not None:
        records = records[: args.max_samples]
    if not records:
        raise ValueError("No records to process")

    inferred_task = infer_task(records[0], args.task)
    id_key = pick_first_key(records[0], ["id", "ID", "sample_id", "idx"], args.id_key)
    answer_pool = build_answer_pool(records) if inferred_task == "squad2" else []

    written = 0
    skipped = 0

    with output_path.open("w", encoding="utf-8") as fout:
        for i, rec in enumerate(records):
            task = infer_task(rec, inferred_task)
            if task == "squad2":
                candidates = make_squad2_candidates(rec, args.pairing, rng, answer_pool)
            else:
                candidates = make_custom_candidate(rec, args)

            for cand_idx, cand in enumerate(candidates):
                prompt = cand["prompt"]
                response = cand["response"]

                full_prompt = build_prompt(
                    tokenizer,
                    prompt,
                    use_chat_template=use_chat_template,
                    system_prompt=args.system_prompt,
                )
                prompt_ids = tokenize_text(tokenizer, full_prompt)
                response_ids = tokenize_text(tokenizer, response, max_len=args.max_response_tokens)

                if response_ids.numel() == 0:
                    skipped += 1
                    continue

                hidden_states, attentions = collect_stepwise_cache(
                    model=model,
                    prompt_ids=prompt_ids,
                    response_ids=response_ids,
                    device=device,
                )

                input_len = int(prompt_ids.numel())
                core_positions = {
                    "user_prompt_start": 0,
                    "user_prompt_end": input_len,
                    "response_start": input_len,
                }

                icr_calculator = ICRScore(
                    hidden_states=hidden_states,
                    attentions=attentions,
                    skew_threshold=args.skew_threshold,
                    entropy_threshold=args.entropy_threshold,
                    core_positions=core_positions,
                    icr_device=device,
                )
                icr_scores, top_p_mean = icr_calculator.compute_icr(
                    top_k=args.top_k,
                    top_p=args.top_p,
                    pooling=args.pooling,
                    attention_uniform=args.attention_uniform,
                    hidden_uniform=args.hidden_uniform,
                    use_induction_head=args.use_induction_head,
                )

                row = {
                    "index": i + args.start_index,
                    "candidate_index": cand_idx,
                    "task": task,
                    "pairing": args.pairing,
                    "prompt": prompt,
                    "response": response,
                    "response_type": cand.get("response_type"),
                    "label": cand.get("label"),
                    "model_name_or_path": args.model_name_or_path,
                    "icr_scores": icr_scores,
                    "top_p_mean": float(top_p_mean),
                    "num_layers": len(icr_scores),
                    "num_response_tokens": int(response_ids.numel()),
                    "core_positions": core_positions,
                }
                if task == "squad2":
                    row["question"] = rec.get("question")
                    row["context"] = rec.get("context")
                    row["title"] = rec.get("title")
                    row["is_impossible"] = rec.get("is_impossible")

                if id_key is not None and id_key in rec:
                    row["id"] = rec[id_key]

                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1

            if (i + 1) % 10 == 0:
                print(f"Processed samples: {i + 1}/{len(records)}, written rows: {written}, skipped: {skipped}")

    print(f"Done. Saved ICR scores to {output_path}. Written rows: {written}, skipped rows: {skipped}")


if __name__ == "__main__":
    main()
