# HaluEval
python scripts/compute_icr_halueval.py \
    --model_name_or_path /data/sjx/models/Qwen3-4B-Instruct-2507 \
    --data_path /home/sjx/hallucination/dataset/HaluEval/data/qa_data.json \
    --task qa \
    --pairing random \
    --seed 42 \
    --attn_implementation eager \
    --output_path /home/sjx/hallucination/ICR_Probe/outputs/icr_qa_random_qwen3.jsonl

nohup python scripts/compute_icr_halueval.py \
    --model_name_or_path /data/sjx/models/Qwen2.5-7B-Instruct \
    --data_path /home/sjx/hallucination/dataset/HaluEval/data/qa_data.json \
    --task qa \
    --pairing random \
    --seed 42 \
    --attn_implementation eager \
    --output_path /home/sjx/hallucination/ICR_Probe/outputs/icr_halu_eval_random_qwen2.5.jsonl \
    > logs/icr_halu_eval_qwen2.5.log 2>&1 &


# SQuAD

