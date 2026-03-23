    #!/usr/bin/env bash
set -euo pipefail

# ============================================================
# ICR Probe 实验运行器
# 用法：
#   bash run.sh                          # 默认配置
#   bash run.sh configs/my_config.yaml   # 自定义配置
#   bash run.sh --fg                     # 前台运行
# ============================================================

CONFIG="${1:-configs/experiment.yaml}"
FG_MODE=false
if [[ "${1:-}" == "--fg" ]] || [[ "${2:-}" == "--fg" ]]; then
    FG_MODE=true
    [[ "${1:-}" == "--fg" ]] && CONFIG="${2:-configs/experiment.yaml}"
fi

if [[ ! -f "$CONFIG" ]]; then
    echo "Error: Config file not found: $CONFIG"
    exit 1
fi

# 固定参数
DATA_DIR="${DATA_DIR:-./data}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs}"
LOG_DIR="${LOG_DIR:-./logs}"
SEED=42
ATTN_IMPL=eager
MAX_RESP_TOKENS=128

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

PYTHON="${PYTHON:-python}"

# 从 YAML 读取模型列表，为每个模型运行 HaluEval QA
$PYTHON - "$CONFIG" "$FG_MODE" "$DATA_DIR" "$OUTPUT_DIR" "$LOG_DIR" \
         "$SEED" "$ATTN_IMPL" "$MAX_RESP_TOKENS" <<'PYEOF'
import sys, subprocess, os

config_path, fg_mode = sys.argv[1], sys.argv[2] == "True"
data_dir, output_dir, log_dir = sys.argv[3], sys.argv[4], sys.argv[5]
seed, attn_impl, max_resp = sys.argv[6], sys.argv[7], sys.argv[8]

try:
    import yaml
except ImportError:
    print("Error: pyyaml not installed. Run: pip install pyyaml")
    sys.exit(1)

with open(config_path) as f:
    cfg = yaml.safe_load(f)

models = cfg.get("models", [])
if not models:
    print("No models configured."); sys.exit(1)

for model_path in models:
    # 从路径提取短名: "Qwen/Qwen2.5-7B-Instruct" -> "Qwen2.5-7B-Instruct"
    name = model_path.rsplit("/", 1)[-1]

    out_file = f"{output_dir}/icr_halueval_qa_random_{name}.jsonl"
    log_file = f"{log_dir}/icr_halueval_qa_{name}.log"

    cmd = [
        sys.executable, "scripts/compute_icr_halueval.py",
        "--model_name_or_path", model_path,
        "--data_path", f"{data_dir}/HaluEval/qa_data.json",
        "--task", "qa",
        "--pairing", "random",
        "--seed", seed,
        "--attn_implementation", attn_impl,
        "--max_response_tokens", max_resp,
        "--dtype", "float16",
        "--output_path", out_file,
    ]

    print(f"\n{'='*60}")
    print(f"Model:  {name} ({model_path})")
    print(f"Output: {out_file}")
    print(f"{'='*60}")

    if fg_mode:
        subprocess.run(cmd, check=True)
    else:
        log_f = open(log_file, "w")
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
        print(f"Background PID {proc.pid}. Logs: {log_file}")

print("\nAll jobs launched.")
PYEOF
