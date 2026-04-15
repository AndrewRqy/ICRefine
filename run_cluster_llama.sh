#!/usr/bin/env bash
# run_cluster_llama.sh — Launch Llama-3.3-70B vLLM server, then submit
# ICR_partition training as a dependent job that waits for it to be ready.
#
# Single-model run (no Gemma, no ensemble scoring).
# For two-model ensemble see run_cluster_ensemble.sh.
#
# Run from the ICRefine/ directory on the cluster login node:
#   bash run_cluster_llama.sh
#   bash run_cluster_llama.sh --output /net/scratch/renqy/runs/my_run
#
# Defaults (all parameters pre-configured to match previous hard runs):
#   --dataset          <ICRefine>/../SAIR_eval_pipeline/datasets/hard.jsonl
#   --oracle           <ICRefine>/gpt5.4_hard_correct.csv
#   --prior-knowledge  <ICRefine>/../SAIR_eval_pipeline/prompts/NeuriCo_cheatsheet.txt
#   --output           /net/scratch/renqy/runs/icr_llama_<timestamp>  (auto-generated)
#   --max-outer-iters  5
#   --concurrency      50
#   --partition-concurrency 8
#   --reasoning-effort low
#   --no-render-limit  (on)
#   --no-cot-first     (on)
#
# Resume behaviour:
#   If <output>/cheatsheet_current.json exists when the training job starts,
#   --resume is passed automatically to ICR_partition.  No extra flags needed —
#   just rerun the same command with the same --output to continue an
#   interrupted run (a new vLLM job will be submitted and training picks up
#   from the latest checkpoint).

set -euo pipefail

# ---------------------------------------------------------------------------
# Cluster paths (edit DSI_USER if running as a different user)
# ---------------------------------------------------------------------------

DSI_USER="renqy"
VLLM_BIN="/net/scratch/${DSI_USER}/vllm-env/bin/vllm"
LOG_DIR="/net/scratch/${DSI_USER}"
SCRATCH="/net/scratch/${DSI_USER}"

LLAMA70B_PATH="/net/projects2/chai-lab/shared_models/hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b"

# ICRefine project root (directory containing this script)
ICR_ROOT="$(cd "$(dirname "$0")" && pwd)"

# ---------------------------------------------------------------------------
# Defaults (match previous hard runs exactly)
# ---------------------------------------------------------------------------

DATASET="${ICR_ROOT}/../SAIR_eval_pipeline/datasets/hard.jsonl"
ORACLE="${ICR_ROOT}/gpt5.4_hard_correct.csv"
PRIOR_KNOWLEDGE="${ICR_ROOT}/../SAIR_eval_pipeline/prompts/NeuriCo_cheatsheet.txt"
OUTPUT_DIR="${SCRATCH}/runs/icr_llama_$(date +%Y%m%d_%H%M%S)"
LIMIT=""
MAX_OUTER_ITERS=5
PARTITION_CONCURRENCY=8
CONCURRENCY=50
REASONING_EFFORT="low"
NO_RENDER_LIMIT="--no-render-limit"
NO_COT_FIRST="--no-cot-first"
EXTRA_TRAIN_ARGS=""

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)               DATASET="$2";               shift 2 ;;
        --oracle)                ORACLE="$2";                 shift 2 ;;
        --output)                OUTPUT_DIR="$2";             shift 2 ;;
        --prior-knowledge)       PRIOR_KNOWLEDGE="$2";        shift 2 ;;
        --limit)                 LIMIT="$2";                  shift 2 ;;
        --max-outer-iters)       MAX_OUTER_ITERS="$2";        shift 2 ;;
        --partition-concurrency) PARTITION_CONCURRENCY="$2";  shift 2 ;;
        --concurrency)           CONCURRENCY="$2";            shift 2 ;;
        --reasoning-effort)      REASONING_EFFORT="$2";       shift 2 ;;
        --no-render-limit)       NO_RENDER_LIMIT="--no-render-limit"; shift ;;
        --cot-first)             NO_COT_FIRST="";             shift ;;
        *)
            # Unknown args forwarded directly to the training command
            EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS} $1"
            shift ;;
    esac
done

# Resolve all paths to absolute (handles ../relative forms)
DATASET="$(cd "$(dirname "$DATASET")" && pwd)/$(basename "$DATASET")"
ORACLE="$(cd "$(dirname "$ORACLE")" && pwd)/$(basename "$ORACLE")"
[[ -n "$PRIOR_KNOWLEDGE" ]] && \
    PRIOR_KNOWLEDGE="$(cd "$(dirname "$PRIOR_KNOWLEDGE")" && pwd)/$(basename "$PRIOR_KNOWLEDGE")"

# Validate inputs
if [[ ! -f "$DATASET" ]]; then
    echo "[error] Dataset not found: $DATASET"
    exit 1
fi
if [[ ! -f "$ORACLE" ]]; then
    echo "[error] Oracle CSV not found: $ORACLE"
    exit 1
fi
if [[ -n "$PRIOR_KNOWLEDGE" && ! -f "$PRIOR_KNOWLEDGE" ]]; then
    echo "[error] Prior knowledge file not found: $PRIOR_KNOWLEDGE"
    exit 1
fi

# ---------------------------------------------------------------------------
# Endpoint sentinel file — training job polls this until server is up.
# Named with a timestamp so concurrent runs don't collide.
# ---------------------------------------------------------------------------

RUN_TS="$(date +%Y%m%d_%H%M%S)"
LLAMA_ENDPOINT_FILE="${SCRATCH}/vllm-llama70b-${RUN_TS}.endpoint"

# ---------------------------------------------------------------------------
# Auto-install vLLM if needed
# ---------------------------------------------------------------------------

if [[ ! -x "$VLLM_BIN" ]]; then
    echo "[run_cluster_llama] vLLM not found at $VLLM_BIN — installing..."
    python3 -m venv "${SCRATCH}/vllm-env"
    "${SCRATCH}/vllm-env/bin/pip" install --quiet --upgrade pip
    "${SCRATCH}/vllm-env/bin/pip" install --quiet vllm
    echo "[run_cluster_llama] vLLM installed."
fi

# ---------------------------------------------------------------------------
# Job 1 — Llama-3.3-70B on port 8000 (2×A100-80GB, tensor-parallel-size 2)
# ---------------------------------------------------------------------------

echo "Submitting Llama-3.3-70B vLLM job..."

LLAMA_JID=$(sbatch \
    --partition=general \
    --gres=gpu:a100:2 \
    --mem=160G \
    --time=12:00:00 \
    --job-name=vllm-llama70b \
    --output="${LOG_DIR}/vllm-llama70b-%j.log" \
    --parsable \
    --wrap="bash -c '
        source ~/.bashrc
        ${VLLM_BIN} serve \
            ${LLAMA70B_PATH} \
            --served-model-name llama-3.3-70b \
            --tensor-parallel-size 2 \
            --max-model-len 65536 \
            --port 8000 &
        VLLM_PID=\$!
        # Poll until server accepts requests, then write endpoint sentinel
        until curl -sf http://localhost:8000/v1/models > /dev/null 2>&1; do
            sleep 10
        done
        echo \"http://\$(hostname):8000/v1/chat/completions\" > ${LLAMA_ENDPOINT_FILE}
        echo \"[llama70b] Endpoint ready: \$(cat ${LLAMA_ENDPOINT_FILE})\"
        wait \$VLLM_PID
    '")

echo "  Llama job ID : $LLAMA_JID"
echo "  Log          : ${LOG_DIR}/vllm-llama70b-${LLAMA_JID}.log"

# ---------------------------------------------------------------------------
# Build optional training flags
# ---------------------------------------------------------------------------

TRAIN_OPTS=""
[[ -n "$PRIOR_KNOWLEDGE" ]] && TRAIN_OPTS="${TRAIN_OPTS} --prior-knowledge ${PRIOR_KNOWLEDGE}"
[[ -n "$LIMIT"           ]] && TRAIN_OPTS="${TRAIN_OPTS} --limit ${LIMIT}"
TRAIN_OPTS="${TRAIN_OPTS} --max-outer-iters ${MAX_OUTER_ITERS}"
TRAIN_OPTS="${TRAIN_OPTS} --partition-concurrency ${PARTITION_CONCURRENCY}"
TRAIN_OPTS="${TRAIN_OPTS} --concurrency ${CONCURRENCY}"
TRAIN_OPTS="${TRAIN_OPTS} --reasoning-effort ${REASONING_EFFORT}"
[[ -n "$NO_RENDER_LIMIT" ]] && TRAIN_OPTS="${TRAIN_OPTS} ${NO_RENDER_LIMIT}"
[[ -n "$NO_COT_FIRST"    ]] && TRAIN_OPTS="${TRAIN_OPTS} ${NO_COT_FIRST}"
TRAIN_OPTS="${TRAIN_OPTS} ${EXTRA_TRAIN_ARGS}"

# ---------------------------------------------------------------------------
# Job 2 — ICR_partition training (starts after Llama vLLM job begins)
#
# Resume logic: if <output>/cheatsheet_current.json already exists when the
# job starts, --resume is passed automatically so training continues from
# the latest checkpoint rather than starting over.  This handles both
# explicit re-runs of an interrupted job and the case where SLURM pre-empts
# the training job mid-run.
# ---------------------------------------------------------------------------

echo "Submitting ICR_partition training job (after:${LLAMA_JID})..."

# Write the training script to a temp file to avoid --wrap quoting issues
# with multi-word TRAIN_OPTS expansions.
TRAIN_SCRIPT=$(mktemp "${SCRATCH}/icr_train_XXXXXX.sh")
cat > "$TRAIN_SCRIPT" << SCRIPTEOF
#!/bin/bash
source ~/.bashrc
cd ${ICR_ROOT}

# ---- Activate Python environment ----
if [[ -f .venv/bin/activate ]]; then
    source .venv/bin/activate
else
    echo "[train] .venv not found — running uv sync first ..."
    uv sync
    source .venv/bin/activate
fi

# ---- Wait for vLLM endpoint ----
echo "[train] Waiting for Llama endpoint (${LLAMA_ENDPOINT_FILE})..."
until [[ -f ${LLAMA_ENDPOINT_FILE} ]]; do sleep 15; done
LLAMA_URL=\$(cat ${LLAMA_ENDPOINT_FILE})
echo "[train] Llama ready: \$LLAMA_URL"

# ---- Set vLLM env vars ----
export VLLM_BASE_URL="\$LLAMA_URL"
export VLLM_MODEL="llama-3.3-70b"
export OPENROUTER_API_KEY="unused-local"

# ---- Auto-resume if checkpoint exists ----
RESUME_FLAG=""
if [[ -f ${OUTPUT_DIR}/cheatsheet_current.json ]]; then
    echo "[train] Checkpoint found at ${OUTPUT_DIR}/cheatsheet_current.json — resuming."
    RESUME_FLAG="--resume"
else
    echo "[train] No checkpoint found — starting fresh."
fi

# ---- Remove stale lock file from a previous crashed run ----
rm -f ${OUTPUT_DIR}/.icr_lock

# ---- Run training ----
python3 -m ICR_partition.pipeline \
    --dataset             ${DATASET} \
    --oracle-csv          ${ORACLE} \
    --output-dir          ${OUTPUT_DIR} \
    --model-score         llama-3.3-70b \
    --model-casestudy     llama-3.3-70b \
    \${RESUME_FLAG} \
    ${TRAIN_OPTS}

EXIT_CODE=\$?

# ---- Clean up on success ----
if [[ \$EXIT_CODE -eq 0 ]]; then
    rm -f ${LLAMA_ENDPOINT_FILE}
    echo "[train] Completed successfully. Sentinel file removed."
else
    echo "[train] Exited with code \$EXIT_CODE. Sentinel file preserved for debugging."
    echo "[train] To resume: bash run_cluster_llama.sh --output ${OUTPUT_DIR} --dataset ${DATASET} --oracle ${ORACLE}"
fi

exit \$EXIT_CODE
SCRIPTEOF
chmod +x "$TRAIN_SCRIPT"

TRAIN_JID=$(sbatch \
    --partition=general \
    --gres=gpu:0 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=12:00:00 \
    --job-name=icr-llama \
    --output="${LOG_DIR}/icr-llama-%j.log" \
    --dependency="after:${LLAMA_JID}" \
    --parsable \
    "$TRAIN_SCRIPT")

echo "  Training job ID : $TRAIN_JID"
echo "  Log             : ${LOG_DIR}/icr-llama-${TRAIN_JID}.log"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

echo ""
echo "========================================"
echo "  All jobs submitted"
echo "========================================"
echo "  vLLM Llama  : $LLAMA_JID  (port 8000)"
echo "  Training    : $TRAIN_JID  (starts after vLLM begins)"
echo ""
echo "  dataset     : $DATASET"
echo "  oracle      : $ORACLE"
echo "  output      : $OUTPUT_DIR"
echo ""
echo "Monitor:"
echo "  squeue -u \$USER"
echo "  tail -f ${LOG_DIR}/icr-llama-${TRAIN_JID}.log"
echo ""
echo "Checkpoints : ${OUTPUT_DIR}/cheatsheet_current.json  (saved each outer iteration)"
echo ""
echo "To resume if training is interrupted (re-runs both jobs):"
echo "  bash run_cluster_llama.sh \\"
echo "      --output   ${OUTPUT_DIR} \\"
echo "      --dataset  ${DATASET} \\"
echo "      --oracle   ${ORACLE}"
