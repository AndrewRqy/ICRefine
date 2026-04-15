#!/usr/bin/env bash
# run_cluster_ensemble.sh — Launch Llama-3.3-70B + Gemma-4-31B vLLM servers, then
# submit ICR_partition training as a dependent job that waits for both to be ready.
#
# Run from the ICRefine/ directory on the cluster login node:
#   bash run_cluster_ensemble.sh \
#       --dataset  /net/projects2/chai-lab/sair/hard1.jsonl \
#       --oracle   /net/projects2/chai-lab/sair/gpt5.4_hard_correct.csv \
#       --output   /net/scratch/renqy/runs/ensemble_run1
#
# Optional flags (passed through to ICR_partition.pipeline):
#   --prior-knowledge  /path/to/NeuriCo_cheatsheet.txt
#   --limit            N
#   --max-outer-iters  N   (default 5)
#   --concurrency      N   (default 25)
#   --reasoning-effort low|medium|high|none  (default low)
#
# The training job writes checkpoints to --output; --resume restarts from there.

set -euo pipefail

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------

DATASET=""
ORACLE=""
OUTPUT_DIR=""
PRIOR_KNOWLEDGE=""
LIMIT=""
MAX_OUTER_ITERS=5
CONCURRENCY=25
REASONING_EFFORT="low"
EXTRA_TRAIN_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)         DATASET="$2";          shift 2 ;;
        --oracle)          ORACLE="$2";            shift 2 ;;
        --output)          OUTPUT_DIR="$2";        shift 2 ;;
        --prior-knowledge) PRIOR_KNOWLEDGE="$2";   shift 2 ;;
        --limit)           LIMIT="$2";             shift 2 ;;
        --max-outer-iters) MAX_OUTER_ITERS="$2";   shift 2 ;;
        --concurrency)     CONCURRENCY="$2";       shift 2 ;;
        --reasoning-effort) REASONING_EFFORT="$2"; shift 2 ;;
        *)
            # Unknown args forwarded directly to the training command
            EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS $1"
            shift ;;
    esac
done

if [[ -z "$DATASET" || -z "$ORACLE" || -z "$OUTPUT_DIR" ]]; then
    echo "Usage: bash run_cluster_ensemble.sh --dataset FILE --oracle FILE --output DIR [opts]"
    exit 1
fi

# ---------------------------------------------------------------------------
# Cluster paths (edit DSI_USER if running as a different user)
# ---------------------------------------------------------------------------

DSI_USER="renqy"
VLLM_BIN="/net/scratch/${DSI_USER}/vllm-env/bin/vllm"
LOG_DIR="/net/scratch/${DSI_USER}"
SCRATCH="/net/scratch/${DSI_USER}"

LLAMA70B_PATH="/net/projects2/chai-lab/shared_models/hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b"
GEMMA31B_PATH="/net/projects2/chai-lab/shared_models/google/gemma-4-31B-it"

# Endpoint sentinel files — training job polls these until servers are up
LLAMA_ENDPOINT_FILE="${SCRATCH}/vllm-llama70b.endpoint"
GEMMA_ENDPOINT_FILE="${SCRATCH}/vllm-gemma31b.endpoint"

# Remove stale sentinel files from any previous run
rm -f "$LLAMA_ENDPOINT_FILE" "$GEMMA_ENDPOINT_FILE"

# ICRefine project root (assumes script lives inside it)
ICR_ROOT="$(cd "$(dirname "$0")" && pwd)"

# ---------------------------------------------------------------------------
# Auto-install vLLM if needed
# ---------------------------------------------------------------------------

if [[ ! -x "$VLLM_BIN" ]]; then
    echo "[run_cluster_ensemble] vLLM not found at $VLLM_BIN — installing..."
    python3 -m venv "${SCRATCH}/vllm-env"
    "${SCRATCH}/vllm-env/bin/pip" install --quiet --upgrade pip
    "${SCRATCH}/vllm-env/bin/pip" install --quiet vllm
    echo "[run_cluster_ensemble] vLLM installed."
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
            --port 8000 &
        VLLM_PID=\$!
        # Poll until server is accepting requests, then write endpoint
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
# Job 2 — Gemma-4-31B on port 8001 (1×A100-80GB)
# ---------------------------------------------------------------------------

echo "Submitting Gemma-4-31B vLLM job..."

GEMMA_JID=$(sbatch \
    --partition=general \
    --gres=gpu:a100:1 \
    --mem=80G \
    --time=12:00:00 \
    --job-name=vllm-gemma31b \
    --output="${LOG_DIR}/vllm-gemma31b-%j.log" \
    --parsable \
    --wrap="bash -c '
        source ~/.bashrc
        ${VLLM_BIN} serve \
            ${GEMMA31B_PATH} \
            --served-model-name gemma-4-31b \
            --port 8001 &
        VLLM_PID=\$!
        # Poll until server is accepting requests, then write endpoint
        until curl -sf http://localhost:8001/v1/models > /dev/null 2>&1; do
            sleep 10
        done
        echo \"http://\$(hostname):8001/v1/chat/completions\" > ${GEMMA_ENDPOINT_FILE}
        echo \"[gemma31b] Endpoint ready: \$(cat ${GEMMA_ENDPOINT_FILE})\"
        wait \$VLLM_PID
    '")

echo "  Gemma job ID : $GEMMA_JID"
echo "  Log          : ${LOG_DIR}/vllm-gemma31b-${GEMMA_JID}.log"

# ---------------------------------------------------------------------------
# Build optional training flags
# ---------------------------------------------------------------------------

TRAIN_OPTS=""
[[ -n "$PRIOR_KNOWLEDGE" ]] && TRAIN_OPTS="$TRAIN_OPTS --prior-knowledge ${PRIOR_KNOWLEDGE}"
[[ -n "$LIMIT"            ]] && TRAIN_OPTS="$TRAIN_OPTS --limit ${LIMIT}"
TRAIN_OPTS="$TRAIN_OPTS --max-outer-iters ${MAX_OUTER_ITERS}"
TRAIN_OPTS="$TRAIN_OPTS --concurrency ${CONCURRENCY}"
TRAIN_OPTS="$TRAIN_OPTS --reasoning-effort ${REASONING_EFFORT}"
TRAIN_OPTS="$TRAIN_OPTS ${EXTRA_TRAIN_ARGS}"

# ---------------------------------------------------------------------------
# Job 3 — ICR_partition training (starts after both vLLM jobs begin)
# ---------------------------------------------------------------------------

echo "Submitting ICR_partition training job (after:${LLAMA_JID}:${GEMMA_JID})..."

TRAIN_JID=$(sbatch \
    --partition=general \
    --gres=gpu:0 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=24:00:00 \
    --job-name=icr-ensemble \
    --output="${LOG_DIR}/icr-ensemble-%j.log" \
    --dependency="after:${LLAMA_JID}:${GEMMA_JID}" \
    --parsable \
    --wrap="bash -c '
        source ~/.bashrc
        cd ${ICR_ROOT}

        echo \"[train] Waiting for Llama endpoint at ${LLAMA_ENDPOINT_FILE}...\"
        until [[ -f ${LLAMA_ENDPOINT_FILE} ]]; do sleep 15; done
        LLAMA_URL=\$(cat ${LLAMA_ENDPOINT_FILE})
        echo \"[train] Llama ready: \$LLAMA_URL\"

        echo \"[train] Waiting for Gemma endpoint at ${GEMMA_ENDPOINT_FILE}...\"
        until [[ -f ${GEMMA_ENDPOINT_FILE} ]]; do sleep 15; done
        GEMMA_URL=\$(cat ${GEMMA_ENDPOINT_FILE})
        echo \"[train] Gemma ready: \$GEMMA_URL\"

        export VLLM_BASE_URL=\"\$LLAMA_URL\"
        export VLLM_MODEL=\"llama-3.3-70b\"
        export VLLM_BASE_URL_2=\"\$GEMMA_URL\"
        export VLLM_MODEL_2=\"gemma-4-31b\"
        export OPENROUTER_API_KEY=\"unused-local\"

        python3 -m ICR_partition.pipeline \
            --dataset     ${DATASET} \
            --oracle-csv  ${ORACLE} \
            --output-dir  ${OUTPUT_DIR} \
            --model-score       llama-3.3-70b \
            --model-score-2     gemma-4-31b \
            --model-score-weights 1.0,1.0 \
            --model-casestudy   llama-3.3-70b \
            ${TRAIN_OPTS}
    '")

echo "  Training job ID : $TRAIN_JID"
echo "  Log             : ${LOG_DIR}/icr-ensemble-${TRAIN_JID}.log"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

echo ""
echo "========================================"
echo "  All jobs submitted"
echo "========================================"
echo "  vLLM Llama  : $LLAMA_JID  (port 8000)"
echo "  vLLM Gemma  : $GEMMA_JID  (port 8001)"
echo "  Training    : $TRAIN_JID  (starts after both vLLMs begin)"
echo ""
echo "Monitor:"
echo "  squeue -u \$USER"
echo "  tail -f ${LOG_DIR}/icr-ensemble-${TRAIN_JID}.log"
echo ""
echo "Output dir  : ${OUTPUT_DIR}"
echo "Checkpoints : ${OUTPUT_DIR}/cheatsheet_current.json  (saved every iteration)"
echo ""
echo "To resume if training is interrupted:"
echo "  bash run_cluster_ensemble.sh \\"
echo "      --dataset  ${DATASET} \\"
echo "      --oracle   ${ORACLE} \\"
echo "      --output   ${OUTPUT_DIR} \\"
echo "      --resume"
