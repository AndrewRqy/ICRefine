# Common Commands

All commands run from `/Users/renqy/Desktop/School/CHAI Project/ICRefine` unless noted.

---

## vLLM

**Restart vLLM on cluster (SLURM)**
```bash
bash restart_vllm.sh              # DeepSeek-R1-32B (default, 2x A100-80GB)
bash restart_vllm.sh deepseek14b  # DeepSeek-R1-14B (1x A100-40GB)
bash restart_vllm.sh llama70b     # Llama-3.3-70B-Instruct (2x A100-80GB)
bash restart_vllm.sh gemma31b     # Gemma-4-31B-IT (1x A100-80GB)
```

**SSH port-forward to vLLM (run locally)**
```bash
# Replace <node> with the assigned node shown in squeue
ssh -L 8000:<node>:8000 <cluster-login>
```

---

## ICR_partition

**Resume run on hard_combined (269 items)**
```bash
python -m ICR_partition.pipeline \
    --dataset ../SAIR_eval_pipeline/datasets/hard_combined.jsonl \
    --oracle-csv gpt5.4_hard_correct.csv \
    --prior-knowledge ../SAIR_eval_pipeline/prompts/NeuriCo_cheatsheet.txt \
    --no-render-limit \
    --model-score deepseek-r1-32b \
    --model-casestudy gpt-4o \
    --max-outer-iters 5 \
    --partition-concurrency 8 \
    --concurrency 50 \
    --no-cot-first \
    --output-dir runs/partition_hard_combined \
    --resume
```
Ctrl+C stops gracefully (finishes current iteration, saves checkpoint).
Add `2>&1 | tee runs/partition_hard_combined.log` to also save output to a file.

**Fresh run on hard1 only**
```bash
python -m ICR_partition.pipeline \
    --dataset ../SAIR_eval_pipeline/datasets/hard1.jsonl \
    --oracle-csv gpt5.4_hard_correct.csv \
    --prior-knowledge ../SAIR_eval_pipeline/prompts/NeuriCo_cheatsheet.txt \
    --no-render-limit \
    --model-score deepseek-r1-32b \
    --model-casestudy gpt-4o \
    --max-outer-iters 5 \
    --partition-concurrency 8 \
    --concurrency 50 \
    --no-cot-first \
    --output-dir runs/partition_hard1
```

---

## Scoring / Evaluation

**Score a cheatsheet on hard1**
```bash
python3 score_eval.py \
    --dataset ../SAIR_eval_pipeline/datasets/hard1.jsonl \
    --cheatsheet runs/partition_hard_combined/cheatsheet_current \
    --model deepseek-r1-32b \
    --concurrency 50 \
    --no-cot-first
```

**Score a cheatsheet on hard_combined**
```bash
python3 score_eval.py \
    --dataset ../SAIR_eval_pipeline/datasets/hard_combined.jsonl \
    --cheatsheet runs/partition_hard_combined/cheatsheet_current \
    --model deepseek-r1-32b \
    --concurrency 50 \
    --no-cot-first
```

Results stream to `runs/eval_<dataset>_<cheatsheet>.jsonl` automatically.
Ctrl+C to stop; re-run the same command to resume from where it left off.

---

## ICR_select

**Ablation run (resumes if interrupted)**
```bash
python run_ablation.py \
    --dataset ../SAIR_eval_pipeline/datasets/hard1.jsonl \
    --oracle-csv gpt5.4_hard_correct.csv \
    --model-score deepseek-r1-32b \
    --model-casestudy gpt-4o \
    --output-dir runs/ablation_hard1 \
    --resume \
    > runs/ablation_hard1.log 2>&1 & echo 'PID:' $!
```

---

## Smoke Tests

```bash
python smoke_partition.py      # ICR_partition (all mocked)
python smoke_roadmap.py        # roadmap generation
python smoke_test_gates.py     # fix-rate / regression gates
```
