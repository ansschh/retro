#!/bin/bash
#SBATCH --job-name=rt5_smoke
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --partition=beta
#SBATCH --qos=debug
#SBATCH --output=/resnick/scratch/atiwari2/retro/logs/%x_%j.out
#SBATCH --error=/resnick/scratch/atiwari2/retro/logs/%x_%j.err

set -e
cd /resnick/scratch/atiwari2/retro
source env.sh
source .venv/bin/activate

mkdir -p data/smoke results logs

# Generate the hermetic smoke dataset (idempotent)
python scripts/data/make_smoke_dataset.py

# Run ReactionT5 inference, top-10
python -m baselines.reactiont5.run \
  --test-data data/smoke/uspto50k_smoke_test.parquet \
  --output-jsonl results/reactiont5_smoke_topk10.jsonl \
  --topk 10 \
  --batch-size 8

# Score and append a leaderboard row
python -m harness.leaderboard \
  --predictions results/reactiont5_smoke_topk10.jsonl \
  --leaderboard results/leaderboard.csv \
  --model-name reactiont5_v2_uspto50k \
  --model-checkpoint-sha sagawa_ReactionT5v2_USPTO_50k \
  --benchmark-name uspto50k_SMOKE \
  --benchmark-split-sha smoke_v0_10rxn \
  --wallclock-seconds $(grep -oP 'in \K[0-9.]+(?=s)' "$SLURM_SUBMIT_DIR/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" | head -1 || echo 0) \
  --hardware "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" \
  --git-sha "$(git rev-parse --short HEAD)"

echo
echo "--- leaderboard ---"
cat results/leaderboard.csv
