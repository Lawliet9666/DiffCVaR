#!/usr/bin/env bash
set -euo pipefail

# Optional: uncomment for a one-off login/key injection.
# Do not commit a real API key to a public repo.
# export WANDB_API_KEY="..."

export WANDB_MODE="${WANDB_MODE:-online}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_PROJECT="${WANDB_PROJECT:-diff_cvar}"
MODEL="${MODEL:-diff_cvar}"
ROBOT="${ROBOT:-single_integrator}"
RUN_NAME="${RUN_NAME:-${MODEL}}"
DEVICE="${DEVICE:-auto}"
NUM_ENVS="${NUM_ENVS:-8}"

cd "$(dirname "$0")/.."

python scripts/run.py \
  model="${MODEL}" \
  robot="${ROBOT}" \
  run_name="${RUN_NAME}" \
  wandb_entity="${WANDB_ENTITY}" \
  wandb_project="${WANDB_PROJECT}" \
  device="${DEVICE}" \
  trainer.num_envs="${NUM_ENVS}" \
  "$@"
