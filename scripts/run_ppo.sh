#!/usr/bin/env bash
set -euo pipefail


export WANDB_MODE="${WANDB_MODE:-online}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_PROJECT="${WANDB_PROJECT:-diff_cvar}"
: "${RUN_NAME:?Set RUN_NAME before launching training}"

cd "$(dirname "$0")/.."

python scripts/run_ppo_base.py \
  run_name="${RUN_NAME}" \
  wandb_entity="${WANDB_ENTITY}" \
  wandb_project="${WANDB_PROJECT}" \
  "$@"
