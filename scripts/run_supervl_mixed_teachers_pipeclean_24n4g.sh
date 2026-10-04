#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# 24-node pipeclean for the mixed-teacher Super VL 3.5 recipe on the
# SingleController path. Thin wrapper over run_supervl_mixed_teachers_nv_main.sh,
# which already carries the container env, caches, mounts and the entrypoint.
#
# Shape: 16 training nodes (TP2 * CP2 * EP16 = 64 GPUs, the production layout,
# unchanged) plus 8 generation nodes, against the production recipe's 16 + 16.
# Steps drop 125 -> 5 and the cohort 128 -> 32 prompts. All of that lives in the
# pipeclean config, not here.
#
# Two things this wrapper has to work around in the base launcher:
#
#   1. Its env block is a flat list of unconditional `export VAR=value` lines, so
#      exporting a var before calling it has no effect. CONFIG is the exception --
#      it was made `${CONFIG:-...}` so a caller can select a recipe. The node
#      counts and step budget it exports (NUM_NODES, MAX_STEPS, TRAIN_GBS, ...)
#      are read by nothing in this repo, so there is no point setting them; the
#      config is the source of truth.
#
#   2. It hardcodes Rohit's MM_TRAINER_RESULTS_DIR and MM_TRAINER_WANDB_*, which
#      the config reads via `oc.env`. Writing there would land this run's output in
#      someone else's experiment directory, so results, logs and W&B identity are
#      redirected with `++` Hydra overrides instead -- the same approach
#      run_router_replay_off.sh uses. Overrides are appended before "$@", so
#      anything passed on the command line still wins.
#
# The HF/uv/Triton caches still point at Rohit's Lustre directories. That is
# deliberate for a pipeclean -- it avoids re-downloading the model -- but it means
# this writes into his cache tree.
#
# This script does not allocate nodes; run it inside an allocation sized for
# NUM_NODES below (the base launcher assumes Ray is already up).
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
BASE_LAUNCHER="${SCRIPT_DIR}/run_supervl_mixed_teachers_nv_main.sh"

# The base launcher selects --env / --setup off $1, so those modes have to be
# forwarded on their own rather than behind the Hydra overrides below.
if [[ "${1:-}" == "--env" || "${1:-}" == "--setup" ]]; then
  exec bash "${BASE_LAUNCHER}" "$@"
fi

EXP_NAME="${EXP_NAME:-super-vl-35-mixed-teachers-pipeclean-24n4g}"
RESULTS_ROOT="${RESULTS_ROOT:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sauramishra/supervl35-pipeclean}"
EXP_DIR="${RESULTS_ROOT}/${EXP_NAME}"
WANDB_PROJECT="${WANDB_PROJECT:-saumishr-supervl35-pipeclean}"

export CONFIG="${CONFIG:-examples/configs/recipes/vlm/super_vl_35_mixed_teachers_pipeclean_24n4g.yaml}"

mkdir -p "${EXP_DIR}/logs/nemo_gym" "${EXP_DIR}/checkpoints" "${EXP_DIR}/image_tool_outputs"

echo "================================================================"
echo "  Super VL 3.5 mixed teachers - 24-node SingleController pipeclean"
echo "================================================================"
echo "  Config   : ${CONFIG}"
echo "  Nodes    : 24 (16 training TP2/CP2/EP16 + 8 generation TP4/EP4)"
echo "  Budget   : 5 steps, 32 prompts x 16 generations, GBS 512"
echo "  Results  : ${EXP_DIR}"
echo "  W&B      : ${WANDB_PROJECT}/${EXP_NAME}"
echo "================================================================"

exec bash "${BASE_LAUNCHER}" \
  "++checkpointing.checkpoint_dir=${EXP_DIR}/checkpoints" \
  "++logger.log_dir=${EXP_DIR}/logs" \
  "++env.nemo_gym.nemo_gym_log_dir=${EXP_DIR}/logs/nemo_gym" \
  "++env.nemo_gym.image_tools_simple_agent.responses_api_agents.image_tools_agent.crop_dir=${EXP_DIR}/image_tool_outputs" \
  "++logger.wandb.project=${WANDB_PROJECT}" \
  "++logger.wandb.name=${EXP_NAME}" \
  "++logger.wandb.id=${EXP_NAME}" \
  "$@"
