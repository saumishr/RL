#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

# =============================================================================
# Nemotron 3.5 Nano — 68-node SingleController convergence chain
#
# Submits one campaign as several same-named jobs so a step budget that outlives
# the 4 h wall on partition `batch` can finish across requeues.
#
# Usage:
#   bash examples/nemo_gym/nemotron-3.5-nano/nano35_dolphin_convergence_chain.sh
#   NUM_JOBS=2 bash .../nano35_dolphin_convergence_chain.sh      # stage the chain
#   DRY_RUN=1 NUM_JOBS=1 bash .../nano35_dolphin_convergence_chain.sh
#
# HOW THE CHAIN RUNS. ultra_launch.sh always submits with
# --dependency=singleton against a job name derived from EXP_NAME, so Slurm runs
# exactly one of these at a time and the rest wait. Each job resumes from
# CHECKPOINT_DIR, which is stable across jobs, so the order Slurm picks does not
# matter and no afterany chain is needed. Set SLURM_DEPENDENCY=afterany:<jobid>
# to also chain behind something outside this campaign.
#
# WHY THE STEP BUDGET IS A TOTAL. The SC train loop runs
# `while self._train_steps < grpo.max_num_steps` against the step count restored
# from the checkpoint, so NRL_MAX_STEPS is a whole-campaign target. Five jobs
# with NRL_MAX_STEPS=100 produce 100 steps, not 500: whichever job crosses 100
# saves and exits, and any queued behind it come up, find the budget met, and
# exit immediately.
#
# WHY THE PATHS ARE OVERRIDDEN. nano35_dolphin_launch.sh inherits the reference
# run's RESULTS_DIR, PERSISTENT_CACHE and HF_HOME, which live under another
# user's portfolio and are not group-writable. That was survivable while the SC
# recipe had checkpointing off; with saving on, CHECKPOINT_DIR sits under
# RESULTS_DIR and the first save would fail hours into the run.
#
# ONE W&B RUN. Every job passes the same logger.wandb.id with resume=allow, so
# the campaign is a single continuous curve rather than five overlapping ones.
# wandb.init receives the whole logger.wandb dict, so no code change is needed.
# Caveat: a job that dies after saving at step N but having trained past it
# re-trains those steps, and W&B keeps the first copy of a step's history rather
# than the replayed one.
# =============================================================================

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

NUM_JOBS="${NUM_JOBS:-5}"

# The campaign definition. SAMPLER and the lag are what the sweep varies; every
# job in one chain must agree on the sampler, because the replay-buffer restore
# is skipped when the checkpoint's sampler name differs from the running one and
# the run would come back cold instead of at its steady-state staleness.
export SAMPLER="${SAMPLER:-ready_first}"
export MAX_LOOKAHEAD_VERSIONS="${MAX_LOOKAHEAD_VERSIONS:-4}"
export NRL_MAX_STEPS="${NRL_MAX_STEPS:-100}"

# Retention headroom over the generation quota. At 1 the buffer equals the
# in-flight cap, so groups that finished generating but have not been selected
# hold permits from the same pool admission draws from. ready_first is a gated
# sampler -- admission is capped by version, not by the buffer -- so raising this
# raises retention only, which is the v1 shape (late_arrival_slack=2).
export BUFFER_RETENTION_MULTIPLIER="${BUFFER_RETENTION_MULTIPLIER:-2}"

export EXP_NAME="${EXP_NAME:-${USER}-nano35-sc-conv-${SAMPLER}-lag${MAX_LOOKAHEAD_VERSIONS}}"

_USER_SCRATCH="/lustre/fsw/portfolios/llmservice/users/${USER}"
export RESULTS_DIR="${RESULTS_DIR:-${_USER_SCRATCH}/runs/${EXP_NAME}}"
export PERSISTENT_CACHE="${PERSISTENT_CACHE:-${_USER_SCRATCH}/.cache/nano35-dolphin}"
export HF_HOME="${HF_HOME:-${_USER_SCRATCH}/hf_home}"

# Run the live worktree rather than a submission-time snapshot: the branch under
# test is the point of the campaign, and a mid-chain fix should reach the jobs
# still queued.
export USE_SNAPSHOT="${USE_SNAPSHOT:-0}"

# Judges inside the job rather than on a warm out-of-band pool. The pool would
# amortize GenRM's 470 GB load across the chain instead of paying ~10 min of it
# per job, but it would also have to stay healthy for the ~20 h the chain spans,
# and its death takes every job still queued behind it with it. A self-contained
# job also puts the reward model in its own provenance, which matters when the
# whole point is comparing convergence across arms.
export EXTERNAL_JUDGES="${EXTERNAL_JUDGES:-1}"

WANDB_RUN_ID="${WANDB_RUN_ID:-${EXP_NAME}}"

if (( NUM_JOBS < 1 )); then
  echo "NUM_JOBS must be >= 1, got ${NUM_JOBS}." >&2
  exit 1
fi

_CHECKPOINT_DIR="${RESULTS_DIR}/checkpoints"
_existing=0
if [[ -d "${_CHECKPOINT_DIR}" ]]; then
  _existing=$(find "${_CHECKPOINT_DIR}" -maxdepth 1 -name 'step_*' -type d 2>/dev/null | wc -l)
fi

echo "================================================================"
echo "  Nemotron 3.5 Nano — SC convergence chain"
echo "================================================================"
echo "  Experiment : ${EXP_NAME}"
echo "  Sampler    : ${SAMPLER} (lag ${MAX_LOOKAHEAD_VERSIONS})"
echo "  Budget     : ${NRL_MAX_STEPS} steps total, across ${NUM_JOBS} job(s)"
echo "  Buffer     : x${BUFFER_RETENTION_MULTIPLIER} retention over the in-flight quota"
echo "  Results    : ${RESULTS_DIR}"
echo "  Checkpoints: ${_CHECKPOINT_DIR} (${_existing} present — resumes from the latest)"
echo "  W&B run id : ${WANDB_RUN_ID} (resume=allow)"
echo "================================================================"
echo ""

# ultra_launch.sh derives RUN_DIR from `date +%Y%m%d-%H%M`, so submissions inside
# one minute would share a run directory and overwrite each other's
# provenance.txt. Wait out the minute instead of interleaving them.
_wait_for_new_run_minute() {
  local previous="$1"
  while [[ "$(date +%Y%m%d-%H%M)" == "${previous}" ]]; do
    sleep 5
  done
}

_overrides=(
  "++logger.wandb.id=${WANDB_RUN_ID}"
  "++logger.wandb.resume=allow"
)

# One pass, no chain: ultra_launch.sh exits non-zero after printing TRAIN_CMD,
# which under `set -e` would abort the loop before it could say why.
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[chain] DRY_RUN=1 — one pass through the launcher, nothing submitted."
  exec bash "${SCRIPT_DIR}/nano35_dolphin_launch_sc.sh" "${_overrides[@]}" "$@"
fi

_submitted=()
_previous_minute=""

for (( job = 1; job <= NUM_JOBS; job++ )); do
  if [[ -n "${_previous_minute}" ]]; then
    echo "[chain] waiting for a fresh run directory minute before job ${job}/${NUM_JOBS}..."
    _wait_for_new_run_minute "${_previous_minute}"
  fi
  _previous_minute=$(date +%Y%m%d-%H%M)

  echo "[chain] submitting job ${job}/${NUM_JOBS}"
  _output=$(bash "${SCRIPT_DIR}/nano35_dolphin_launch_sc.sh" \
    "${_overrides[@]}" "$@" 2>&1 | tee /dev/stderr)

  _job_id=$(printf '%s\n' "${_output}" \
    | grep -oE 'Submitted batch job [0-9]+' \
    | grep -oE '[0-9]+' \
    | tail -1)
  if [[ -z "${_job_id}" ]]; then
    echo "[chain] job ${job} did not report a Slurm job id; stopping so the rest" >&2
    echo "        of the chain is not submitted against an unknown state." >&2
    exit 1
  fi
  _submitted+=("${_job_id}")
done

echo ""
echo "================================================================"
echo "  Submitted ${#_submitted[@]} job(s): ${_submitted[*]}"
echo "  Only one runs at a time (--dependency=singleton on ${EXP_NAME})."
echo ""
echo "  Queue:   squeue -u ${USER} -n ${EXP_NAME}"
echo "  Driver:  tail -F ${RESULTS_DIR}/ray_logs/<jobid>-logs/ray-driver.log"
echo "  Saves:   ls -1 ${_CHECKPOINT_DIR}"
echo "================================================================"
