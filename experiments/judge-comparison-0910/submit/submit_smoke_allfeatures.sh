#!/bin/bash
# 8-node smoke for the all-features arm: SingleController + ready_first at lag 1
# + NVCF judges + disaggregated sandboxes + CPU RDMA + nccl_reshard refit.
#
# This exists as a file rather than an inline command so a re-run is byte-identical
# to the last one. Job 3408324 passed 2/2 steps with exactly this shape; the only
# thing that changed since is that the four launcher/config fixes it depended on
# are now committed (deec78b28b, 955b132c03) rather than sitting dirty in the
# tree, which is what this re-run is checking.
#
# Usage:  bash submit_smoke_allfeatures.sh [extra hydra overrides...]
#         DRY_RUN=1 bash submit_smoke_allfeatures.sh    # print, do not submit
set -euo pipefail

MINE=/lustre/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/sauramishra

# --- credentials ------------------------------------------------------------
# OPENSANDBOX_BASE_URL, OPENSANDBOX_API_KEY, WANDB_API_KEY, HF_TOKEN,
# NVIDIA_API_KEY. NVIDIA_API_KEY reaches hydra as ${oc.env:NVIDIA_API_KEY},
# resolved inside the job.
#
# Was $HEM/creds.env, which is now a trap: that file's NVIDIA_API_KEY went dead
# on 2026-09-04, and because `set -a` sourcing runs after .bashrc it does not
# merely fail to help -- it CLOBBERS a working key already in the environment.
# submit_full_allfeatures.sh moved off it for exactly this reason; this script
# predates that and would have taken the dead key with it. Same CREDS_FILE
# indirection as the full arm so the two cannot drift apart again.
CREDS_FILE="${CREDS_FILE:-$MINE/creds.env}"
if [[ ! -r "$CREDS_FILE" ]]; then
  echo "ERROR: creds file not readable: $CREDS_FILE" >&2
  exit 1
fi
set -a
# shellcheck disable=SC1090
source "$CREDS_FILE"
set +a

# uv must read the image's baked cache. An inherited UV_CACHE_DIR -- including an
# empty one, which uv rejects outright -- makes it re-download wheels the image
# already carries, and on an egress-restricted cluster that looks like a hang.
unset UV_CACHE_DIR

# --- what to run ------------------------------------------------------------
export EXP_NAME="${EXP_NAME:-allfeatures-v3-$(date +%m%d-%H%M)}"
export CONTAINER="$MINE/containers/rl-gym.allfeatures-v2.sqsh"
export CONFIG_PATH=examples/nemo_gym/nemotron-3.5-nano/rlvr_sc_nvcf_disagg_smoke.yaml

# --- cluster shape: 4 train + 2 gen + 2 gym = 8 -----------------------------
export SLURM_ACCOUNT=nemotron_sw_post
export SLURM_PARTITION=batch
export NUM_TRAIN_NODES=4
export NUM_GEN_NODES=2
export NUM_GYM_NODES=2
export WALLTIME="${WALLTIME:-02:00:00}"

# Reaper exemption. This script execs launch_rlvr_sc_nvcf_disagg.sh DIRECTLY and
# never sources submit_full_allfeatures.sh, so it inherits nothing from the full
# arm -- including the SLURM_COMMENT added there. Set it here or this smoke is
# unexempted.
#
# Lower stakes than the full arm, since 8 nodes start fast enough that job
# 3654067 reached both its steps inside the ~30 min default threshold. It is
# still 3 lines against losing an allocation to a cancel that leaves only an
# empty Comment field as evidence. 60 min covers the whole walltime's risk
# window at this size.
export REAPER_EXEMPT_MINS="${REAPER_EXEMPT_MINS:-60}"
export SLURM_COMMENT="${SLURM_COMMENT:-{\"OccupiedIdleGPUsJobReaper\":{\"exemptIdleTimeMins\":\"${REAPER_EXEMPT_MINS}\",\"reason\":\"model_loading\",\"description\":\"8-node smoke for the NVCF-hosted-judge arm. Training GPUs are idle through the cold start because the Megatron ranks cannot take an optimizer step until the first rollout cohort returns, behind container setup, the vLLM generation engines and the Gym rollout tier. Two steps at reduced batch, so the run ends shortly after the first step rather than idling further.\"}}}"

# --- assets (md5/byte-verified against the reference study) ------------------
export MODEL_PATH=/lustre/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/kajalj/nvcf-poc-data/policy
export TRAIN_PATH=/lustre/fsw/portfolios/nemotron/users/arnavk/nano_35_rlvr/train_data.jsonl
export VAL_PATH="$TRAIN_PATH"

# --- output and caches ------------------------------------------------------
# PERSISTENT_CACHE is warm from the earlier runs (vLLM compile, Triton, Inductor,
# flashinfer cubins). Keeping the same path is why startup is minutes rather than
# a recompile.
export RESULTS_DIR="$MINE/runs/$EXP_NAME"
export PERSISTENT_CACHE="$MINE/cache"
export HF_HOME="$PERSISTENT_CACHE/hf_home"

# --- mounts -----------------------------------------------------------------
# Four mounts, none optional:
#
# /lustre and /scratch, because the snapshot path only covers the code. ray.sub
# writes a .shared_fs_canary under LOG_DIR and every head and worker exits 1 if
# it cannot see it, so without these the job dies at ~110s with a message that
# reads like a filesystem fault rather than a missing mount. Job 3415078 died
# exactly this way when the mounts were dropped.
#
# pyproject.toml and uv.lock, because --container-save does not persist
# bind-mounted files: the v2 image's venv was synced against OUR lock, but the
# lock file baked into the image is still the base image's. Mounting ours keeps
# what `uv run --frozen` reads consistent with what the venv actually contains.
export EXTRA_MOUNTS="/lustre:/lustre,/scratch:/scratch,$MINE/rlvr-allfeatures/pyproject.toml:/opt/nemo-rl/pyproject.toml,$MINE/rlvr-allfeatures/uv.lock:/opt/nemo-rl/uv.lock"

# --- sandbox pool: claim 8, not the full run's 256 ---------------------------
export NS_SANDBOX_POOL_SIZE=8
export NS_SANDBOX_IMAGE="${NS_SANDBOX_IMAGE:-942195279341.dkr.ecr.us-east-2.amazonaws.com/nemo-skills/sandbox@sha256:b6731ddf387fedb07a818861dfa9d230ff31b20ccf748e3b654b39f3b1c23c9a}"

# --- W&B --------------------------------------------------------------------
export WANDB_ENABLED=True
export WANDB_PROJ="${WANDB_PROJ:-nemotron-3.5-nano}"

cd "$MINE/rlvr-allfeatures"

# min_step_batch_fraction=0.75 rather than the fragment's 0.9: at 8 prompts per
# step a single dropped prompt is 12.5% of the batch, so 0.9 cannot be met.
exec bash examples/nemo_gym/nemotron-3.5-nano/launch_rlvr_sc_nvcf_disagg.sh \
  async_rl.rollout_failure.min_step_batch_fraction=0.75 \
  "$@"
