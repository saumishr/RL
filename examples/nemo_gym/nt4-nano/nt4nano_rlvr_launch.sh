#!/bin/bash
# Launch NT4 (Nemotron-4) Nano text RLVR on 86 nodes with the external judge
# fleet.
#
# This is a profile, not a launcher. Every piece of Slurm and external-pool
# machinery it needs already exists in lightning35_launch.sh and is fully
# parameterised by environment, so this script sets the NT4-specific values and
# execs that -- the same relationship launch_rlvr_v43.sh has to the pipeline's
# tools/launch.sh in the Super 3.5 production journey. Forking 900 lines to
# change a dozen variables would guarantee the copy rots.
#
# Node shape (86 = 64 Ray + 22 external), unchanged from Lightning:
#   32 training / 32 policy vLLM / 2 Gym safety / 16 GenRM / 4 NL2Bash
#
# The judge checkpoints are the ones the Super 3.5 production text RLVR stage
# uses. They are read-only to us and are referenced in place.
#
# Required from the caller (no sensible default exists):
#   EXP_NAME RESULTS_DIR SLURM_PARTITION SLURM_ACCOUNT CONTAINER
#   PERSISTENT_CACHE SANDBOX_CONTAINER
#
# Usage:
#   EXP_NAME=nt4nano-rlvr-86n \
#   RESULTS_DIR=/lustre/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/$USER/rlvr-86n \
#   SLURM_PARTITION=... SLURM_ACCOUNT=... \
#   CONTAINER=... PERSISTENT_CACHE=... SANDBOX_CONTAINER=... \
#   bash examples/nemo_gym/nt4-nano/nt4nano_rlvr_launch.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(realpath "${SCRIPT_DIR}/../../..")"

export CONFIG_PATH="examples/nemo_gym/nt4-nano/rlvr.yaml"

# Run from the live checkout rather than an rsync snapshot. The Megatron-Bridge
# pin this branch carries has dangling symlinks under its nested Megatron-LM
# test tree, which rsync reports as errors and which abort the snapshot. The
# NM4 pipeclean launcher has always mounted the checkout live for the same
# reason. Set USE_SNAPSHOT=1 once that pin is fixed if submission-time
# immutability is wanted.
export USE_SNAPSHOT="${USE_SNAPSHOT:-0}"

# Lightning's launcher passes its own logger overrides on the command line, so
# the project set in rlvr.yaml would lose and these runs would file themselves
# under Lightning's. Name it here instead.
export WANDB_PROJ="${WANDB_PROJ:-nemo-rl-nt4nano}"

# -----------------------------------------------------------------------------
# Policy
# -----------------------------------------------------------------------------
# The HF export the math pipeclean trained from. The Megatron->HF->vLLM bridge
# is validated against this specific export, so do not point it elsewhere
# without re-checking the sampling importance ratio on step 1.
export MODEL_PATH="${MODEL_PATH:-/lustre/fsw/portfolios/nemotron/users/tbarnatan/models/nm4/nemotron4-nano-wsm-vlm}"

# -----------------------------------------------------------------------------
# Node allocation
# -----------------------------------------------------------------------------
export GPUS_PER_NODE=4
export NUM_TRAIN_NODES="${NUM_TRAIN_NODES:-32}"
export NUM_GEN_NODES="${NUM_GEN_NODES:-32}"
export NUM_GYM_NODES="${NUM_GYM_NODES:-2}"
# 2, not the 8 the earlier NM4 runs used. The Ray allocation is train + gen +
# gym = 66 nodes, and the launcher requires the segment size to divide it; 8
# does not. This is a constraint of the shape, not a preference.
export SEGMENT_SIZE="${SEGMENT_SIZE:-2}"
export EXTERNAL_VLLM_SEGMENT_SIZE="${EXTERNAL_VLLM_SEGMENT_SIZE:-2}"

# -----------------------------------------------------------------------------
# Judge fleet -- Super 3.5 production checkpoints
# -----------------------------------------------------------------------------
export GENRM_MODEL="${GENRM_MODEL:-/lustre/fsw/portfolios/llmservice/users/jiaqiz/models/ultra_genrm_v2_step1040}"
export GENRM_REPLICAS="${GENRM_REPLICAS:-8}"
export GENRM_TENSOR_PARALLEL_SIZE=8
export GENRM_MAX_MODEL_LEN="${GENRM_MAX_MODEL_LEN:-131072}"

export NL2BASH_JUDGE_MODEL="${NL2BASH_JUDGE_MODEL:-/lustre/fs1/portfolios/llmservice/projects/llmservice_modelalignment_ppo/users/pjin/checkpoints/qwen3-235b-a22b-instruct-2507-fp8-hf}"
export NL2BASH_REPLICAS="${NL2BASH_REPLICAS:-4}"
export NL2BASH_TENSOR_PARALLEL_SIZE=4

export SAFETY_JUDGE_MODEL="${SAFETY_JUDGE_MODEL:-/lustre/fs1/portfolios/llmservice/projects/llmservice_modelalignment_ppo/users/pjin/checkpoints/nemotron-content-safety-reasoning-4b-hf}"

export ENABLE_EXTERNAL_VLLM=1

# Lightning's pool definitions assume 8-GPU nodes and a pre-Blackwell judge
# fleet. Both assumptions are wrong here, so correct them after the fact rather
# than forking the launcher.
export EXTERNAL_VLLM_POOL_OVERRIDES="${SCRIPT_DIR}/pool_overrides.sh"

# Do not set GENRM_CONTAINER to a standalone vllm-openai image, however much
# the Super 3.5 production script looks like a template for it. That script
# orchestrates through pipeline/tools/launch.sh, not this path. Here the pools
# run serve_vllm_on_ray.py, which imports
# nemo_rl.models.generation.vllm.patches, and their srun mounts only
# EXTERNAL_VLLM_SHARED_ROOT -- no repo overlay -- so nemo_rl has to be baked
# into the image. Lightning's default of CONTAINER is therefore the right one
# and is left alone: probed on this cluster, that image carries nemo_rl at
# /opt/nemo-rl, vLLM 0.29.0 and the /opt/ray_venvs interpreter that
# GENRM_VLLM_PYTHON defaults to. Overriding either one breaks both pools.

# The sandbox master builds one nginx config listing every peer node as an
# upstream, and nginx resolves all of them at config-test time. A single
# transient lookup failure anywhere in that list aborts the test, so
# start-with-nginx.sh exits, ray.sub sees a background srun die and tears the
# whole job down -- which killed an 86-node allocation 2 minutes in after one
# node failed to resolve. Retry instead: the script is a server that never
# returns on success, so any return is a failure worth another attempt, and a
# second config test re-resolves the name. Clear the first attempt's uwsgi
# workers first or the retry lands on ports they still hold; the pattern below
# matches exactly how the script spawns them. Three attempts fit inside the
# 300s readiness deadline ray.sub gives each sandbox task.
export SANDBOX_COMMAND="${SANDBOX_COMMAND:-_try=0; while :; do _try=\$((_try+1)); /start-with-nginx.sh; _rc=\$?; if [ \"\$_try\" -ge 3 ]; then echo \"[FATAL] sandbox exited rc=\$_rc on attempt \$_try; giving up\" >&2; exit \"\$_rc\"; fi; echo \"[WARN] sandbox exited rc=\$_rc on attempt \$_try; clearing workers and retrying\" >&2; pkill -f \"uwsgi --ini /tmp/worker\" || true; sleep 5; done}"

# NM4's MTP head exists (mtp_num_layers 2) but speculative decoding has never
# been exercised on it through the vLLM fork. Off for a first run; it is a
# throughput knob, not a correctness one.
export ENABLE_MTP_INFERENCE="${ENABLE_MTP_INFERENCE:-0}"

# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
# The Super 3.5 production RLVR set is built for 64k and NM4 caps at 8192, so
# it must be filtered before use -- and it cannot be filtered from config, see
# the data block in rlvr.yaml. Point TRAIN_PATH at the filtered output of:
#
#   python examples/nemo_gym/nt4-nano/filter_rlvr_by_length.py \
#     --input  <...>/rl-v43-broad-falcon-baf8_noncommercial-resume.train.len64k.jsonl \
#     --output <...>/rl-v43.nt4nano.len4k.jsonl \
#     --tokenizer <NT4 processor dir> --max-prompt-tokens 4096
export TRAIN_PATH="${TRAIN_PATH:?TRAIN_PATH is required (filtered RLVR jsonl; see filter_rlvr_by_length.py)}"
export VAL_PATH="${VAL_PATH:-${TRAIN_PATH}}"

# -----------------------------------------------------------------------------
# Mounts
# -----------------------------------------------------------------------------
# ray.sub mounts nothing but MOUNTS, and Lightning's launcher builds that from a
# fixed list that includes its own recipe directory but not ours. Everything
# this profile reaches for therefore has to be declared, including the model
# directories -- none of /lustre is visible by default.
#
# Checkpoints are mounted read-only at their own paths so the config can name
# them literally. They belong to other users and are read-only to us anyway;
# the flag makes that explicit rather than incidental.
_nt4_mounts=(
  "${SCRIPT_DIR}:/opt/nemo-rl/examples/nemo_gym/nt4-nano"
  # The checkout at its own path, not just overlaid onto /opt/nemo-rl. The
  # external pools' load balancer runs with --container-workdir=$SLURM_SUBMIT_DIR
  # -- the directory sbatch was invoked from, which is this repo -- so pyxis
  # fails task_init with "couldn't chdir" unless that path resolves inside the
  # container. The NM4 pipeclean launcher self-mounts for the same reason.
  "${PROJECT_ROOT}:${PROJECT_ROOT}"
  "$(dirname "${MODEL_PATH}"):$(dirname "${MODEL_PATH}"):ro"
  "$(dirname "${GENRM_MODEL}"):$(dirname "${GENRM_MODEL}"):ro"
  "$(dirname "${NL2BASH_JUDGE_MODEL}"):$(dirname "${NL2BASH_JUDGE_MODEL}"):ro"
  "$(dirname "${TRAIN_PATH}"):$(dirname "${TRAIN_PATH}")"
)
# The NL2Bash and safety checkpoints usually share a parent; only add it once,
# since a duplicate container-mounts entry is an error rather than a no-op.
if [[ "$(dirname "${SAFETY_JUDGE_MODEL}")" != "$(dirname "${NL2BASH_JUDGE_MODEL}")" ]]; then
  _nt4_mounts+=("$(dirname "${SAFETY_JUDGE_MODEL}"):$(dirname "${SAFETY_JUDGE_MODEL}"):ro")
fi

_joined="$(
  IFS=,
  printf '%s' "${_nt4_mounts[*]}"
)"
export EXTRA_MOUNTS="${EXTRA_MOUNTS:+${EXTRA_MOUNTS},}${_joined}"

# -----------------------------------------------------------------------------
# Stage the external vLLM pool tooling onto shared storage
# -----------------------------------------------------------------------------
# The judge pools run in a second Slurm hetgroup, outside the Ray cluster, so
# they read this tooling straight off the shared filesystem rather than through
# a container mount. A repo checked out under /home is not reachable that way,
# so copy it next to the run's own results.
if [[ -z "${EXTERNAL_VLLM_TOOLS_DIR_HOST:-}" ]]; then
  : "${RESULTS_DIR:?RESULTS_DIR is required}"
  EXTERNAL_VLLM_TOOLS_DIR_HOST="${RESULTS_DIR}/external_gym_vllm"
  mkdir -p "${EXTERNAL_VLLM_TOOLS_DIR_HOST}"
  cp -r "${PROJECT_ROOT}/tools/external_gym_vllm/." "${EXTERNAL_VLLM_TOOLS_DIR_HOST}/"
  export EXTERNAL_VLLM_TOOLS_DIR_HOST
  echo "  Staged external vLLM tooling: ${EXTERNAL_VLLM_TOOLS_DIR_HOST}"
fi

# -----------------------------------------------------------------------------
# Hand off to Lightning's launcher, which owns the Slurm hetjob and the
# external vLLM pools.
# -----------------------------------------------------------------------------
exec bash "${PROJECT_ROOT}/examples/nemo_gym/nemotron-3.5-lightning/lightning35_launch.sh" "$@"
