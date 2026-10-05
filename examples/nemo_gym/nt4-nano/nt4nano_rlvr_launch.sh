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
# Hand off to Lightning's launcher, which owns the Slurm hetjob and the
# external vLLM pools.
# -----------------------------------------------------------------------------
exec bash "${PROJECT_ROOT}/examples/nemo_gym/nemotron-3.5-lightning/lightning35_launch.sh" "$@"
