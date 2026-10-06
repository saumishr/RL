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

# policy.tokenizer.name comes from the inherited VLM recipe as
# ${oc.env:NT4_NANO_PROCESSOR,<this path>}. Name it here so it is mounted
# below rather than inherited as a path the container cannot see: get_tokenizer
# runs before any cluster work, and when the directory is missing transformers
# falls back to treating it as a Hub repo id and dies on the leading slash.
export NT4_NANO_PROCESSOR="${NT4_NANO_PROCESSOR:-/lustre/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/sauramishra/pipeclean-v2/models/nt4-nano-processor}"

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

# Two independent defects have to be worked around here, and the sandbox is now
# load-bearing: ns_tools, code_gen and terminus_judge all call it, so it is no
# longer acceptable for it to merely start.
#
# First, in multi-node mode the sandbox master builds one nginx config listing
# every peer node as an upstream, and nginx resolves all of them when it
# validates that config. A single transient lookup failure aborts the test
# before the master reports ready, so start-with-nginx.sh exits non-zero, srun's
# --kill-on-bad-exit fires, and ray.sub tears the whole job down -- which killed
# an 86-node allocation 2 minutes in after one node failed to resolve. Forcing
# single-node mode keeps each node's upstreams on 127.0.0.1, so no peer name is
# ever resolved. Each node then has its own sandbox on 127.0.0.1:6000, which is
# exactly where ns_tools looks by default.
#
# Second, the image ships a pgrep that cannot run:
#   start-with-nginx.sh: line 697: /usr/bin/pgrep: cannot execute: required file
#   not found
# Line 697 is the monitoring loop's liveness probe, `if ! pgrep nginx`. A probe
# that cannot execute reads as "nginx is gone", so the loop calls cleanup, which
# logs "Shutting down workers and nginx..." and kills an nginx that was serving
# fine. In job 7744902 this happened on 64 of 66 nodes about three minutes after
# each reported ready. It was harmless only while nothing called port 6000.
# Putting a working pgrep ahead of /usr/bin on PATH keeps the probe honest; it
# reports liveness from /proc, which is the same question pgrep would ask.
#
# ray.sub splices this value inside a single-quoted bash -xc, so it must contain
# no single quotes.
#
# The probe must answer 0 or 1 and nothing else. Processes come and go while
# /proc is being walked, so a bare grep can exit 2 on a vanished entry even
# though nginx is up -- which would reintroduce the false negative. Piping the
# match list into `grep -q .` reports presence only, and discards read errors.
_nt4_pgrep_shim="mkdir -p /tmp/nrl-shim && { echo \"#!/bin/sh\"; echo \"grep -l nginx /proc/[0-9]*/comm 2>/dev/null | grep -q .\"; } > /tmp/nrl-shim/pgrep && chmod +x /tmp/nrl-shim/pgrep"
export SANDBOX_COMMAND="${SANDBOX_COMMAND:-${_nt4_pgrep_shim}; PATH=/tmp/nrl-shim:\$PATH SANDBOX_FORCE_SINGLE_NODE=1 /start-with-nginx.sh}"
unset _nt4_pgrep_shim

# Mounting Megatron-Bridge from this checkout (see the mounts below) also
# replaces that subtree's pyproject.toml and uv.lock, and the root project
# depends on it as an editable path. That makes uv consider the lock stale, so
# the driver's `uv run` tries to re-resolve the whole dependency graph, reaches
# a private GitLab index it has no credentials for, and fails -- which is how
# job 7752254 died seven minutes in, taking 86 nodes with it.
#
# Nothing needs resolving: the image already ships the environment, and the
# mount only swaps Python source behind editable finders that map
# megatron.core and megatron.bridge into that tree. The text pipeclean launcher
# passes `uv run --no-sync` for exactly this reason while overlaying even more
# of the repo. Lightning builds its own `uv run` without that flag, so set the
# environment equivalent here rather than forking its command construction.
#
# Worker venvs are unaffected: they are pre-materialized under /opt/ray_venvs,
# so create_local_venv returns early and never invokes uv at all.
export UV_NO_SYNC="${UV_NO_SYNC:-1}"

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

# An uninitialised submodule would mount an empty directory over the image's
# working bridge, which fails the same way as not mounting it at all but 66
# nodes later. The text pipeclean launcher preflights this same file.
_nt4_provider="${PROJECT_ROOT}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src/megatron/bridge/models/experimental_nm4_llava_provider.py"
if [[ ! -f "${_nt4_provider}" ]]; then
  echo "ERROR: NM4 bridge provider missing: ${_nt4_provider}" >&2
  echo "  Initialize it with: git submodule update --init --recursive \\" >&2
  echo "    3rdparty/Megatron-Bridge-workspace/Megatron-Bridge" >&2
  exit 1
fi
unset _nt4_provider

_nt4_mounts=(
  "${SCRIPT_DIR}:/opt/nemo-rl/examples/nemo_gym/nt4-nano"
  # The checkout at its own path, not just overlaid onto /opt/nemo-rl. The
  # external pools' load balancer runs with --container-workdir=$SLURM_SUBMIT_DIR
  # -- the directory sbatch was invoked from, which is this repo -- so pyxis
  # fails task_init with "couldn't chdir" unless that path resolves inside the
  # container. The NM4 pipeclean launcher self-mounts for the same reason.
  "${PROJECT_ROOT}:${PROJECT_ROOT}"
  # Megatron-Bridge has to come from this checkout, not the image. Two separate
  # things break without it, and both killed job 7746306 at the first policy
  # worker -- after the sandbox, the dataset, the config and all eight judge
  # pools had finally come up clean:
  #
  #   1. The NM4 provider this branch adds (experimental_nm4_llava_provider,
  #      imported by nemo_rl/models/megatron/nemotron4/__init__.py) does not
  #      exist in the image's bridge, which predates it.
  #   2. The image's nested Megatron-LM is a newer Mistral fork whose
  #      megatron/training/training.py imports mistral_adapter
  #      unconditionally, and that chain ends at a bare `import cv2`. NeMo-RL
  #      excludes OpenCV from the image on purpose (pyproject pins it to
  #      `sys_platform == 'never'` over FFmpeg codec royalties), so importing
  #      megatron.training there cannot work. This checkout's Megatron-LM has
  #      no such import, so the question never arises.
  #
  # One mount fixes both because Megatron-LM is nested inside Megatron-Bridge
  # at 3rdparty/Megatron-LM, which is where the image resolves `megatron` from.
  #
  # The text pipeclean got this by overlaying the whole repo at /opt/nemo-rl.
  # Lightning deliberately mounts piecewise instead and documents this exact
  # container path as an EXTRA_MOUNTS override, so narrow the substitution to
  # the one tree that actually differs rather than re-overlaying everything.
  "${PROJECT_ROOT}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge:/opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge"
  "$(dirname "${MODEL_PATH}"):$(dirname "${MODEL_PATH}"):ro"
  # The processor is its own export, so mount the directory itself rather than
  # its parent -- the parent also holds the dist checkpoint and the 128B model.
  "${NT4_NANO_PROCESSOR}:${NT4_NANO_PROCESSOR}:ro"
  "$(dirname "${GENRM_MODEL}"):$(dirname "${GENRM_MODEL}"):ro"
  "$(dirname "${NL2BASH_JUDGE_MODEL}"):$(dirname "${NL2BASH_JUDGE_MODEL}"):ro"
  "$(dirname "${TRAIN_PATH}"):$(dirname "${TRAIN_PATH}")"
)
# The NL2Bash and safety checkpoints usually share a parent; only add it once,
# since a duplicate container-mounts entry is an error rather than a no-op.
if [[ "$(dirname "${SAFETY_JUDGE_MODEL}")" != "$(dirname "${NL2BASH_JUDGE_MODEL}")" ]]; then
  _nt4_mounts+=("$(dirname "${SAFETY_JUDGE_MODEL}"):$(dirname "${SAFETY_JUDGE_MODEL}"):ro")
fi

# RESULTS_DIR holds ray.sub's log dir and the checkpoint tree, and both halves of
# the job read it from inside the container: the head and every worker gate on
# signal files there, starting with the .shared_fs_canary that proves the
# directory really is shared. Since none of /lustre is mounted by default, a
# RESULTS_DIR outside the paths above means all 66 containers fail that check
# the moment they start and ray.sub tears the job down. PERSISTENT_CACHE is the
# same story for the HF and FlashInfer caches.
for _nt4_shared in "${RESULTS_DIR:-}" "${PERSISTENT_CACHE:-}"; do
  [[ -n "${_nt4_shared}" ]] || continue
  for _nt4_existing in "${_nt4_mounts[@]}"; do
    [[ "${_nt4_existing%%:*}" == "${_nt4_shared}" ]] && continue 2
  done
  _nt4_mounts+=("${_nt4_shared}:${_nt4_shared}")
done
unset _nt4_shared _nt4_existing

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
