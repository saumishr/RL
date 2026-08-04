#!/bin/bash
# =============================================================================
# ultra_swe_streaming.sh — Nemotron 3 Ultra SWE, streaming vs legacy.
#
#     bash ultra_swe_streaming.sh              # DRY_RUN=1: print the command
#     DRY_RUN=0 bash ultra_swe_streaming.sh    # submit the streaming arm
#     DRY_RUN=0 FLOW=legacy bash ultra_swe_streaming.sh    # submit the control
#
# SHAPE — 48 nodes / 192 GB200 GPUs, mirroring the 48-node Ultra SWE smoke:
#   train 32 nodes (128 GPUs)  TP8 · CP16 · EP32, 65k context
#   gen   16 nodes ( 64 GPUs)  vLLM
#   gym    0 nodes             SWE rewards are code execution in apptainer .sif
#
# SWE is the workload that makes the long tail visible. Two statistically
# identical 32-rollout batches measured on this exact shape finished at 18:05 and
# 40:25, so a trainer that waits for a whole batch waits on the slower one every
# step — the observed cost was 74% of every step spent in exposed_generation.
#
# FLOW selects the arm:
#
#   FLOW=streaming (default)  SingleController + TransferQueue, prompt-group
#                             streaming (examples/run_grpo_single_controller.py)
#   FLOW=legacy               the stage's own batch-granularity async flow with
#                             the in-memory ReplayBuffer over Ray's object store
#                             (examples/nemo_gym/run_grpo_nemo_gym.py)
#
# Both arms share this file so the only differences are the entrypoint, the
# config and the streaming knobs. Model, data, container, node split, step
# sizing, KL penalty, checkpointing and validation are identical, which is what
# makes the wall-clock numbers comparable. Both emit `total_step_time` and
# `exposed_generation` (the trainer waiting on trajectories), so the headline
# metric is exposed_generation / total_step_time.
# =============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

_DRY_RUN_IN="${DRY_RUN:-}"
# Captured before the `set -a` block below assigns them unconditionally.
# WALLTIME under two hours makes ultra_launch.sh select the `short` QoS, which
# outranks `normal` enough to skip most of the queue; EXP_SUFFIX keeps such a run
# from writing over the results and W&B run of a full-length arm.
_WALLTIME_IN="${WALLTIME:-}"
_EXP_SUFFIX_IN="${EXP_SUFFIX:-}"

FLOW="${FLOW:-streaming}"
if [[ "${FLOW}" != "streaming" && "${FLOW}" != "legacy" ]]; then
  echo "ERROR: FLOW must be 'streaming' or 'legacy', got '${FLOW}'." >&2
  exit 1
fi

set -a

# --- Secrets (optional) ------------------------------------------------------
NRL_SECRETS_FILE="${NRL_SECRETS_FILE:-${HOME}/.nrl_secrets.sh}"
if [[ -r "${NRL_SECRETS_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${NRL_SECRETS_FILE}"
  echo "[SECRETS] sourced ${NRL_SECRETS_FILE} (wandb=${WANDB_API_KEY:+set})"
else
  # Not fatal: ultra_launch.sh decides W&B from WANDB_API_KEY, which may already
  # be exported in the environment. It disables W&B only when that is unset.
  echo "[INFO] no secrets file at ${NRL_SECRETS_FILE} (wandb=${WANDB_API_KEY:+set})"
fi

# --- Container ---------------------------------------------------------------
CONTAINER=/lustre/fsw/portfolios/coreai/users/yifuw/enroot-images/gitlab-master.nvidia.com/yifuw/images/nemo-rl:main_ultra_recipes_prebaked_venvs_nccl2304_20260726_b.squashfs
USE_CUSTOM_VLLM=0
ENABLE_MTP_INFERENCE=0

# No mooncake settings on either arm. The test arm's data plane runs the TQ
# "simple" backend, which keeps sample bytes in Ray actors, so the delta being
# measured here is the streaming control plane rather than the transport.
# Moving to mooncake_cpu/RDMA is a separate change and a separate measurement.

# --- Model and data ----------------------------------------------------------
# The 65k-context Ultra checkpoint the 48-node SWE smoke is built around.
MODEL_PATH=/lustre/fsw/portfolios/llmservice/users/jiaqiz/results/ultra-v3-pipeclean/pipeclean-ultra-rl-prod_ultra_stage2sft300_fixlc_tp8_cp8_ep64_pp1_gpp16_pps512_gbs8192-20260417-jiaqi-resumestep128-65k/eval/step_152/hf

# SWE agent tasks in nemo-gym format (responses_create_params / agent_ref /
# uuid, agent swe_agents_train). Unlike the runs this comparison is measured
# against, train and validation are genuinely disjoint files here.
TRAIN_PATH=/lustre/fsw/portfolios/llmservice/users/zhiyul/RL/ultra_data/swe.train.jsonl
VAL_PATH=/lustre/fsw/portfolios/llmservice/users/zhiyul/RL/ultra_data/swe.val.jsonl

# Apptainer images for the SWE-Bench / SWE-Gym instances the reward executes in.
SIF_DIR=/lustre/fsw/portfolios/llmservice/users/sdevare/images

# nemo-skills sandbox by a path we can read.
SANDBOX_CONTAINER=/lustre/fs1/portfolios/llmservice/projects/llmservice_nemo_reasoning/users/igitman/images/nemo-skills-sandbox-dc43f3e.sqsh

# --- Paths we own ------------------------------------------------------------
CODE_DIR="${HERE}"
NRL_USER_LUSTRE=/lustre/fsw/portfolios/llmservice/users/sauramishra
WORKSPACE_DIR="${NRL_USER_LUSTRE}/ultra-streaming/workspace"
HF_HOME="${NRL_USER_LUSTRE}/hf_cache"
PERSISTENT_CACHE="${NRL_USER_LUSTRE}/persistent_cache"
# Holds a symlink to an already-converted copy of the 550B torch_dist
# checkpoint, so neither arm pays the 1.1 TB HF->Megatron import at startup.
NRL_MEGATRON_CHECKPOINT_DIR="${PERSISTENT_CACHE}/megatron_ckpt_cache"

# CODE_DIR is on NFS home, so it needs an explicit bind mount for the
# ${CODE_DIR}-relative entrypoint and config to resolve inside the container.
EXTRA_MOUNTS="/lustre:/lustre,${CODE_DIR}:${CODE_DIR}"

# --- Cluster -----------------------------------------------------------------
SLURM_PARTITION=batch
SLURM_ACCOUNT=nemotron_sw_post
SLURM_QOS=
GPUS_PER_NODE=4
NUM_TRAIN_NODES=32
NUM_GEN_NODES=16
NUM_GYM_NODES=0
SEGMENT_SIZE=16
# Double the 1:59 window the reference runs had. At the step sizing below the
# streaming arm should close ~10 steps; the control arm's first step alone cost
# 57 minutes, so the extra headroom is what lets it produce more than one.
WALLTIME=3:59:00
[ -n "${_WALLTIME_IN}" ] && WALLTIME="${_WALLTIME_IN}"

# --- Run ---------------------------------------------------------------------
if [[ "${FLOW}" == "streaming" ]]; then
  EXP_NAME=ultra-swe-sc-streaming
  NRL_ENTRYPOINT="${CODE_DIR}/examples/run_grpo_single_controller.py"
  CONFIG_PATH="${CODE_DIR}/examples/configs/ultra/tiny_swe_teacher_sc_streaming.yaml"
else
  EXP_NAME=ultra-swe-legacy-batch
  NRL_ENTRYPOINT="${CODE_DIR}/examples/nemo_gym/run_grpo_nemo_gym.py"
  CONFIG_PATH="${CODE_DIR}/examples/configs/ultra/tiny_swe_teacher.yaml"
fi
EXP_NAME="${EXP_NAME}${_EXP_SUFFIX_IN}"
RESULTS_DIR="${WORKSPACE_DIR}/results/${EXP_NAME}"
BASE_LOG_DIR="${WORKSPACE_DIR}/ray_logs/${EXP_NAME}"
WANDB_PROJ=ultra-streaming
USE_SNAPSHOT=0
DRY_RUN=1
[ -n "${_DRY_RUN_IN}" ] && DRY_RUN="${_DRY_RUN_IN}"
set +a

mkdir -p "${WORKSPACE_DIR}" "${HF_HOME}" "${PERSISTENT_CACHE}"

# Step sizing is the config's native shape (8 prompt groups x 16 generations =
# GBS 128), so only the horizon is overridden here and the streaming knobs are
# taken from the YAML unchanged.
#
# An earlier revision ran this at GBS 32 to match the reference runs exactly.
# That shape structurally caps the result: measured on this recipe, a step costs
# a GBS-independent 56.6s of weight sync plus 5.77s per sequence, so GBS 32
# leaves only ~241s of compute to hide generation behind while the observed
# rollout tail is ~800s. Even flawless streaming would stay wait-bound. At GBS
# 128 compute per step is ~795s, roughly equal to the tail, so the tail can be
# absorbed almost entirely — 6.3s/sequence against 32.6s measured at GBS 32.
#
# Raising GBS is memory-neutral here: the mesh is TP8 x CP16 = 128 GPUs exactly,
# so DP=1 and a larger GBS only adds gradient-accumulation micro-steps at the
# same train_micro_batch_size=1. (Raising the micro-batch instead would double
# tokens per CP rank from 4096 to 8192, which is not memory-neutral, and the
# measured 91 vs 334 tokens/sec/gpu for backward vs forward says the step is
# already compute-bound, so there is little there to win.)
SMOKE_OVERRIDES=(
  grpo.max_num_steps=11
)

if [[ "${FLOW}" == "streaming" ]]; then
  # Nothing to override: tiny_swe_teacher_sc_streaming.yaml already pins the
  # lag-4 least-starvation shape (windowed, max_staleness_versions=4,
  # min_groups_for_streaming_train=1, max_inflight=max_buffered=40).
  :
else
  # The SC arm cannot checkpoint or validate, so silence both here too;
  # otherwise the control pays save and validation time the test arm never sees
  # and the wall-clock comparison is meaningless. The KL penalty needs no
  # override: swe_teacher.yaml ships 0.0 and the streaming config pins the same
  # value, so neither arm pays a reference-logprob pass the other avoids.
  SMOKE_OVERRIDES+=(
    checkpointing.enabled=false
    grpo.val_period=0
  )
fi

bash "${HERE}/ultra_launch.sh" "${SMOKE_OVERRIDES[@]}" "$@"
