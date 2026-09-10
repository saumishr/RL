#!/bin/bash
# 66-node convergence run for the all-features arm: SingleController + ready_first
# at lag 1 + NVCF judges + disaggregated sandboxes + CPU RDMA + nccl_reshard refit.
#
# Sibling of submit_smoke_allfeatures.sh, which validated this exact stack at 8
# nodes twice (jobs 3408324 and 3415231, both 2/2 steps). The only differences
# here are scale, step count, and pool size -- deliberately, so a failure at 66
# nodes cannot be blamed on a configuration the smoke never ran.
#
# Two properties of this arm to have in mind while it runs:
#
#   No checkpointing. mooncake_cpu and data-plane checkpointing are mutually
#   exclusive, so checkpointing.enabled is false. There is nothing to resume
#   from: a failure at step 9 costs the whole run, and re-running with the same
#   EXP_NAME starts from scratch rather than resuming.
#
#   Fail-fast on the first exhausted prompt. ready_first stamps no target step,
#   so validation requires max_skipped_prompts=0 and
#   max_consecutive_dropped_prompts=0. One prompt exhausting its 5 infra or 2
#   data attempts ends the run. The reference in_order run allowed 32 of each,
#   so this arm is less fault-tolerant than the run it reproduces.
#
# Usage:  bash submit_full_allfeatures.sh [extra hydra overrides...]
#         DRY_RUN=1 bash submit_full_allfeatures.sh    # print, do not submit
set -euo pipefail

MINE=/lustre/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/sauramishra
# --- credentials ------------------------------------------------------------
# Our own file, not $HEM/creds.env. That one is unwritable (hemild, -rw-r-----)
# and its NVIDIA_API_KEY went dead on 2026-09-04 -- and because it is sourced
# after .bashrc, it silently clobbered the working key with the dead one. Ours
# deliberately leaves NVIDIA_API_KEY alone so the live key in ~/.bashrc survives.
# Override with CREDS_FILE=... to point somewhere else.
CREDS_FILE="${CREDS_FILE:-$MINE/creds.env}"
if [[ ! -r "$CREDS_FILE" ]]; then
  echo "ERROR: creds file not readable: $CREDS_FILE" >&2
  exit 1
fi
set -a
# shellcheck disable=SC1090
source "$CREDS_FILE"
set +a

# uv must read the image's baked cache; an inherited (or empty) UV_CACHE_DIR
# makes it re-download wheels the image already has.
unset UV_CACHE_DIR

# --- what to run ------------------------------------------------------------
export EXP_NAME="${EXP_NAME:-allfeatures-conv-$(date +%m%d-%H%M)}"
export CONTAINER="$MINE/containers/rl-gym.allfeatures-v2.sqsh"
export CONFIG_PATH=examples/nemo_gym/nemotron-3.5-nano/rlvr_sc_nvcf_disagg.yaml
export NRL_MAX_STEPS="${NRL_MAX_STEPS:-10}"

# --- cluster shape: 32 train + 32 gen + 2 gym = 66, no judge hetgroup --------
export SLURM_ACCOUNT=nemotron_sw_post
export SLURM_PARTITION=batch
export NUM_TRAIN_NODES=32
export NUM_GEN_NODES=32
export NUM_GYM_NODES=2
# Reference startup is ~12 min to first rollouts, and NVCF saturation can treble
# step time. 4h covers 10 steps with margin at 3x; without checkpointing, hitting
# the walltime loses everything, so do not trim this.
export WALLTIME="${WALLTIME:-04:00:00}"

# --- job-reaper exemption ----------------------------------------------------
# WITHOUT THIS THE RUN IS CANCELLED AT ~30 MINUTES, BEFORE ITS FIRST STEP.
# nano35_launch.sh forwards SLURM_COMMENT to sbatch --comment; an absent or
# malformed comment counts as NO exemption, silently, leaving an empty Comment
# field as the only trace.
#
# Job 3654874 (66 nodes) was already RUNNING and unexempted when this was
# noticed, and had to be rescued with `scontrol update JobId=... Comment=...`
# eight minutes before the threshold. Its sibling control arm lost job 3655075
# outright the same way at 30:50, with every judge healthy at the time, and the
# gymscale campaign lost 3577249, 3578472 and 3578802 at 30:32-30:43. This is
# the OccupiedIdleGPUsJobReaper on its default idle-GPU threshold, NOT
# preemption: cluster PreemptExemptTime is 04:05:00 and no preemption record
# appears in any of them.
#
# Why this arm legitimately idles GPUs. The 32 Megatron training ranks hold no
# work until the first optimizer step, which cannot happen until the first
# ready_first rollout cohort returns, behind a 66-node cold start: container
# setup, 32 vLLM generation engines, 32 Megatron ranks and the Gym rollout tier.
# Hosting the judges on NVCF removes their load time but not this wait. Past
# startup, TRAINER IDLE TIME IS THE MEASUREMENT -- this arm is compared against
# the Slurm-hosted-judge control arm on exactly that quantity, so the idleness
# is the experiment rather than a fault. Hence reason=benchmarking.
REAPER_EXEMPT_MINS="${REAPER_EXEMPT_MINS:-120}"
export SLURM_COMMENT="${SLURM_COMMENT:-{\"OccupiedIdleGPUsJobReaper\":{\"exemptIdleTimeMins\":\"${REAPER_EXEMPT_MINS}\",\"reason\":\"benchmarking\",\"description\":\"NVCF-hosted-judge arm of an NVCF-vs-Slurm judge-deployment comparison. Judges are served off-allocation through NVCF, matched GPU-for-GPU to the Slurm control arm (GenRM 6 replicas x TP8, Qwen3-235B 8 x TP2, safety 4 x TP1). Training GPUs are idle through the cold start because the 32 Megatron ranks cannot take an optimizer step until the first ready_first rollout cohort returns, behind a 66-node start: container setup, 32 vLLM generation engines, 32 Megatron ranks and the Gym rollout tier. Beyond startup, trainer idle time while the judge tier absorbs the rollout load IS the quantity being measured against the control arm, so the idleness is the experiment rather than a fault.\"}}}"

# Invalid JSON is silently treated as NO exemption, so validate here rather than
# discovering it as a cancel at ~30 min. The summary is printed BY the parser so
# it cannot drift from what was actually submitted.
_reaper=$(python3 -c "import json,os
c=json.loads(os.environ['SLURM_COMMENT'])['OccupiedIdleGPUsJobReaper']
assert int(c['exemptIdleTimeMins'])>0 and c['description'] and c['reason'] in {
  'model_loading','data_loading','interactive','benchmarking',
  'disproportionate_resource_requirement','inference_server','other'}, c
print(f\"{c['exemptIdleTimeMins']} min, reason={c['reason']}, \"
      f\"{len(c['description'])}-char description\")") \
  || { echo "[PREFLIGHT FAIL] SLURM_COMMENT is not a valid reaper exemption" >&2; exit 1; }
echo "[PREFLIGHT] reaper exemption valid: ${_reaper}"

# --- assets (md5/byte-verified against the reference study) ------------------
export MODEL_PATH=/lustre/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/kajalj/nvcf-poc-data/policy
export TRAIN_PATH=/lustre/fsw/portfolios/nemotron/users/arnavk/nano_35_rlvr/train_data.jsonl
export VAL_PATH="$TRAIN_PATH"

# --- output and caches ------------------------------------------------------
export RESULTS_DIR="$MINE/runs/$EXP_NAME"
export PERSISTENT_CACHE="$MINE/cache"
export HF_HOME="$PERSISTENT_CACHE/hf_home"

# --- mounts -----------------------------------------------------------------
# /lustre and /scratch: ray.sub's shared-FS canary lives under LOG_DIR, and every
# head and worker exits 1 if it cannot see it. Job 3415078 died at 112s this way.
# pyproject.toml and uv.lock: --container-save does not persist bind mounts, so
# the image's venv matches our lock but its lock file does not. Mounting ours
# keeps `uv run --frozen` consistent with what the venv actually contains.
export EXTRA_MOUNTS="/lustre:/lustre,/scratch:/scratch,$MINE/rlvr-allfeatures/pyproject.toml:/opt/nemo-rl/pyproject.toml,$MINE/rlvr-allfeatures/uv.lock:/opt/nemo-rl/uv.lock"

# --- sandbox pool -----------------------------------------------------------
# ns_tools warms every slot eagerly at server boot (its lifespan awaits
# SandboxPool.start()), independent of the blend routing zero rows to it. Against
# a cold pool that means 256 failed claims at startup and a heal loop retrying at
# 4 creates/s for the run's duration. Non-fatal -- warmup failures only warn --
# but it is churn against a shared control plane. Warm the pool first if you can:
#   kubectl --context cell-2 -n opensandbox patch pools.sandbox.opensandbox.io \
#     ns-tools-warm --type merge -p '{"spec":{"capacitySpec":{"poolMin":512}}}'
export NS_SANDBOX_POOL_SIZE="${NS_SANDBOX_POOL_SIZE:-256}"
export NS_SANDBOX_IMAGE="${NS_SANDBOX_IMAGE:-942195279341.dkr.ecr.us-east-2.amazonaws.com/nemo-skills/sandbox@sha256:b6731ddf387fedb07a818861dfa9d230ff31b20ccf748e3b654b39f3b1c23c9a}"

# --- W&B --------------------------------------------------------------------
export WANDB_ENABLED=True
export WANDB_PROJ="${WANDB_PROJ:-nemotron-3.5-nano}"

cd "$MINE/rlvr-allfeatures"

# No min_step_batch_fraction override here, unlike the smoke: 0.75 existed only
# because 8 prompts per step cannot satisfy the fragment's 0.9 floor. At 512
# prompts the configured 0.9 is the intended value.
exec bash examples/nemo_gym/nemotron-3.5-nano/launch_rlvr_sc_nvcf_disagg.sh "$@"
