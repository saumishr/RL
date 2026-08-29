#!/bin/bash
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

set -euo pipefail

# =============================================================================
# launch_rlvr_sc_local_disagg.sh
#
# Submits the all-features arm of the Nemotron 3.5 Nano Amplified Dolphin RLVR
# study with the judges served ON the allocation: SingleController with
# streaming train, the ready_first sampler at a staleness window of 1, ns_tools
# sandboxes on a remote OpenSandbox pool, rollouts over CPU RDMA, and
# nccl_reshard refit.
#
# The sibling launch_rlvr_sc_nvcf_disagg.sh is this arm with the judges hosted
# on NVCF. Judge hosting is the only difference, in the config and here.
#
# Costs 20 nodes more than that arm (86 vs 66) and buys two things: reward
# scores from the same GenRM checkpoint the convergence baseline used, and no
# exposure to the shared NVCF GenRM's 504s under load. Take this arm when the
# reward curve has to be comparable with the baseline, or when NVCF capacity is
# contended; take the NVCF arm when nodes are the scarce resource.
#
# Usage:
#
#   set -a; source /path/to/creds.env; set +a   # see "Credentials" below
#   EXP_NAME=nano35-allfeatures-local \
#   MODEL_PATH=... TRAIN_PATH=... VAL_PATH=... \
#   CONTAINER=... PERSISTENT_CACHE=... \
#   SLURM_ACCOUNT=... SLURM_PARTITION=batch \
#   GENRM_MODEL=... GENRM_REASONING_PARSER=... \
#   NL2BASH_JUDGE_MODEL=... SAFETY_JUDGE_MODEL=... \
#   bash examples/nemo_gym/nemotron-3.5-nano/launch_rlvr_sc_local_disagg.sh
#
# The four judge variables are the ones rlvr.yaml has always taken and
# nano35_launch.sh validates; they are what this arm has instead of an NVCF
# key. Hydra overrides are forwarded verbatim, so a cheap smoke run is:
#
#   EXP_NAME=allfeatures-local-smoke NUM_TRAIN_NODES=4 NUM_GEN_NODES=2 \
#   NUM_GYM_NODES=6 NUM_EXTERNAL_SERVICE_NODES=16 NS_SANDBOX_POOL_SIZE=8 \
#   NRL_MAX_STEPS=2 WALLTIME=02:00:00 \
#   bash .../launch_rlvr_sc_local_disagg.sh \
#     grpo.num_prompts_per_step=8 policy.train_global_batch_size=128 \
#     async_rl.min_groups_for_streaming_train=2 async_rl.max_inflight_prompts=16 \
#     async_rl.max_buffered_rollouts=16
#
# Note the last three: at a reduced prompt count the inherited 1024-slot buffers
# are far past what admission can reach, and min_groups_for_streaming_train must
# come down or a step never accumulates enough groups to train on. Note also
# that the judges do not shrink with the training nodes -- they are sized by
# their own parallelism, so a smoke run still pays for all 22 judge nodes.
#
# Credentials (export before running; read inside the job, not baked into the
# submission):
#
#   OPENSANDBOX_BASE_URL  Sandbox control plane, read by Gym's ns_tools.yaml.
#   OPENSANDBOX_API_KEY   Its key, likewise.
#   NS_SANDBOX_IMAGE      Sandbox image. Required even with a pool_ref: pods
#                         claimed from the pool template use it, and selecting
#                         the sandbox_pool backend with an empty image is a hard
#                         startup error rather than a fallback.
#   WANDB_API_KEY         Optional; absent disables W&B.
#   HF_TOKEN              Optional.
#
# No NVIDIA_API_KEY: nothing on this arm calls a hosted endpoint.
#
# Pool contract (nothing here touches the pool -- run these yourself):
#
#   Before submitting, raise the warm pool to cover the claim and confirm it is
#   warm, because a cold pool with pool_fallback=false fails slots rather than
#   creating pods on demand:
#
#     kubectl -n opensandbox patch pools.sandbox.opensandbox.io ${NS_SANDBOX_POOL_REF:-<pool>} \
#       --type merge -p '{"spec":{"capacitySpec":{"poolMin":<>=size>}}}'
#     kubectl -n opensandbox get pools.sandbox.opensandbox.io ${NS_SANDBOX_POOL_REF:-<pool>} \
#       -o jsonpath='{.status}{"\n"}'
#
#   The claim defaults to 768 here rather than the NVCF arm's 256, because the
#   claim scales with the ns_tools servers and this arm runs 6 gym nodes to that
#   arm's 2. The pool is shared, so size NS_SANDBOX_POOL_SIZE to the run.
#   Afterwards reap claims by run label and drop poolMin back:
#
#     kubectl -n opensandbox delete bsbx \
#       -l "nemo-gym.nvidia.com/run=<NEMO_GYM_RUN_ID>" --wait=false
#
# Optional knobs (all defaulted below): NS_SANDBOX_POOL_REF,
# NS_SANDBOX_POOL_SIZE, NS_SANDBOX_POOL_FALLBACK, NS_SANDBOX_TTL_S,
# NEMO_GYM_RUN_ID, plus everything nano35_launch.sh accepts.
# =============================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# -----------------------------------------------------------------------------
# Fail at submit time, not after the allocation is up
# -----------------------------------------------------------------------------
# A missing sandbox setting kills the ns_tools server once Gym starts, well
# after the nodes are held. nano35_launch.sh validates the judge checkpoints
# and the rest of the paths.
: "${OPENSANDBOX_BASE_URL:?OPENSANDBOX_BASE_URL is required for the remote sandbox pool}"
: "${OPENSANDBOX_API_KEY:?OPENSANDBOX_API_KEY is required for the remote sandbox pool}"
: "${NS_SANDBOX_IMAGE:?NS_SANDBOX_IMAGE is required: sandbox_pool rejects an empty image even when claiming from a pool}"

# -----------------------------------------------------------------------------
# Judges on the allocation
# -----------------------------------------------------------------------------
# nano35_launch.sh's rlvr defaults already describe this shape -- 32 train, 32
# gen, 6 gym, a 16-node GenRM hetgroup, and ray.sub wrapped by
# run_in_allocation.sh to raise the GenRM servers. Left implicit rather than
# restated so the two stay in step; HOSTED_JUDGES stays off.
export HOSTED_JUDGES=0

# -----------------------------------------------------------------------------
# Sandboxes off the allocation
# -----------------------------------------------------------------------------
# Gym's ns_tools.yaml reads all of this straight from the environment at the
# pinned commit, which is why none of it appears in the NeMo-RL config.
export NO_COLOCATED_SANDBOX=1
export NS_TOOLS_SANDBOX_TYPE=sandbox_pool
export NS_SANDBOX_POOL_REF="${NS_SANDBOX_POOL_REF:-ns-tools-warm}"
export NS_SANDBOX_POOL_SIZE="${NS_SANDBOX_POOL_SIZE:-768}"
# false: fail the slot instead of creating a pod on demand when the pool is
# short. A silent fallback would answer with cold pods and read as a sandbox
# latency regression rather than as an under-provisioned pool.
export NS_SANDBOX_POOL_FALLBACK="${NS_SANDBOX_POOL_FALLBACK:-false}"
export NS_SANDBOX_TTL_S="${NS_SANDBOX_TTL_S:-21600}"
# Lean verification goes to the same sandboxes over HTTP rather than a local
# server, so it does not need the sidecar this arm no longer starts.
export MATH_FORMAL_LEAN_BACKEND="${MATH_FORMAL_LEAN_BACKEND:-ns_http}"
# Labels every claimed pod, and is the only handle for reaping leaks afterwards.
export NEMO_GYM_RUN_ID="${NEMO_GYM_RUN_ID:-nano35-allfeatures-local-$(date +%m%d-%H%M%S)}"

# -----------------------------------------------------------------------------
# SingleController entrypoint and config
# -----------------------------------------------------------------------------
export CONFIG_PATH="${CONFIG_PATH:-examples/nemo_gym/nemotron-3.5-nano/rlvr_sc_local_disagg.yaml}"
export TRAIN_ENTRYPOINT="${TRAIN_ENTRYPOINT:-./examples/run_grpo_single_controller.py}"

echo "[allfeatures-local] judges: served on the allocation (GenRM hetgroup + in-Gym vLLM)"
echo "[allfeatures-local] sandboxes: pool=${NS_SANDBOX_POOL_REF} size=${NS_SANDBOX_POOL_SIZE} fallback=${NS_SANDBOX_POOL_FALLBACK}"
echo "[allfeatures-local] run id (sandbox claim label): ${NEMO_GYM_RUN_ID}"
echo "[allfeatures-local] confirm the pool is warm before this job starts claiming"

exec bash "${SCRIPT_DIR}/nano35_launch.sh" rlvr "$@"
