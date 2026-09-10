#!/bin/bash
# 82-node Slurm-judge control for the NVCF comparison.
#
# Sibling of submit_full_allfeatures.sh, which runs the identical stack with the
# three judges hosted on NVCF instead of on the allocation. Everything that
# could move a step time other than judge placement is held equal between the
# two: same container, same policy checkpoint, same data, same node counts for
# training and generation, same sampler and data plane, same colocated
# sandboxes, same step budget. The judges are the independent variable.
#
# MATCHING THE NVCF DEPLOYMENT
#
# Reported NVCF GB300 allocation, and what it implies per replica:
#
#   GenRM     48 GPU / 6 replicas -> TP8    Nemotron-3-Ultra-550B-A55B-GenRM
#   nl2bash   16 GPU / 8 replicas -> TP2    Qwen3-235B-A22B-Instruct-2507-FP8
#   safety     4 GPU / 4 replicas -> TP1    Nemotron-Content-Safety-Reasoning-4B
#                                           ------
#                                           68 GPU
#
# Both sides are GB300 (288 GB/GPU), so matching GPU count matches capacity --
# this cluster's nodes are nvl72*, and the external judge vLLM logs from the
# 86-node run name the device GB300 explicitly. Ignore the "GB200 192GB" comment
# above nl2bash_judge_model in rlvr.yaml; it is stale, and the same logs
# contradict it (176 GiB of KV per GPU at TP4 does not fit a 192 GB card).
#
# rlvr.yaml's own shape is DIFFERENT and is deliberately overridden below:
# nl2bash TP4xDP4 and safety TP4xDP2, sized for a standalone run. Two changes
# follow from matching NVCF instead.
#
#   nl2bash is no longer gym-resident at all, so its NVCF shape is expressed as
#   an external vLLM pool: 4 replicas at TP4, 16 GPUs, the same GPU count NVCF
#   gives it. NVCF splits those 16 GPUs as 8 replicas at TP2; this arm serves
#   them as 4 at TP4 because the pool wrapper gives each replica whole nodes and
#   this cluster has 4 GPUs per node. TP4 leaves MORE KV per replica than NVCF's
#   TP2 (weights replicated four times rather than eight), so the tier is not
#   handicapped by the difference -- but it is a difference, and it is the one
#   place this arm's judge topology departs from NVCF's.
#
#   WHY nl2bash MOVED OFF THE GYM TIER. Job 3656607 deadlocked in
#   nl2bash_judge_model's single-process Gym HTTP proxy: it accepted 594
#   concurrent judge requests and forwarded none of them, with its own vLLM
#   engines reporting "Running: 0, Waiting: 0" throughout. That proxy is a
#   local_vllm_model, the one server class whose num_workers must not fan out
#   because forking it forks the engine, so there is no in-Gym fix. An external
#   pool has a load balancer in front of N independent servers instead.
#
#   safety drops to TP1 and doubles to 4 replicas. It is a 4B model, so TP4 was
#   heavy over-provisioning and TP1 is simply the right shape. It STAYS
#   gym-resident: it was healthy at the concurrency that deadlocked nl2bash.
#
# GenRM keeps TP8, which rlvr.yaml already used, so only the replica count moves
# (8 -> 6). It is the one tier whose per-replica shape is unchanged.
#
# NODE ARITHMETIC at 4 GPUs/node
#
#   train                                        32
#   generation                                   32
#   gym    safety 4xTP1 = 4 GPU -> 1 node         2   (NUM_GYM_NODES; 1 would
#                                                --    fit, see below)
#          Ray subtotal                          66
#   external  GenRM   6 x TP8 = 48 GPU -> 12
#             NL2Bash 4 x TP4 = 16 GPU ->  4     16   (NUM_EXTERNAL_SERVICE_NODES,
#                                                --    derived from the pools)
#                                                82
#
# against the NVCF arm's 66. The 16-node delta IS the thing being measured: it is
# what hosting the judges elsewhere buys, and it has to be weighed against
# whatever step-time difference the two arms show. The total is unchanged from
# the 6-gym + 12-GenRM split this arm used before nl2bash moved out; four nodes
# crossed the hetgroup boundary rather than being added.
#
# GYM IS 2 NODES BUT ONLY NEEDS 1. Safety at TP1x4 wants 4 GPUs, which is one
# node -- but nano35_launch.sh requires the RAY node total to divide by
# SEGMENT_SIZE, and 32 + 32 + 1 = 65 is odd. Two gym nodes make it 66 and leave
# 4 GPUs for the CPU-only Gym servers to share. Raising SEGMENT_SIZE instead
# would change the training topology, which is precisely what this arm has to
# hold equal against the NVCF run, so the spare node is the cheaper concession.
#
# The 1-node figure itself only holds because data_parallel_size_local packs
# safety's four replicas onto one node. Left at rlvr.yaml's 1, each replica
# would take a whole node regardless of TP: 4 gym nodes using 4 of 16 GPUs. Do
# not drop that override.
#
# WHY THE OVERRIDES ARE ON THE COMMAND LINE. nano35_launch.sh appends ${*} last,
# so they beat the config, and they are readable in the printed TRAIN_CMD before
# anything is submitted. Setting vllm_serve_kwargs in YAML instead risks
# replacing the mapping wholesale and losing attention_backend,
# gpu_memory_utilization and max_model_len with it.
#
# No checkpointing on this path, same as the NVCF arm: a failure late in the run
# costs the whole run, and re-running the same EXP_NAME starts over.
#
# Usage:  bash submit_control_slurm_judges.sh [extra hydra overrides...]
#         DRY_RUN=1 bash submit_control_slurm_judges.sh    # print, do not submit
set -euo pipefail

MINE=/lustre/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/sauramishra
ASSETS=/lustre/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/akamehra

# --- credentials ------------------------------------------------------------
# Same indirection as submit_full_allfeatures.sh, and for the same reason: the
# shared creds.env carries an NVIDIA_API_KEY that went dead on 2026-09-04 and,
# being sourced after .bashrc, silently clobbers the working one.
CREDS_FILE="${CREDS_FILE:-$MINE/creds.env}"
if [[ ! -r "$CREDS_FILE" ]]; then
  echo "ERROR: creds file not readable: $CREDS_FILE" >&2
  exit 1
fi
set -a
# shellcheck disable=SC1090
source "$CREDS_FILE"
set +a

# uv must read the image's baked cache; an inherited UV_CACHE_DIR makes it
# re-download wheels the image already has.
unset UV_CACHE_DIR

# --- what to run ------------------------------------------------------------
export EXP_NAME="${EXP_NAME:-slurmjudges-control-$(date +%m%d-%H%M)}"
export CONTAINER="$MINE/containers/rl-gym.allfeatures-v2.sqsh"
export CONFIG_PATH="${CONFIG_PATH:-examples/nemo_gym/nemotron-3.5-nano/rlvr_sc_slurm_judges.yaml}"
export NRL_MAX_STEPS="${NRL_MAX_STEPS:-10}"

# --- cluster shape: 32 + 32 + 2 gym + 16 external = 82 ----------------------
export SLURM_ACCOUNT=nemotron_sw_post
export SLURM_PARTITION=batch
# Overridable so submit_control_slurm_judges_smoke.sh can reuse this whole file
# at 10 nodes instead of copying it. The defaults ARE the control arm; anything
# that changes them is by definition not the control, so a caller that overrides
# these is responsible for saying why.
#
# Whatever a caller picks, NUM_TRAIN_NODES + NUM_GEN_NODES + NUM_GYM_NODES must
# be even (SEGMENT_SIZE=2), or nano35_launch.sh refuses the shape after you have
# already waited in the queue.
export NUM_TRAIN_NODES="${NUM_TRAIN_NODES:-32}"
export NUM_GEN_NODES="${NUM_GEN_NODES:-32}"
export NUM_GYM_NODES="${NUM_GYM_NODES:-2}"
export NUM_EXTERNAL_SERVICE_NODES="${NUM_EXTERNAL_SERVICE_NODES:-16}"
# Matches the NVCF arm rather than being tuned for this one. Local judges add
# their own vLLM startup on top of the reference ~12 min to first rollouts, so
# if anything this arm needs the margin more. Without checkpointing, hitting the
# walltime loses everything.
export WALLTIME="${WALLTIME:-04:00:00}"

# --- job-reaper exemption ----------------------------------------------------
# WITHOUT THIS THE RUN IS CANCELLED AT ~30 MINUTES, BEFORE ITS FIRST STEP.
# nano35_launch.sh forwards SLURM_COMMENT to sbatch --comment, and an absent or
# malformed comment is treated as NO exemption -- silently, with an empty
# Comment field being the only trace.
#
# The 10-node control smoke 3655075 was lost exactly this way: cancelled by
# svc-hwinf-cs-sched (uid 146504) at 30:50 elapsed on BOTH hetjob components,
# with every judge healthy at the time -- safety serving, GenRM's load balancer
# up, nl2bash through its weight load. The gymscale campaign lost jobs 3577249,
# 3578472 and 3578802 the same way at 30:32-30:43. It is the
# OccupiedIdleGPUsJobReaper on its default idle-GPU threshold, NOT preemption:
# cluster PreemptExemptTime is 04:05:00 and no preemption record appears.
#
# Why this arm legitimately idles GPUs, and why the reason is benchmarking
# rather than model_loading. Both apply, and the second outlasts the first. The
# 32 Megatron training ranks hold no work until the first optimizer step, which
# waits on the first rollout cohort, which waits on the judges: 550B GenRM at
# TP8 across 12 nodes, Qwen3-235B at TP2, and a 4B safety model, none of which
# can answer a rollout until loaded. Past that, THE TRAINER IDLING IS THE
# MEASUREMENT -- this arm exists to compare judge-serving throughput with the
# judges on Slurm against the same judges on NVCF, so trainer wait time while
# the judge tier absorbs the load is the quantity being recorded, not a fault.
#
# Sized under the walltime rather than at it, since after the first step the
# training GPUs should stay busy and the doc notes exemptions are monitored.
REAPER_EXEMPT_MINS="${REAPER_EXEMPT_MINS:-120}"
export SLURM_COMMENT="${SLURM_COMMENT:-{\"OccupiedIdleGPUsJobReaper\":{\"exemptIdleTimeMins\":\"${REAPER_EXEMPT_MINS}\",\"reason\":\"benchmarking\",\"description\":\"Slurm-hosted-judge control arm for an NVCF judge-deployment comparison, matched GPU-for-GPU to the NVCF allocation (GenRM 6 replicas x TP8, Qwen3-235B 8 x TP2, safety 4 x TP1; 68 judge GPUs either way). Training GPUs are idle through the cold start because the 32 Megatron ranks cannot take an optimizer step until the first rollout cohort returns, and no rollout can complete until the on-allocation judges finish loading: 550B GenRM at TP8 over 12 nodes, Qwen3-235B FP8 at TP2, and a 4B safety model. Judge startup alone measured ~35-40 min on the 10-node smoke. Beyond startup, trainer idle time while the judge tier absorbs the rollout load IS the quantity being measured against the NVCF arm, so the idleness is the experiment rather than a fault.\"}}}"

# Invalid JSON is silently treated as NO exemption, so validate here rather than
# discovering it as a cancel at ~30 min. The summary is printed BY the parser, so
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

# --- judges served on the allocation ----------------------------------------
# HOSTED_JUDGES is left at its default of 0. Setting it to 1 is what the NVCF
# arm does; nano35_launch.sh then refuses a judge hetgroup and blanks these.
export GENRM_MODEL="$ASSETS/models/hf_judge_models/hub/models--nvidia--NVIDIA-Nemotron-3-Ultra-550B-A55B-GenRM/snapshots/116af7cb1a23ce9017b2c412945b0623252655d2"
export NL2BASH_JUDGE_MODEL="$ASSETS/models/Qwen3-235B-A22B-Instruct-2507-FP8"
export SAFETY_JUDGE_MODEL="$ASSETS/models/Nemotron-Content-Safety-Reasoning-4B"
export GENRM_REASONING_PARSER="$ASSETS/evaluation/ultra_v3_reasoning_parser.py"
# MUST be spelled /lustre, and must be set here rather than left to default.
# nano35_launch.sh derives this from PROJECT_ROOT, which it computes with
# `realpath` -- and /lustre is a symlink to /scratch on this cluster, so
# realpath always yields /scratch/... . The wrapper then rejects it: every path
# it bind-mounts into an external-service container has to start with /lustre,
# and it exits 1 before starting anything.
#
# There is no default that works. Any run using the external judge hetgroup has
# to set this explicitly, which is why job 3654343 (the control smoke) died in
# 20s with
#   [FATAL] Path must be under /lustre for the GenRM container mount: /scratch/...
# The 82-node arm would have failed identically. Now that the launcher validates
# the submission before sbatch, a wrong value fails at DRY_RUN instead of after
# the allocation is up -- but it still fails, so it still has to be set.
export EXTERNAL_VLLM_TOOLS_DIR_HOST="${EXTERNAL_VLLM_TOOLS_DIR_HOST:-$MINE/rlvr-allfeatures/tools/external_gym_vllm}"

export GENRM_REPLICAS="${GENRM_REPLICAS:-6}"
# Not overridable in practice even though it reads that way: 550B at bf16 is
# ~1.1 TB, so TP8 puts ~137 GB on each 288 GB card while TP4 would need ~275 GB
# against a ~274 GB usable ceiling. TP8 is the floor, not a tuning choice.
export GENRM_TENSOR_PARALLEL_SIZE="${GENRM_TENSOR_PARALLEL_SIZE:-8}"

# nl2bash's external pool: 16 GPUs, the same count NVCF gives it, as 4 replicas
# of TP4 because each pool replica owns whole 4-GPU nodes. NVCF's 8xTP2 split
# cannot be expressed here without sharing a node between replicas, which the
# pool wrapper deliberately does not do -- each replica runs its own private Ray
# cluster on fixed ports, and that is only safe while no two share a host.
export NL2BASH_REPLICAS="${NL2BASH_REPLICAS:-4}"
export NL2BASH_TENSOR_PARALLEL_SIZE="${NL2BASH_TENSOR_PARALLEL_SIZE:-4}"

# --- assets (shared with the NVCF arm, byte-for-byte) -----------------------
export MODEL_PATH=/lustre/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/kajalj/nvcf-poc-data/policy
export TRAIN_PATH=/lustre/fsw/portfolios/nemotron/users/arnavk/nano_35_rlvr/train_data.jsonl
export VAL_PATH="$TRAIN_PATH"

# --- output and caches ------------------------------------------------------
export RESULTS_DIR="$MINE/runs/$EXP_NAME"
export PERSISTENT_CACHE="$MINE/cache"
export HF_HOME="$PERSISTENT_CACHE/hf_home"

# --- mounts -----------------------------------------------------------------
# /lustre and /scratch: ray.sub's shared-FS canary lives under LOG_DIR and every
# head and worker exits 1 if it cannot see it. pyproject.toml and uv.lock:
# --container-save does not persist bind mounts, so the image's venv matches our
# lock but its lock file does not.
export EXTRA_MOUNTS="/lustre:/lustre,/scratch:/scratch,$MINE/rlvr-allfeatures/pyproject.toml:/opt/nemo-rl/pyproject.toml,$MINE/rlvr-allfeatures/uv.lock:/opt/nemo-rl/uv.lock"

# --- sandbox: colocated, matching the NVCF arm as resubmitted ---------------
# Both arms run colocated sandboxes so the comparison isolates judge placement.
# The same image the NVCF arm was resubmitted with, so ns_tools executes against
# identical tooling on both sides.
export SANDBOX_CONTAINER="${SANDBOX_CONTAINER:-/lustre/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/kajalj/3kruns/ultra-3k/images-striped/nemo-skills-sandbox-dc43f3e.sqsh}"
export NO_COLOCATED_SANDBOX=0

# UNSET rather than merely omit, copying launch_rlvr_sc_nvcf_disagg.sh's
# DISAGG_SANDBOX=0 branch and for its stated reason: ray.sub inherits the
# submitting shell, and these are exactly the variables a previous disaggregated
# submission leaves exported in it. Omitting them would hand this arm a live
# sandbox_pool backend and silently reintroduce the difference the colocated
# choice exists to remove -- on the control arm only, which is the worst place
# for it. MATH_FORMAL_LEAN_BACKEND goes too, so Lean takes its in-config default
# (the sidecar) rather than pointing at a pool that is not there.
unset NS_TOOLS_SANDBOX_TYPE NS_SANDBOX_POOL_REF NS_SANDBOX_POOL_SIZE \
      NS_SANDBOX_POOL_FALLBACK NS_SANDBOX_TTL_S NS_SANDBOX_IMAGE \
      OPENSANDBOX_BASE_URL OPENSANDBOX_API_KEY MATH_FORMAL_LEAN_BACKEND

# --- SingleController entrypoint --------------------------------------------
# nano35_launch.sh defaults TRAIN_ENTRYPOINT to run_grpo_nemo_gym.py, the V1
# driver. The NVCF arm reaches nano35_launch.sh through a wrapper that overrides
# it; this script calls nano35_launch.sh directly, so it must do so itself.
# Without this the control would run the V1 stack -- no SingleController, no
# streaming -- against an SC config, and the comparison would be meaningless.
# A dry run caught exactly that.
export TRAIN_ENTRYPOINT="${TRAIN_ENTRYPOINT:-./examples/run_grpo_single_controller.py}"

# Labels Gym's run; also the handle for reaping anything left behind.
export NEMO_GYM_RUN_ID="${NEMO_GYM_RUN_ID:-nano35-slurmjudges-$(date +%m%d-%H%M%S)}"

# UV_FROZEN=1 makes the driver's `uv run` use the committed lock instead of
# re-locking. A re-lock fetches packages, which CMH compute nodes cannot do, so
# the driver hangs against a blackholed CDN rather than failing. Same value the
# NVCF arm uses, which also keeps the two arms on identical dependencies.
export UV_FROZEN="${UV_FROZEN:-1}"

# --- W&B --------------------------------------------------------------------
export WANDB_ENABLED=True
export WANDB_PROJ="${WANDB_PROJ:-nemotron-3.5-nano}"

cd "$MINE/rlvr-allfeatures"

_JUDGE=examples/nemo_gym/nemotron-3.5-nano/nano35_launch.sh
_SAFETY=env.nemo_gym.safety_judge_model.responses_api_models.local_vllm_model.vllm_serve_kwargs

# One level up from the *_serve_kwargs prefix above: ray_worker_py_executable is
# a sibling of vllm_serve_kwargs, not a member of it.
_SAFETY_M=env.nemo_gym.safety_judge_model.responses_api_models.local_vllm_model

# THERE ARE NO nl2bash OVERRIDES HERE ANY MORE. nl2bash serves from an external
# vLLM pool, so nano35_launch.sh passes it a base_url and Gym never launches the
# in-process engine those keys would have configured. They were removed rather
# than left in place: a vllm_serve_kwargs override on a server that is not
# launched parses cleanly, changes nothing, and reads like the shape being
# served. Its topology now travels as NL2BASH_REPLICAS and
# NL2BASH_TENSOR_PARALLEL_SIZE above.
#
# Interpreter for the gym-resident judge's vLLM, overriding rlvr.yaml's choice
# of the RL generation-worker venv (line 507).
#
# THIS IS A WORKAROUND FOR AN UPSTREAM vLLM PACKAGING BUG, vllm-project/vllm#49103.
# vLLM gained Responses-API namespace-tool support in #47024 without raising its
# openai floor, so any build carrying that code needs openai>=2.25.0
# (openai-python#2891, where NamespaceTool was added). Every venv in
# rl-gym.allfeatures-v2.sqsh ships openai 2.6.1, so the RL venv's vllm 0.25.1
# cannot import its own module:
#
#   vllm/tool_parsers/__init__.py -> abstract_tool_parser.py -> utils.py
#   ImportError: cannot import name 'NamespaceTool' from 'openai.types.responses'
#
# That killed job 3654693 with `Process nl2bash_judge_model finished unexpectedly!`.
# It is NOT a tool-parser config problem: vllm's cli_args.py imports
# ToolParserManager unconditionally, so every judge fails, tool calling or not.
# The RL generation workers escape it only because they build the engine
# in-process instead of going through the API-server entrypoint.
#
# Gym's own local_vllm_model venv carries vllm 0.24.0, which predates the change
# -- tool_parsers/utils.py does not exist there and neither __init__.py nor
# abstract_tool_parser.py mentions NamespaceTool or imports openai. safety asks
# for no tool parser, but it needs the repoint anyway, because the failing
# import is unconditional.
#
# NEITHER EXTERNAL POOL IS REPOINTED. Both run through
# tools/external_gym_vllm/serve_vllm_on_ray.py, which applies NeMo RL's vLLM
# compatibility patch before importing the API server, and GenRM has served
# from the RL venv over that path on this cluster -- its load balancer came up
# on both the smoke and the 86-node run. nl2bash now takes the same route with
# tool_call_parser=hermes; if a replica dies on the NamespaceTool import, that
# is where to look first.
_JUDGE_PY="${JUDGE_PY:-/opt/gym_venvs/responses_api_models/local_vllm_model/.venv/bin/python}"

# vLLM admission cap, applied to ALL THREE judges so the arm is comparable to
# the NVCF deployment it is the control for. rlvr.yaml ships 256 for each
# (lines 521, 558, 651), and so did the retired single-service GenRM wrapper
# until the cap was parameterised.
#
# It has to be all three or none. Judge throughput is bounded by how many
# sequences a replica will admit, so if one arm admits 256 and the other 1024,
# the measurement is of the cap and not of where the judges are hosted -- which
# is the single variable this arm exists to isolate. The two external pools
# travel by a different route from the gym-resident safety judge
# (GENRM_MAX_NUM_SEQS and NL2BASH_MAX_NUM_SEQS into the pool definitions, versus
# a Hydra override), and that asymmetry is exactly how one of them gets left
# behind at 256.
export JUDGE_MAX_NUM_SEQS="${JUDGE_MAX_NUM_SEQS:-1024}"
export GENRM_MAX_NUM_SEQS="${GENRM_MAX_NUM_SEQS:-$JUDGE_MAX_NUM_SEQS}"
export NL2BASH_MAX_NUM_SEQS="${NL2BASH_MAX_NUM_SEQS:-$JUDGE_MAX_NUM_SEQS}"

# Each value is a variable with the control-arm shape as its default, rather
# than a literal, so the smoke can shrink the replica counts WITHOUT emitting a
# second override for the same key. Passing the same dotted key twice and
# relying on last-wins is a bet on the loader's merge semantics that there is no
# reason to take when a default costs nothing.
#
# TP has no override hook on purpose. Matching NVCF's per-replica shape is the
# whole point of this arm, and serving at that shape is what the smoke verifies;
# a knob that let a caller quietly raise it would defeat both.
exec bash "$_JUDGE" rlvr \
  "${_SAFETY}.tensor_parallel_size=1" \
  "${_SAFETY}.data_parallel_size=${SAFETY_REPLICAS:-4}" \
  "${_SAFETY}.data_parallel_size_local=${SAFETY_DP_LOCAL:-4}" \
  "${_SAFETY_M}.ray_worker_py_executable=${_JUDGE_PY}" \
  "${_SAFETY}.max_num_seqs=${JUDGE_MAX_NUM_SEQS}" \
  "$@"
