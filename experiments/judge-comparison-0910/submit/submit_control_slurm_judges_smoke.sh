#!/bin/bash
# 10-node smoke for the Slurm-judge control arm.
#
# Sibling of submit_smoke_allfeatures.sh, which smokes the NVCF arm at 8 nodes.
# This one proves the OTHER leg: that the three judges come up ON the allocation
# and answer, rather than being reached at an NVCF endpoint.
#
# It delegates to submit_control_slurm_judges.sh rather than copying it, so the
# container, checkpoint, data, credentials, colocated-sandbox handling and
# SingleController entrypoint are all inherited from the arm being smoked. Only
# the shape and the batch differ, and both are visible here.
#
# WHY 10 NODES AND NOT 8
#
# The NVCF smoke fits in 8 because its judges are hosted elsewhere and cost it
# nothing. This arm pays for them:
#
#   train                                     4   (floor: 16 GPUs is exactly
#                                                  TP4 x CP4 x PP1, so DP=1)
#   generation                                2
#   gym     nl2bash 1xTP2 = 2 GPU
#           safety  1xTP1 = 1 GPU             2   (see parity note below)
#                                            ---
#           Ray subtotal                      8
#   GenRM   1 replica x TP8 = 8 GPU           2   (2 nodes per replica)
#                                            ---
#                                            10
#
# Both gym-resident judges fit on ONE node (3 of its 4 GPUs), which would make
# the Ray group 7. nano35_launch.sh requires that total to divide by
# SEGMENT_SIZE=2 and refuses an odd number, so gym takes 2 nodes. This is the
# same parity rule that pads the full arm's gym tier from 5 nodes to 6.
#
# GenRM cannot go below 2 nodes: Nemotron-3-Ultra-550B at bf16 is ~1.1 TB, TP8
# puts ~137 GB on each 288 GB card, and TP4 would need ~275 GB against a ~274 GB
# usable ceiling. TP8 x 4 GPUs/node = 2 nodes for a single replica.
#
# WHAT IT PROVES
#
# Every judge runs at ONE replica, so this does NOT exercise the replication or
# load-balancer spread the full arm exists to measure. It covers the unknowns
# that would otherwise surface only after an 82-node allocation is up:
#
#   * Qwen3-235B loads and serves at TP2. The full arm halves TP from rlvr.yaml's
#     4 to match the NVCF deployment, and nothing has run that shape here. Too
#     little TP fails at KV-cache allocation, AFTER the weights load.
#   * The 4B safety judge serves at TP1, down from rlvr.yaml's TP4.
#   * GenRM serves on-allocation behind genrm_lb, not at an NVCF endpoint.
#   * nano35_launch.sh takes the SingleController entrypoint rather than
#     silently falling back to the V1 driver.
#
# Usage:  bash submit_control_slurm_judges_smoke.sh [extra hydra overrides...]
#         DRY_RUN=1 bash submit_control_slurm_judges_smoke.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXP_NAME="${EXP_NAME:-slurmjudges-control-smoke-$(date +%m%d-%H%M)}"
export CONFIG_PATH=examples/nemo_gym/nemotron-3.5-nano/rlvr_sc_slurm_judges_smoke.yaml
export NRL_MAX_STEPS="${NRL_MAX_STEPS:-2}"

# 4 + 2 + 2 = 8 Ray nodes (even, as SEGMENT_SIZE=2 requires) + 4 external.
#
# The external tier is now two pools, not one: GenRM 1 x TP8 = 2 nodes plus
# NL2Bash 2 x TP4 = 2 nodes. NL2Bash takes TWO replicas rather than one for two
# reasons. GENRM_SEGMENT_SIZE inherits SEGMENT_SIZE=2 and the launcher refuses an
# external total that is not divisible by it, so a single TP4 replica gives 3 and
# fails before sbatch. And a second replica is what makes the smoke actually
# exercise vllm_pool_lb's balancing across replicas, which is the new machinery
# this smoke exists to verify -- a one-replica pool would leave the load
# balancer's fan-out untested and pass regardless.
export NUM_TRAIN_NODES=4
export NUM_GEN_NODES=2
export NUM_GYM_NODES=2
export NUM_EXTERNAL_SERVICE_NODES=4
export GENRM_REPLICAS=1

# 2h is the MaxWall of the short QoS and ultra_launch.sh selects that QoS at
# <= 2:00:00. Worth having: short carries Priority=200 against normal's 100 and
# a far shorter queue, and a 10-node smoke that waits six hours for a slot tells
# you nothing you could not have learned sooner.
export WALLTIME="${WALLTIME:-02:00:00}"

# Reaper exemption, inherited from submit_control_slurm_judges.sh but sized down
# from its 120 min to sit under this job's shorter walltime. 90 min covers the
# ~35-40 min judge cold start measured on job 3655075 plus the first step, with
# margin; after that the training GPUs are busy and the exemption is moot.
#
# This is the setting whose ABSENCE killed 3655075 at 30:50, ten minutes short
# of its first step, with all three judges healthy.
export REAPER_EXEMPT_MINS="${REAPER_EXEMPT_MINS:-90}"

# --- shrunk judge tiers ------------------------------------------------------
# Set through submit_control_slurm_judges.sh's own hooks, so exactly ONE
# override per key reaches the command line.
#
# NL2BASH_REPLICAS now counts EXTERNAL POOL REPLICAS, each an independent DP=1
# server owning a whole 4-GPU node, where it used to be an in-Gym DP degree. So
# it sizes hetgroup 1 rather than a vLLM argument, and NL2BASH_DP_LOCAL is gone
# with the in-Gym deployment it configured. TP is still left alone -- serving at
# the full arm's TP4 is what this smoke verifies -- and safety keeps its in-Gym
# DP hooks because safety is still a Gym-managed local_vllm_model.
export NL2BASH_REPLICAS=2
export SAFETY_REPLICAS=1
export SAFETY_DP_LOCAL=1

# 8 prompts per step cannot satisfy the configured 0.9 floor, exactly as in
# submit_smoke_allfeatures.sh. Not a fault-tolerance loosening: with 8 prompts,
# 0.9 rounds to requiring all of them.
exec bash "$HERE/submit_control_slurm_judges.sh" \
  async_rl.rollout_failure.min_step_batch_fraction=0.75 \
  "$@"
