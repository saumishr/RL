#!/bin/bash
# Milestone watcher for one RLVR arm. Usage: watch_arm.sh <jobid> <run-dir>
#
# Watches the run's *wandb-captured* console output, not the Ray driver log. The
# step markers only exist there:
#   train_pump: step N chunk M: X group(s), Y/Z dispatched
#   _sync_weights: sync done in <t>s
# The driver log's `Collecting rollouts:` tqdm bars are carriage-return updates
# that render as runs of blank lines, so tailing it looks dead while it grows.
#
# Prints MILESTONE for anything worth waking up for. The error set is the two
# bugs that killed job 3654874 at 31 minutes -- vLLM's `deque mutated during
# iteration` on the abort path (Bug A) and genrm_compare's cohort-overflow
# AssertionError (Bug B) -- plus the failure classes that end a run outright
# under ready_first, where max_skipped_prompts is pinned at 0.
set -uo pipefail

JOBID="${1:?usage: watch_arm.sh <jobid> <run-dir>}"
RUNDIR="${2:?usage: watch_arm.sh <jobid> <run-dir>}"

ERRPAT='EngineDeadError|deque mutated|AssertionError|RolloutDataFailure|CohortEvaluationError|ClientPayloadError|ContentLengthError'

prev_asm=""
prev_sync=0
prev_err=0
announced_first_step=0
waited=0

for _ in $(seq 1 400); do
  state=$(squeue -j "$JOBID" -h -o "%T" 2>/dev/null | head -1)
  if [[ "$state" != "RUNNING" ]]; then
    echo "MILESTONE [$(date +%H:%M:%S)]: job $JOBID state='${state:-GONE}' -- stopping watch"
    break
  fi

  # Resolve lazily: the wandb run dir does not exist until the driver calls
  # wandb.init, which on a 66-node job is ~10 min after the allocation starts.
  P=$(ls -1t "$RUNDIR"/runs/*/logs/exp_001/wandb/wandb/run-*/files/output.log 2>/dev/null | head -1)
  if [[ -z "$P" ]]; then
    waited=$((waited + 1))
    # Every ~10 min, so a stuck startup is visible without spamming.
    if (( waited % 13 == 0 )); then
      echo "[$(date +%H:%M:%S)] still starting up (no wandb output.log yet, ${waited} polls)"
    fi
    sleep 45
    continue
  fi

  asm=$(grep -ohE "step [0-9]+ chunk [0-9]+: [0-9]+ group\(s\), [0-9]+/[0-9]+ dispatched" "$P" 2>/dev/null | tail -1)
  sync=$(grep -c "_sync_weights: sync done" "$P" 2>/dev/null)
  err=$(grep -cE "$ERRPAT" "$P" 2>/dev/null)
  step1=$(grep -cE "train_pump: step 1 " "$P" 2>/dev/null)

  if [[ "$asm" != "$prev_asm" ]]; then
    echo "[$(date +%H:%M:%S)] $asm | syncs=$sync errs=$err"
    prev_asm="$asm"
  fi

  if (( sync > prev_sync )); then
    last=$(grep -oE "_sync_weights: sync done in [0-9.]+s" "$P" 2>/dev/null | tail -1)
    # The first sync is the startup weight load, not a post-step refit.
    if (( prev_sync > 0 )); then
      echo "MILESTONE [$(date +%H:%M:%S)]: WEIGHT REFIT #$sync -- $last"
    else
      echo "MILESTONE [$(date +%H:%M:%S)]: startup weight sync -- $last"
    fi
    prev_sync="$sync"
  fi

  if (( step1 > 0 && announced_first_step == 0 )); then
    echo "MILESTONE [$(date +%H:%M:%S)]: FIRST STEP COMPLETE (step 0 trained + refit), step 1 assembling"
    announced_first_step=1
  fi

  if (( err > prev_err )); then
    echo "MILESTONE [$(date +%H:%M:%S)]: ERRORS +$((err - prev_err)) (total=$err)"
    grep -ohE "$ERRPAT" "$P" 2>/dev/null | sort | uniq -c | sed 's/^/    /'
    prev_err="$err"
  fi

  sleep 45
done

echo "watch loop exited at $(date +%H:%M:%S)"
