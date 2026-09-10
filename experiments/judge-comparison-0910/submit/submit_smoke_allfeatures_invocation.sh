#!/bin/bash
# 8-node smoke for the NVCF-hosted-judge arm, on the SAME routing as the full
# invocation run (submit_full_allfeatures_invocation.sh).
#
# Why this file exists. submit_smoke_allfeatures.sh execs
# launch_rlvr_sc_nvcf_disagg.sh *directly* and never goes through
# submit_full_allfeatures.sh, so it inherits none of the invocation arm's setup:
# no dedicated GenRM endpoint, no h2-ping-sidecar, no colocated sandbox, and
# none of the timeout ordering. Run bare, it would smoke a configuration we are
# not shipping -- GenRM on the shared integrate.api.nvidia.com gateway instead
# of the dedicated function, and disaggregated sandboxes instead of colocated.
# This wrapper is to the smoke what the invocation wrapper is to the full arm.
#
# Usage:  bash submit_smoke_allfeatures_invocation.sh [extra hydra overrides...]
#         DRY_RUN=1 bash submit_smoke_allfeatures_invocation.sh
set -euo pipefail

MINE=/lustre/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/sauramishra

# --- endpoints: identical to the full invocation arm ------------------------
# Each NVCF function gets its own invocation hostname, so one shared base_url
# cannot serve all three. GenRM points at the dedicated function; the other two
# fall through to nvcf_judges.yaml's integrate.api.nvidia.com default, which is
# why NVCF_JUDGE_BASE_URL is unset rather than assigned.
GENRM_FUNCTION_UUID=94b8435d-b749-41d5-a831-f1c1e0102870
GENRM_INVOCATION_HOST="https://${GENRM_FUNCTION_UUID}.invocation.api.nvcf.nvidia.com"

# --- keepalive: h2-ping-sidecar, NOT the CRLF heartbeat ---------------------
# The CRLF heartbeat (Gym PR #3117) truncates large judge responses: it writes
# bytes into a response body that is still open, and job 3654262 died when one
# such truncation surfaced as an HTTP 400, which classifies DATA and so ends the
# run outright under ready_first. nvcf_judges.yaml pins
# global_aiohttp_crlf_heartbeat_seconds to 0; the sidecar's PING is a real
# HTTP/2 frame on its own stream and cannot corrupt a body.
export H2_PING_SIDECAR="$MINE/h2-ping-sidecar/h2-ping-sidecar"
export H2_PING_LISTEN="127.0.0.1:1250"
export H2_PING_INTERVAL="60s"
export H2_PING_UPSTREAM="${GENRM_INVOCATION_HOST}"

# Loopback is correct: the sidecar is a forward proxy that re-speaks the request
# to H2_PING_UPSTREAM over HTTP/2.
export NVCF_GENRM_BASE_URL="http://${H2_PING_LISTEN}/v1"
unset NVCF_JUDGE_BASE_URL 2>/dev/null || true

# --- sandbox: colocated, matching the full arm and the Slurm control --------
# The launcher defaults DISAGG_SANDBOX=1. submit_smoke_allfeatures.sh sets
# NS_SANDBOX_POOL_SIZE=8 unconditionally, which cannot be overridden from here
# -- but it does not need to be: the launcher's DISAGG_SANDBOX=0 branch unsets
# the whole NS_SANDBOX_* family itself, for the inherited-shell reason.
#
# Colocated also removes this smoke's dependency on the remote ns-tools-warm
# pool being provisioned, which is one of the open blockers on the full arm.
export DISAGG_SANDBOX=0
export SANDBOX_CONTAINER="${SANDBOX_CONTAINER:-/lustre/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/kajalj/3kruns/ultra-3k/images-striped/nemo-skills-sandbox-dc43f3e.sqsh}"
export NO_COLOCATED_SANDBOX=0

# Fresh name every run: tools/code_snapshot.sh keys the snapshot directory on
# EXP_NAME and REUSES an existing one rather than refreshing it, so a repeated
# name silently runs older code. The cohort-retry-identity fix lives in the
# working tree and only lands in the copy the snapshot actually performs.
export EXP_NAME="${EXP_NAME:-allfeatures-nvcf-inv-smoke-$(date +%m%d-%H%M)}"

echo "[inv-smoke] GenRM  -> ${GENRM_INVOCATION_HOST}/v1 (VIA h2-ping-sidecar at ${H2_PING_LISTEN})"
echo "[inv-smoke] others -> integrate.api.nvidia.com (direct, no proxy)"
echo "[inv-smoke] keepalive: h2-ping-sidecar, PING every ${H2_PING_INTERVAL}"
echo "[inv-smoke] CRLF heartbeat: OFF"
echo "[inv-smoke] sandbox: COLOCATED (DISAGG_SANDBOX=0), matching full arm + control"
echo "[inv-smoke] EXP_NAME=${EXP_NAME}"

# Same three-way timeout ordering the full arm documents, and for the same
# reason -- the inherited values are wrong in both directions:
#
#   poll budget 1900s  <  rollout_timeout_s 2400s  <  stall_timeout_s 3000s
#
# 1900 is NeMoGymAsyncOpenAI.nvcf_poll_timeout_seconds. The inherited
# rollout_timeout_s of 1800 would fire BEFORE the poll finished, recording every
# slow judgement as an abandoned rollout. stall_timeout_s must then clear
# rollout_timeout_s strictly -- the validator rejects >= outright, and
# `stall_action: warn` does not make an equal value harmless.
# Carried as CLI overrides to match submit_full_allfeatures_invocation.sh. A
# smoke that rehearses the full run at different deadlines cannot clear the full
# run's timeout ordering on its behalf, which is the one failure mode this smoke
# is cheapest at catching: job 3567869 died in 5m17s on exactly that validator.
exec bash "$MINE/submit_smoke_allfeatures.sh" \
  "async_rl.rollout_failure.nemo_gym.rollout_timeout_s=2400.0" \
  "async_rl.stall_watchdog.stall_timeout_s=3000.0" \
  "$@"
