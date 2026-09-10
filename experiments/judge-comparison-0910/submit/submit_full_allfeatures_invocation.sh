#!/bin/bash
# 66-node all-features run, GenRM moved off the integrate gateway and onto NVCF's
# direct invocation API. Sibling of submit_full_allfeatures.sh, which it delegates
# to for everything that is not endpoint routing.
#
# Why this arm exists: job 3544443 logged 4458 `504 Gateway Time-out` from GenRM
# and abandoned 505 rollouts in 2h35m, with zero transport-level disconnects. The
# gateway expires a request at 30 minutes, which is exactly our group deadline, so
# a slow judgement and an abandoned rollout are indistinguishable from the client.
# The invocation API replaces that fixed expiry with NVCF-POLL-SECONDS: past the
# window it answers 202 plus an NVCF-REQID rather than a 504, and the caller polls
# for the real answer. That turns "we cannot tell slow from failed" into a number.
#
# Three things have to be true at once for this to be a valid test, which is why
# they are set here rather than passed by hand:
#
#   1. Only GenRM moves. The invocation URL carries the function UUID in its
#      hostname, so one shared base_url cannot serve three functions -- pointing
#      all three at GenRM's UUID would silently route the safety and nl2bash
#      judges to the GenRM model and quietly corrupt their scores. GenRM reads
#      NVCF_GENRM_BASE_URL; the other two stay on NVCF_JUDGE_BASE_URL's default.
#   2. The 340s idle timeout is now handled in-client by Gym's CRLF heartbeat
#      (PR #3117), not by the h2-ping-sidecar. Nothing proxies GenRM any more,
#      so its base_url is the invocation host itself. See "heartbeat" below.
#   3. The group deadline must outlast the poll budget. See WHY 2400 below.
#
# Usage:  bash submit_full_allfeatures_invocation.sh [extra hydra overrides...]
#         DRY_RUN=1 bash submit_full_allfeatures_invocation.sh
set -euo pipefail

MINE=/lustre/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/sauramishra

# --- the function under test -------------------------------------------------
# UUID as supplied in the run request. The invocation API carries it in the
# hostname, which is why GenRM needs its own base_url variable (see 1 above).
GENRM_FUNCTION_UUID=94b8435d-b749-41d5-a831-f1c1e0102870
GENRM_INVOCATION_HOST="https://${GENRM_FUNCTION_UUID}.invocation.api.nvcf.nvidia.com"

# --- heartbeat: REVERTED TO THE h2-ping-sidecar --------------------------------
# The CRLF heartbeat (Gym PR #3117) is OFF. It was the cleaner mechanism on
# paper -- no proxy hop, no loopback port, no readiness gate -- but measured
# against real traffic it truncates large judge responses, and the sidecar does
# not.
#
# Truncated responses per completed rollout group, same judges, same models:
#
#   run                              heartbeat  trunc lines  groups
#   allfeatures-nvcf-rerun-0903        off             0       905
#   allfeatures-nvcf-h2sidecar-smoke   off             0       372
#   allfeatures-nvcf-h2sidecar-0904    off (sidecar)  15     3,604
#   allfeatures-nvcf-invocation-0905   off             0     4,609
#   allfeatures-nvcf-colocated-1938    60s            63       191
#   allfeatures-nvcf-colocated-1915    60s           143        19
#
# The controlled comparison is the nl2bash/safety leg: it sat on
# integrate.api.nvidia.com throughout, same models, and went from 0 truncations
# in 4,609 groups on 0905 to 39 in 191 groups once the heartbeat was enabled.
# Only the heartbeat changed on that path. (GenRM is confounded, having moved to
# the invocation host on the same day.)
#
# Mechanism. The heartbeat writes a bare CRLF on connections WITH A REQUEST IN
# FLIGHT, justified by RFC 9112 2.2 (servers ignore empty lines before a
# request-line). Those bytes reach the gateway while it is still streaming the
# response back, and an intermediary that reads them as the start of a pipelined
# request can tear the connection down mid-response. That is why every
# truncation lands on a clean buffer boundary -- 124, 125, 192, 285, 512 KiB --
# and why only large bodies are hit: at a 60s cadence, only a response still
# open when a beat lands is exposed, and anything under ~128 KB finishes in
# between. Job 3654262 died on one such truncation surfacing as HTTP 400.
#
# The sidecar avoids this because the PING is an HTTP/2 frame on a separate
# stream, not opportunistic bytes injected into an HTTP/1.1 connection that is
# mid-response. It is protocol-legal keepalive rather than a trick.
#
# Its two historical failures were placement, not judging: 3542717 ran the
# sidecar only on the head while the Gym actor ran elsewhere (fixed -- ray.sub
# now starts one per node and gates the driver on all of them, see H2_READY),
# and 3542407 bound 8080 inside the 7000-8999 vLLM rendezvous band (fixed --
# 1250 sits in the documented 1202-1300 gap).
export H2_PING_SIDECAR="$MINE/h2-ping-sidecar/h2-ping-sidecar"
export H2_PING_LISTEN="127.0.0.1:1250"
export H2_PING_INTERVAL="60s"

# --- judge routing ------------------------------------------------------------
# Effective endpoints, which are what matter and are NOT what the base_urls say:
#
#   GenRM     https://<uuid>.invocation.api.nvcf.nvidia.com/v1   (via sidecar)
#   nl2bash   https://integrate.api.nvidia.com/v1                (direct)
#   safety    https://integrate.api.nvidia.com/v1                (direct)
#
# GenRM's base_url below points at loopback because that is how a forward proxy
# works: the sidecar re-speaks the request to H2_PING_UPSTREAM over HTTP/2.
# Traffic still egresses to the GenRM function UUID, so the dedicated 6-replica
# allocation this arm exists to measure is unchanged.
#
# ONLY GenRM goes through the proxy. It is the only judge that trips the 340s
# accelerator timeout, having the only max_output_tokens=24576, and 0905 showed
# the other two are clean on integrate.api.nvidia.com with no keepalive at all
# (0 truncations, 0 ServerDisconnectedError, 4,609 groups). Routing them through
# it would add a hop for no benefit -- and could not work anyway: the sidecar
# takes a single -upstream, and the invocation API encodes the function UUID in
# the hostname, so one proxy cannot serve three functions.
export H2_PING_UPSTREAM="${GENRM_INVOCATION_HOST}"
export NVCF_GENRM_BASE_URL="http://${H2_PING_LISTEN}/v1"
unset NVCF_JUDGE_BASE_URL 2>/dev/null || true

# Timestamped, and do not override it with a name you have already used:
# tools/code_snapshot.sh keys the snapshot directory on EXP_NAME and *reuses* an
# existing one rather than refreshing it, so a repeated name silently runs the
# older code. The 202-polling change lives in the working tree, which the
# snapshot picks up only on the copy it actually performs.
export EXP_NAME="${EXP_NAME:-allfeatures-nvcf-invocation-$(date +%m%d-%H%M)}"

echo "[invocation] GenRM  -> ${GENRM_INVOCATION_HOST}/v1 (VIA h2-ping-sidecar at ${H2_PING_LISTEN})"
echo "[invocation] others -> integrate.api.nvidia.com (direct, no proxy)"
echo "[invocation] keepalive: h2-ping-sidecar, PING every ${H2_PING_INTERVAL}"
echo "[invocation] CRLF heartbeat: OFF (truncates large responses; see header)"
echo "[invocation] NVCF-POLL-SECONDS: 3600 (set in nvcf_judges.yaml default_headers)"
echo "[invocation] rollout_timeout_s: 2400 (CLI override below, > 1900s poll budget)"

# Three timeouts have to stay strictly ordered, and MasterConfig enforces the
# second inequality with a validator -- getting it wrong fails at config parse,
# which is how job 3567869 died in 5m17s:
#
#   poll budget 1900s  <  rollout_timeout_s 2400s  <  stall_timeout_s 3000s
#
# 1900 is NeMoGymAsyncOpenAI.nvcf_poll_timeout_seconds, sitting just past NVCF's
# own 30-minute expiry so the terminal answer comes from the service rather than
# from us abandoning it early. rollout_timeout_s must clear that budget: the
# base recipe's 1800s would fire *before* the poll finished, recording every
# slow judgement as an abandoned rollout. And stall_timeout_s must clear
# rollout_timeout_s STRICTLY -- `stall_action: warn` does NOT make an equal
# value harmless, the validator rejects >= outright.
#
# These stay as CLI overrides HERE, on this arm only, and must not be migrated
# into a shared overlay. The 1900s poll budget is an NVCF property: it exists
# because judgements travel to a hosted endpoint that can take 30 minutes to
# answer. The Slurm control arm has no poll budget at all, so it keeps the base
# recipe's 1800/2400 and is correct at those values. An asymmetry in these two
# deadlines is therefore REQUIRED for the arms to be comparable, not a break in
# parity -- forcing both to one value is what would distort the comparison, by
# making this arm abandon slow-but-healthy hosted judgements as failures.
#
# 2400/3000 also reproduce job 3657252, the reference this pairing is anchored
# to, exactly as submitted.
exec bash "$MINE/submit_full_allfeatures.sh" \
  "async_rl.rollout_failure.nemo_gym.rollout_timeout_s=2400.0" \
  "async_rl.stall_watchdog.stall_timeout_s=3000.0" \
  "$@"
