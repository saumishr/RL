#!/bin/bash
# Monitor the GenRM-on-invocation-API run.
#
# Usage: bash monitor_invocation.sh <jobid> [run-glob]
#
# Why this is not monitor_disconnects.sh with a new job id: that script watches
# `504 Gateway Time-out` and `ServerDisconnectedError`, and on this path both go
# to zero by construction. The invocation API answers 202 instead of 504 once a
# request outlives NVCF-POLL-SECONDS, so the old script would print "errors=0"
# through a run that was silently feeding default scores into the reward. The
# counters that matter here are the poll outcomes and the GenRM parse fallback.
#
# Scoping is deliberately narrow, because earlier passes over these logs produced
# phantom counts -- a bare "529" matched the port 5299 in a uvicorn banner, "403"
# matched an rsync rate of 403.32kB/s, and "ServerDisconnected" matched the text
# of the monitoring script itself. Every pattern below is a full status phrase, an
# exception class, or one of our own bracketed log tags, greps are confined to the
# gym tree, and this script writes its output outside it.
set -u

JOB="${1:?usage: monitor_invocation.sh <jobid> [run-glob]}"
GLOB="${2:-allfeatures-nvcf-invocation-*}"

B=/lustre/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/sauramishra
R=$(ls -td "$B"/runs/$GLOB/ 2>/dev/null | head -1)
if [[ -z "$R" ]]; then
  echo "FATAL no run dir matching $B/runs/$GLOB" >&2
  exit 1
fi
G="$R/runs/latest/logs/nemo_gym"
L="$R/ray_logs/${JOB}-logs"
echo "watching job=$JOB"
echo "  gym logs:    $G"
echo "  ray logs:    $L"

DEADLINE=$((SECONDS + 6 * 3600))
PREV=-1
TICK=0

# Two log surfaces, and missing the second one hid 10 real errors last run: the
# per-server logs under logs/nemo_gym carry what each uvicorn app reports, but the
# NemoGym Ray actor writes its client-side failures to the driver log instead.
count()     { grep -rhoF "$1" "$G"/*.log 2>/dev/null | wc -l; }
count_drv() { grep -hoF  "$1" "$L"/ray-driver.log 2>/dev/null | wc -l; }
both()      { echo $(( $(count "$1") + $(count_drv "$1") )); }

while (( SECONDS < DEADLINE )); do
  if ! squeue -j "$JOB" -h -o "%t" 2>/dev/null | grep -q .; then
    echo "TERMINAL $(date +%H:%M:%S) job gone: $(sacct -j "$JOB" --format=State%20,Elapsed -X -Pn 2>/dev/null | head -1)"
    exit 0
  fi

  # --- the new path: did 202-then-poll actually work? ------------------------
  # POLLED counts resolutions that reached a terminal status; the two ERR_*
  # counters are the cases where _resolve_nvcf_pending gave up and handed a
  # non-completion body downstream, which is exactly what becomes a default score.
  POLLED=$(both "[nvcf_poll reqid=")
  POLL_OK=$(both "final_status=200")
  POLL_TMO=$(both "error=poll_timeout")
  ERR_REQID=$(both "error=missing_nvcf-reqid")
  ERR_LOC=$(both "error=missing_location")

  # --- the silent failure this run exists to detect -------------------------
  # GenRM falling back to default_score=3.0. logger.warning, so it never reaches
  # a metric -- a run can look healthy while its reward signal is constant.
  FB1=$(both "No parseable JSON found in GenRM output:")
  FB2=$(both "Error parsing GenRM output:")
  FALLBACK=$((FB1 + FB2))

  # --- old counters, kept as a control --------------------------------------
  # Expected to stay at 0 on the invocation path. If gw_timeout climbs, the
  # NVCF-POLL-SECONDS header is not being honoured and we are back on 3544443's
  # behaviour; if disconnect climbs, the CRLF heartbeat is not keeping the
  # connection alive through the accelerator's 340s idle timeout.
  GT=$(both "504 Gateway Time-out")
  BG=$(both "502 Bad Gateway")
  SD=$(both ServerDisconnectedError)
  RTO=$(count_drv RolloutTimeout)

  # The heartbeat replaced the sidecar, and its whole failure mode is being
  # silently off: at the default of 0 the connector is a plain TCPConnector and
  # disconnect would climb with nothing saying why. This counts the Gym
  # processes that reported it ON at startup, so hb=0 means the config key never
  # reached GlobalAIOHTTPAsyncClientConfig and the run is not testing anything.
  HB=$(both "[aiohttp] CRLF heartbeat: every")

  # Progress is COMPLETED prompt groups, not a tqdm position: each "Collecting
  # rollouts" bar tracks the 16 generations of one group, so the last bar's
  # position says nothing about the step. 512 groups is one optimizer cohort.
  PROG=$(sed 's/\x1b\[[0-9;]*[A-Za-z]//g' "$L"/ray-driver.log 2>/dev/null \
    | grep -c 'Collecting rollouts: *100%')

  TOTAL=$((POLLED + FALLBACK + GT + BG + SD + RTO))
  STAMP="$(date +%H:%M:%S) elapsed=$(squeue -j "$JOB" -h -o '%M' 2>/dev/null)"

  if (( TOTAL != PREV )); then
    echo "STAT $STAMP poll[n=$POLLED ok=$POLL_OK timeout=$POLL_TMO no_reqid=$ERR_REQID no_loc=$ERR_LOC] GENRM_FALLBACK=$FALLBACK control[gw_timeout=$GT bad_gw=$BG disconnect=$SD rollout_timeout=$RTO hb_on=$HB] groups=${PROG:-?}"
    PREV=$TOTAL
  elif (( TICK % 5 == 0 )); then
    echo "ok   $STAMP quiet poll_n=$POLLED fallback=$FALLBACK groups=${PROG:-?}"
  fi

  TICK=$((TICK + 1))
  sleep 120
done
echo "TERMINAL watcher deadline reached"
