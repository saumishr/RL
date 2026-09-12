#!/bin/bash
# A/B test of the 340s Global Accelerator cutoff, run from inside a live job so
# the node-local sidecar is already listening.
#
# Arm A goes direct to NVCF and is expected to die at ~340s.
# Arm B goes through the sidecar and should outlive it.
# Both ask GenRM for a very long generation so the response cannot arrive early.
set -u

KEY="${NVIDIA_API_KEY:?NVIDIA_API_KEY must be exported}"
MODEL="private/ultra-genrm/nvidia/nemotron3-ultra-genrm"
OUT="${1:-/tmp}"

read -r -d '' PROMPT <<'EOF'
Write an extremely detailed, exhaustive technical monograph on the design of
distributed reinforcement learning systems. Cover rollout collection, reward
modelling, policy optimisation, checkpointing, fault tolerance, network
topology, and failure analysis. Do not summarise. Be as long and thorough as
you possibly can, with numbered sections and deep subsections throughout.
EOF

body() {
  printf '{"model":"%s","max_tokens":24576,"temperature":1.0,"messages":[{"role":"user","content":"%s"}]}' \
    "$MODEL" "$(echo "$PROMPT" | tr '\n' ' ' | sed 's/"/\\"/g')"
}

probe() {
  local name=$1 url=$2
  local start end
  start=$(date +%s.%N)
  # -sS keeps errors visible; no --max-time so the server decides the outcome.
  code=$(curl -sS -o "$OUT/ab_${name}.body" -w '%{http_code}' \
    -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
    -X POST "$url/chat/completions" --data "$(body)" 2>"$OUT/ab_${name}.err")
  rc=$?
  end=$(date +%s.%N)
  printf '%-10s http=%-4s curl_rc=%-3s elapsed=%.1fs  %s\n' \
    "$name" "${code:-none}" "$rc" "$(echo "$end - $start" | bc)" \
    "$(head -c 120 "$OUT/ab_${name}.err" 2>/dev/null | tr '\n' ' ')"
}

echo "host=$(hostname)  started=$(date -Is)"
probe direct  "https://integrate.api.nvidia.com/v1" &
P1=$!
probe sidecar "http://127.0.0.1:1250/v1" &
P2=$!
wait $P1 $P2
echo "finished=$(date -Is)"
