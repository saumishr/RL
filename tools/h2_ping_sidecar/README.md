# h2-ping-sidecar

A per-node HTTP/1.1 → HTTP/2 proxy that emits PING frames while waiting for a
response, so long judge calls survive an idle-timeout intermediary.

## Why it exists

NVCF endpoints sit behind AWS Global Accelerator, which drops any TCP connection
carrying no application-layer data for 340s. That limit is fixed by AWS, and TCP
keepalive does not reset it because only bytes above the TCP layer count. GenRM
answers in ~296s at p50 and ~439s at p95 with `max_output_tokens=24576`, and
produces no bytes at all until the first response token, so a large share of
requests exceed the cutoff and fail with `ServerDisconnectedError`. Job 3544443
logged 4,458 `504 Gateway Time-out` from this before the sidecar existed.

HTTP/2 PING frames are application data and do reset the timer, but aiohttp
speaks HTTP/1.1 only and cannot send them. The sidecar owns the upstream
connection instead and pings it on the client's behalf.

Only GenRM needs it. It is the only judge with a 24576-token output budget, and
run `allfeatures-nvcf-invocation-0905` confirmed the other two judges are clean
on `integrate.api.nvidia.com` with no keepalive at all (0 truncations, 0
`ServerDisconnectedError`, 4,609 groups). Routing them through it would add a
hop for no benefit, and could not work anyway: the sidecar takes a single
`-upstream`, and the invocation API encodes the function UUID in the hostname,
so one proxy cannot serve three functions.

## Why not the CRLF heartbeat

Gym PR #3117 adds a `global_aiohttp_crlf_heartbeat_seconds` option that writes a
bare CRLF on in-flight connections instead, needing no proxy hop, no loopback
port and no readiness gate. It is the cleaner mechanism on paper, but measured
against real traffic it truncates large judge responses:

| Run | Heartbeat | Trunc lines | Groups |
|---|---|---|---|
| `allfeatures-nvcf-rerun-0903` | off | 0 | 905 |
| `allfeatures-nvcf-h2sidecar-smoke` | off | 0 | 372 |
| `allfeatures-nvcf-h2sidecar-0904` | off (sidecar) | 15 | 3,604 |
| `allfeatures-nvcf-invocation-0905` | off | 0 | 4,609 |
| `allfeatures-nvcf-colocated-1938` | 60s | 63 | 191 |
| `allfeatures-nvcf-colocated-1915` | 60s | 143 | 19 |

The controlled comparison is the nl2bash/safety leg: it stayed on
`integrate.api.nvidia.com` throughout with the same models, and went from 0
truncations in 4,609 groups to 39 in 191 groups once the heartbeat was enabled.
Only the heartbeat changed on that path. (GenRM is confounded, having moved to
the invocation host the same day.)

The mechanism is that those CRLF bytes reach the gateway while it is still
streaming the response back, and an intermediary that reads them as the start of
a pipelined request can tear the connection down mid-response. That is why every
truncation lands on a clean buffer boundary — 124, 125, 192, 285, 512 KiB — and
why only large bodies are hit: at a 60s cadence only a response still open when
a beat lands is exposed, and anything under ~128 KB finishes in between. Job
3654262 died on one such truncation surfacing as HTTP 400.

A PING frame is protocol-legal keepalive on a separate stream rather than
opportunistic bytes injected into an HTTP/1.1 connection that is mid-response,
which is why the sidecar does not have this failure mode.

## Build

Needs Go 1.27+ (it uses `http.Protocols` and `http.HTTP2Config`).

```bash
cd tools/h2_ping_sidecar
go build -o h2-ping-sidecar .
go build -o testserver ./testserver     # optional, for the local test below
```

The binary is standalone with no cgo, so it can be built once and staged on a
shared filesystem for the job to pick up.

## Run

```bash
./h2-ping-sidecar \
  -listen 127.0.0.1:1250 \
  -upstream "https://<function-uuid>.invocation.api.nvcf.nvidia.com" \
  -ping-interval 60s \
  -ready-file /path/to/H2_READY_$(hostname)
```

`-upstream` must be `https://`, since HTTP/2 is negotiated over ALPN, and the
process refuses to start if `-ping-interval` is at or above the 340s limit it
exists to defeat. It is HTTP/2-only on purpose: falling back to HTTP/1.1 would
silently reintroduce the failure, because PING frames do not exist there.

`-ready-file` is written only after the listener is bound. A supervisor that
probes the port instead cannot tell this listener from an unrelated process
already holding it, and would let clients send traffic to the wrong server.

## How the recipe wires it

`ray.sub` starts one sidecar per node and gates the driver on all of them. One
per node is required because it listens on loopback and Ray does not pin the Gym
actor to the head node — job 3542717 failed exactly that way, with the sidecar on
the head and the Gym servers elsewhere.

Port 1250 is deliberate: it sits in the documented 1202–1300 gap. Job 3542407
bound 8080 and collided with the 7000–8999 vLLM rendezvous band.

The submit script exports `H2_PING_SIDECAR`, `H2_PING_LISTEN`,
`H2_PING_INTERVAL` and `H2_PING_UPSTREAM`, then points the judge at
`http://127.0.0.1:1250/v1`. Leaving `H2_PING_SIDECAR` unset disables the sidecar
entirely. See `examples/nemo_gym/nemotron-3.5-nano/nvcf_judges.yaml` and
`experiments/judge-comparison-0910/RUNBOOK-judge-comparison.md`.

Expect `status=502 kind=rate_limit` events carrying `http2: Transport received
Server's graceful shutdown GOAWAY` — 69 of them across a 10-step run. They are
the NVCF front-end recycling connections and all retried successfully at `try=1
max_tries=4`.

## Testing without NVCF

`testserver` is a local HTTP/2 origin that withholds its response for a fixed
delay, which is how the PING behaviour gets verified without a live API key:

```bash
./testserver -listen 127.0.0.1:8443 -delay 400s &
./h2-ping-sidecar -listen 127.0.0.1:1250 -upstream https://127.0.0.1:8443 \
  -ping-interval 60s -insecure-skip-verify &
curl -sS http://127.0.0.1:1250/
```

`ab_test.sh` is the A/B against the real cutoff, run from inside a live job where
the node-local sidecar is already listening. Arm A goes direct to NVCF and is
expected to die at ~340s; arm B goes through the sidecar and should outlive it.
It needs `NVIDIA_API_KEY` exported.

## Caveat when sharing this branch

The `3rdparty/Gym-workspace/Gym` submodule pointer on this branch is
`958f50fc63a8c244529b21c2e20444caaf180416`, which exists **only on the Lustre
filesystem it was built on** and is on no remote branch of `NVIDIA-NeMo/Gym`. A
fresh clone of this branch cannot fetch it, so `git submodule update --init`
will fail on that path. Nothing in this directory depends on the submodule —
the sidecar is a standalone Go module — but the recipe configs do.
