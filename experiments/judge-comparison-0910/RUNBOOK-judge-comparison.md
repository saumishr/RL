# Runbook: NVCF vs Slurm judge-hosting comparison

Everything needed to re-run either arm without repeating the smoke runs, the
topology tuning, or the deadlock debugging. Both arms below are **known-good
and completed 10 steps**. Results are analysed in `nvcf-vs-slurm-judges-report.md`.

Paths use `MINE=/lustre/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/sauramishra`.

---

## READ FIRST: state of the recipe

The recipe **is now committed locally** — it previously existed only as 18
uncommitted files. Recovery points:

| | |
|---|---|
| Recipe commit | `547ff28eeafe6edd742354b6adfcf86af331e227` |
| Docs + ledger + script copies | see `experiments/judge-comparison-0910/` |
| Tag | `judge-comparison-0910` |
| Branch | `rlvr-convergence-allfeatures` in `$MINE/rlvr-allfeatures` |
| Previous HEAD | `7f2947ae661be7a612e44c77755e39f2b7641839` |

**Nothing has been pushed.** All of the above lives only on this filesystem, so
a loss of `$MINE` still loses the work. The Gym submodule commit
`958f50fc63a8c244529b21c2e20444caaf180416` is likewise local-only, and the
parent now records that pointer.

The files the recipe consists of, for reference — modified (tracked):

```
3rdparty/Gym-workspace/Gym                                  <- submodule pointer moved
examples/nemo_gym/nemotron-3-ultra/ultra_launch.sh
examples/nemo_gym/nemotron-3.5-lightning/lightning35_launch.sh
examples/nemo_gym/nemotron-3.5-nano/launch_rlvr_sc_nvcf_disagg.sh
examples/nemo_gym/nemotron-3.5-nano/nano35_launch.sh        <- external pool registration
examples/nemo_gym/nemotron-3.5-nano/nvcf_judges.yaml
examples/nemo_gym/nemotron-3.5-nano/readyfirst_cpu_rdma.yaml
examples/nemo_gym/nemotron-3.5-nano/rlvr_sc_nvcf_disagg.yaml   <- NVCF arm config
examples/nemo_gym/nemotron-3.5-nano/rlvr_sc_slurm_judges.yaml  <- control arm config
nemo_rl/environments/nemo_gym.py
ray.sub
tools/external_genrm/run_in_allocation.sh
```

Untracked: `.check_artifacts.sh`, `.check_gym_configs.sh`, `.paths.txt`,
`examples/nemo_gym/nemotron-3.5-nano/gym_fanout_lag4.yaml`,
`rl6k-include-95552c969.patch`, `rl6k-include-ad1ce8f3d-ee12fc57d.patch`.

`gym_fanout_lag4.yaml` is **dead** — it was deliberately dropped from both
defaults chains to restore parity. It is committed for history only. Do not
re-add it to a defaults chain.

The submit scripts under `$MINE/*.sh` are copied into
`experiments/judge-comparison-0910/submit/`. **Those copies are a snapshot, not
the source of truth** — edit and launch from `$MINE/`, and re-copy when they
change.

---

## Provenance of the two reference runs

| | Control (Slurm judges) | Test (NVCF judges) |
|---|---|---|
| Slurm job | **3659144** | **3657252** |
| wandb | `joc/nemotron-3.5-nano/1wmglhr1` | `joc/nemotron-3.5-nano/3gt4e0ai` |
| Run dir | `$MINE/runs/slurmjudges-control-0910-0240` | `$MINE/runs/allfeatures-nvcf-invocation-0909-2343` |
| Steps completed | 10 | 10 |
| Container | `$MINE/containers/rl-gym.allfeatures-v2.sqsh` | same |
| RL tree | `$MINE/rlvr-allfeatures` @ `7f2947ae` + dirty | same |

Both arms share the container, policy checkpoint, dataset, train/gen node
counts, sampler, data plane, sandbox setup and step budget. **Judge placement is
the only intended independent variable.**

---

## Arm A — Control, judges on Slurm (82 nodes)

```bash
cd /lustre/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/sauramishra
bash submit_control_slurm_judges.sh                 # add hydra overrides as trailing args
DRY_RUN=1 bash submit_control_slurm_judges.sh       # validate without submitting
```

| Setting | Value |
|---|---|
| Script | `$MINE/submit_control_slurm_judges.sh` |
| Config | `examples/nemo_gym/nemotron-3.5-nano/rlvr_sc_slurm_judges.yaml` |
| Entrypoint | `./examples/run_grpo_single_controller.py` |
| Nodes | **82** = 32 train + 32 gen + 2 gym (hetgroup 0, 66) + 16 external (hetgroup 1) |
| `HOSTED_JUDGES` | `0` (default — do not set) |
| Steps / walltime | 10 / `04:00:00` |
| Account / partition | `nemotron_sw_post` / `batch` |

Judge tier — 68 GPUs, matched GPU-for-GPU to NVCF:

| Judge | Placement | Shape | Nodes |
|---|---|---|---|
| GenRM (Nemotron-3-Ultra-550B-A55B) | external pool | `GENRM_REPLICAS=6`, `TP=8` | 12 |
| nl2bash (Qwen3-235B-A22B-Instruct-2507-FP8) | external pool | `NL2BASH_REPLICAS=4`, `TP=4` | 4 |
| safety (Nemotron-Content-Safety-Reasoning-4B) | **in-Gym** `local_vllm_model` | `TP=1`, `DP=4` (`data_parallel_size_local=4`) | in the 2 gym nodes |

`JUDGE_MAX_NUM_SEQS=1024` applies to all three.

External pools are served by `$MINE/rlvr-allfeatures/tools/external_gym_vllm`
(`EXTERNAL_VLLM_TOOLS_DIR_HOST`) behind `vllm_pool_lb.py`. The launcher injects
`base_url` once the load balancer is healthy, which is what makes Gym skip
launching an in-process copy.

**Do not move nl2bash back into Gym.** Job 3656607 deadlocked doing exactly
that: the single-process Gym proxy accepted 594 concurrent judge requests and
forwarded none while the vLLM engines sat at Running 0 / Waiting 0, and ~2,000
rollouts queued behind the graders. `num_workers` cannot fix it, because a
`local_vllm_model` owns its engine and forking the server forks the engine.

---

## Arm B — Test, judges on NVCF (66 nodes)

```bash
cd /lustre/fsw/portfolios/nemotron/projects/nemotron_sw_post/users/sauramishra
bash submit_full_allfeatures_invocation.sh          # add hydra overrides as trailing args
DRY_RUN=1 bash submit_full_allfeatures_invocation.sh
```

`submit_full_allfeatures_invocation.sh` is a thin wrapper that sets the NVCF
endpoint and heartbeat, then `exec`s the real submitter with two overrides:

```bash
exec bash "$MINE/submit_full_allfeatures.sh" \
  "async_rl.rollout_failure.nemo_gym.rollout_timeout_s=2400.0" \
  "async_rl.stall_watchdog.stall_timeout_s=3000.0" \
  "$@"
```

| Setting | Value |
|---|---|
| Wrapper / submitter | `submit_full_allfeatures_invocation.sh` → `submit_full_allfeatures.sh` |
| Config | `examples/nemo_gym/nemotron-3.5-nano/rlvr_sc_nvcf_disagg.yaml` |
| Nodes | **66** = 32 train + 32 gen + 2 gym, `NUM_EXTERNAL_SERVICE_NODES=0` (enforced) |
| `HOSTED_JUDGES` | `1` — all three judges off-allocation |
| GenRM function UUID | `94b8435d-b749-41d5-a831-f1c1e0102870` |
| GenRM base URL | `http://127.0.0.1:1250/v1` (via sidecar) |

**Heartbeat: the h2-ping-sidecar, not the CRLF heartbeat.**
`H2_PING_SIDECAR=$MINE/h2-ping-sidecar/h2-ping-sidecar`, listening on
`127.0.0.1:1250`, `H2_PING_INTERVAL=60s`, upstream
`https://<uuid>.invocation.api.nvcf.nvidia.com`. The CRLF heartbeat (Gym PR
#3117) is deliberately **off**: it is cleaner on paper but measurably truncates
large judge responses, and the sidecar does not.

The sidecar is also the source of this arm's judge errors — 69 GenRM `status=502
kind=rate_limit` events carrying `http2: Transport received Server's graceful
shutdown GOAWAY`. All retried successfully at `try=1 max_tries=4`. Expect them;
they are not a failure.

---

## Smoke runs — already validated, only re-run after structural changes

Both smokes passed before the reference runs. Re-running them is only warranted
if you change topology, node counts, or the pool wiring.

| | Control smoke | NVCF smoke |
|---|---|---|
| Script | `submit_control_slurm_judges_smoke.sh` | `submit_smoke_allfeatures_invocation.sh` |
| Config | `rlvr_sc_slurm_judges_smoke.yaml` | `rlvr_sc_nvcf_disagg_smoke.yaml` |
| Nodes | **12** = 4 train + 2 gen + 2 gym + 4 external | **8** = 4 train + 2 gen + 2 gym |
| Steps | 2 | 2 |
| Judge shapes | `GENRM_REPLICAS=1`, `NL2BASH_REPLICAS=2`, `SAFETY_REPLICAS=1`, `SAFETY_DP_LOCAL=1` | hosted |

Smoke replica counts must keep `NUM_EXTERNAL_SERVICE_NODES` divisible by the
per-replica node demand (`TP / 4` GPUs-per-node). An earlier smoke failed this
check at `external=3`; 4 external nodes with 2 nl2bash replicas is the working
combination.

---

## Preflight

1. `$MINE/creds.env` must exist and be readable — supplies `NVIDIA_API_KEY`,
   `_LIVE_NVIDIA_API_KEY`, `_HEM_CREDS`. Override with `CREDS_FILE=...`.
   The control script hard-fails if it is missing.
2. Container present: `$MINE/containers/rl-gym.allfeatures-v2.sqsh` (~93 GB).
3. Verify the working tree still carries the 18 dirty files (see top section).
4. `DRY_RUN=1` first. It validates node arithmetic and pool registration.
5. Confirm assets under `$ASSETS=.../users/akamehra`: the three judge models and
   `evaluation/ultra_v3_reasoning_parser.py`.

Shared inputs:

- Policy: `.../users/kajalj/nvcf-poc-data/policy`
- Train + val data: `.../users/arnavk/nano_35_rlvr/train_data.jsonl`
- Sandbox image: `.../users/kajalj/3kruns/ultra-3k/images-striped/nemo-skills-sandbox-dc43f3e.sqsh`, `NO_COLOCATED_SANDBOX=0`
- Cache: `$MINE/cache`, `HF_HOME=$MINE/cache/hf_home`
- wandb: project `nemotron-3.5-nano`, entity `joc`

---

## Settings that are deliberate — do not "fix" these

| Setting | Value | Why |
|---|---|---|
| `rollout_timeout_s` / `stall_timeout_s` | **2400 / 3000 on NVCF only** | The arms are **not** matched here. This is the known parity gap behind control's 109 rollout timeouts vs NVCF's 10. Forcing one value would make the NVCF arm abandon slow-but-healthy hosted judgements as failures. Reproduces job 3657252 as submitted. |
| `JUDGE_MAX_NUM_SEQS` | 1024 | Matched across arms. |
| Sampler / lag | `ready_first`, lag 1 | Matched. |
| Global batch | 8k | Matched. |
| nl2bash topology | 4×TP4 (control) vs 8×TP2 (NVCF) | Same 16 GPUs. The pool wrapper gives each replica whole 4-GPU nodes, so TP2 is not expressible. TP4 leaves *more* KV per replica, so the tier is not handicapped — but it is the one place judge topology departs from NVCF's. |
| Gym `num_workers` fan-out | **not applied** | Reverted to restore parity. The fan-out work and its `is_nemo_gym_fastapi_entrypoint` guards live in the Gym submodule but are not enabled by these recipes. |
| Slurm reaper exemption | `REAPER_EXEMPT_MINS=120` | Trainer idle during the ~35–40 min judge cold start is the quantity being measured, not a fault. Without this the job gets reaped. |

---

## Monitoring

```
$MINE/runs/<EXP_NAME>/runs/latest/logs/exp_001/wandb/wandb/run-*/files/output.log
$MINE/runs/<EXP_NAME>/ray_logs/<job>-logs/ray-driver.log
$MINE/runs/<EXP_NAME>/ray_logs/<job>-logs/external_genrm/load_balancer.log   # control only
$MINE/runs/<EXP_NAME>/runs/latest/logs/nemo_gym/*.log
```

The load-balancer access log is the only per-request latency source, and it
exists for the control arm only — the NVCF arm has no equivalent, which is why
the report's NVCF latency figures come from a different instrument.

Helpers already written: `$MINE/watch_arm.sh`, `$MINE/monitor_invocation.sh`,
`$MINE/run_cohort_tests.sh`, `$MINE/judge_error_dist.sh`.

---

## Where the reasoning lives

The submit scripts carry long explanatory headers — `submit_control_slurm_judges.sh`
is 22 KB and most of it is rationale (NVCF shape matching, the nl2bash deadlock,
the reaper comment), and `submit_full_allfeatures_invocation.sh` documents the
heartbeat decision with the truncation measurements behind it. **Read those
headers before changing a topology knob**; this runbook indexes them rather than
replacing them.
