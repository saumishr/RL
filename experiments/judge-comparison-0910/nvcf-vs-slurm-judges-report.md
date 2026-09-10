# Slurm-hosted judges vs NVCF-hosted judges

**Nemotron 3.5 Nano RLVR, matched 10-step lag-1 runs**

Prepared 2026-09-10. Every figure below is measured from the two runs' own wandb history, their resolved `MasterConfig`, or their service logs. Derived quantities are labelled as such, and the places where a number does not exist are called out explicitly rather than estimated.

---

## Summary

The Slurm arm matched and beat the NVCF arm across ten identical optimizer steps.

| Outcome | Control (Slurm judges) | NVCF-hosted judges | Delta |
|---|---|---|---|
| Total step time | 6,185 s (103 min) | 6,691 s (112 min) | **−506 s (−7.6%)** |
| Wall-clock runtime | 6,209 s | 6,712 s | −503 s |
| Mean step time | 618.5 s | 669.1 s | −7.6% |
| Step-time stdev | **125.3 s** | 228.4 s | −45% |
| Worst step | 805.8 s | 1,064.8 s | −24% |
| Training throughput | **292.8 tok/s/GPU** | 275.6 tok/s/GPU | +6.2% |
| Mean exposed generation | **115.6 s** | 165.2 s | −30% |
| Mean reward | 0.7819 | 0.7890 | −0.0071 (noise) |
| Steps completed | 10 / 10 | 10 / 10 | — |

Three things are worth separating out.

**The headline is consistency, not per-step speed.** Control won only 4 of the 10 individual steps. Its aggregate lead comes from never having a bad step: NVCF's step-time standard deviation is 1.8× control's, and its single worst step (1,065 s) is 32% longer than control's worst.

**Reward is indistinguishable.** The 0.007 gap in mean reward is far smaller than the step-to-step scatter in either arm (control 0.63–0.92, NVCF 0.55–1.10). Ten steps is far too few to claim a convergence result, and this run was not designed to produce one. It answers throughput and stability.

**Control carried extra work and still won.** It logged 109 rollout timeouts against NVCF's 10, because of an unintended config difference (below). A rollout timeout here discards and redoes the prompt group rather than truncating it, so those were about 109 attempts that produced no committed data — roughly 2% extra work — and control finished 506 s ahead regardless. How much time that cost is not established; see the latency-independent discussion in the parity section.

**Judge latency is equivalent.** Slurm-hosted GenRM answers in 4.93 min at p50 and 7.32 min at p95, against NVCF-side figures of roughly 5 min and 8 min. Hosting the judges on Slurm cost nothing in per-request latency. See the caveats in the latency section: the two sets of numbers come from different instruments.

**Judge reliability favours Slurm outright.** Control's judges returned HTTP 200 on all 28,012 requests. NVCF's judges produced 73 HTTP 5xx and 5 connection errors. Both arms nonetheless lost zero prompts, because retries absorbed everything.

---

## Run identification

| | Control | NVCF |
|---|---|---|
| Slurm job | 3659144 (heterogeneous, 2 groups) | 3657252 |
| wandb run | `joc/nemotron-3.5-nano/1wmglhr1` | `joc/nemotron-3.5-nano/3gt4e0ai` |
| Run name | `slurmjudges-control-0910-0240` | `allfeatures-nvcf-invocation-0909-2343` |
| Nodes | 82 (66 Ray + 16 external) | 66 |
| Finished | 05:04:37, all 10 steps | 02:56, all 10 steps |

### Control arm topology

Verified against `node-allocation.txt` and each pool's vLLM launch arguments.

| Group | Nodes | Layout |
|---|---|---|
| Ray (train + generation + Gym) | 66 | 32 train, 32 generation, 2 Gym |
| External GenRM pool | 12 | 6 replicas × TP8 (2 nodes each) |
| External nl2bash pool | 4 | 4 replicas × TP4 (1 node each) |

Both pools sit behind `vllm_pool_lb` load balancers (GenRM on `:9213`, nl2bash on `:9214`). The launcher injects `base_url` once a balancer is healthy, which is what makes Gym skip launching an in-process copy. The safety judge remained an in-Gym `local_vllm_model` (TP1 × DP4).

In the NVCF arm all three judges are remote NVCF endpoints, so it needs no external node group.

---

## Config parity audit

Read from each run's resolved `MasterConfig` as printed to its driver log.

| Config key | Control | NVCF | Matched |
|---|---|---|---|
| `grpo.max_num_steps` | 10 | 10 | yes |
| `num_prompts_per_step` | 512 | 512 | yes |
| `num_generations_per_prompt` | 16 | 16 | yes |
| `train_global_batch_size` | 8192 | 8192 | yes |
| Sampler | `ready_first` | `ready_first` | yes |
| `max_staleness_versions` | 1 | 1 | yes |
| `max_inflight_prompts` | 1024 | 1024 | yes |
| Generation nodes × TP | 32 × TP4 | 32 × TP4 | yes |
| Generation `max_num_seqs` | 64 | 64 | yes |
| `data_plane.backend` | `mooncake_cpu` | `mooncake_cpu` | yes |
| `checkpointing.enabled` | false | false | yes |
| **Judge hosting** | external Slurm pools + in-Gym safety | NVCF endpoints | **no — intended** |
| **`rollout_timeout_s`** | **1800** | **2400** | **no — unintended** |

### The one unintended gap

Control ran `rollout_timeout_s=1800` while NVCF ran `2400`, because the NVCF invocation script passes the override and the control submit script does not. The consequence shows up directly in the redispatch counters:

| Counter | Control | NVCF |
|---|---|---|
| `rollout/redispatch_total/RolloutTimeout` | **109** | 10 |
| `rollout/gym_row_redispatch_total` | 63 | 60 |
| `rollout/dropped_prompt_groups` | 0 | 0 |
| `rollout/skipped_total` | 0 | 0 |
| `train/evicted_stale_prompt_groups` | 0 | 0 |

Row-level Gym retries are comparable (63 vs 60), so the 10× difference is specifically rollout-level timeouts.

### Why a shorter timeout is a cost, not a saving

It is natural to assume a shorter deadline helps by cutting slow requests short. It does not, because this timeout does not truncate a rollout and keep the partial result — it **discards and redoes** it. `RolloutTimeout` subclasses `RolloutInfraFailure`, and infra failures re-dispatch the whole prompt group onto a different generation shard. The deadline covers "one whole prompt-group rollout, re-dispatches included," so a 1800 s budget is 25% less envelope for the group plus its internal row retries.

The run confirms nothing was cut short: `dropped_prompt_groups`, `skipped_total` and `evicted_stale_prompt_groups` are all zero, every step assembled its full 512 groups, and committed totals are near-identical (control 5,212, NVCF 5,253). Control delivered the same data and additionally paid for about 109 attempts that produced nothing — roughly 2% extra work.

The one case where a shorter deadline genuinely helps is a *stuck* request rather than a slow one: if a rollout is hung, killing it at 1800 s recovers 600 s sooner than at 2400 s and the re-dispatch is a repair. Some fraction of the 109 may be of that kind; the logs do not distinguish stuck from merely slow.

Per-step, control's two timeout-heaviest steps are also its two slowest:

| Step | Control timeouts | Control step time | NVCF timeouts | NVCF step time |
|---|---|---|---|---|
| 4 | 12 | 480 s | 1 | 400 s |
| 5 | 9 | 567 s | 0 | 414 s |
| 6 | 14 | 595 s | 0 | 1,065 s |
| 7 | 10 | 585 s | 0 | 408 s |
| 8 | 9 | 595 s | 5 | 549 s |
| 9 | **34** | **738 s** | 3 | 857 s |
| 10 | **21** | **806 s** | 1 | 784 s |

That correlation is consistent with the timeouts costing time, but the causation plausibly runs the other way: a congested pipeline makes more rollouts breach 1800 s, so slowness may be producing the timeouts rather than the reverse. The correlation alone cannot separate the two directions.

**The defensible claim is therefore narrow.** The shorter timeout is a pure work tax with no offsetting benefit on the committed-data path, costing roughly 2% in extra attempts. Whether it materially slowed the run is not established, and aligning it to 2400 should be treated as removing a known confound rather than as a guaranteed speedup. It remains the one concrete fix to make before any rerun, because it is a parity defect regardless of its size.

---

## Per-step detail

| Step | Control step time | NVCF step time | Delta | Control tok/s/GPU | NVCF tok/s/GPU | Control reward | NVCF reward |
|---|---|---|---|---|---|---|---|
| 1 | 775 s | 745 s | +30 s | 53.3 | 67.7 | 0.643 | 0.626 |
| 2 | 411 s | 613 s | −202 s | 234.0 | 177.1 | 0.924 | 0.985 |
| 3 | 635 s | 857 s | −222 s | 452.2 | 384.8 | 0.683 | 0.655 |
| 4 | 480 s | 400 s | +80 s | 301.3 | 270.4 | 0.731 | 0.756 |
| 5 | 567 s | 414 s | +153 s | 401.4 | 257.3 | 0.861 | 0.843 |
| 6 | 595 s | 1,065 s | −470 s | 262.4 | 388.3 | 0.799 | 0.876 |
| 7 | 585 s | 408 s | +177 s | 315.0 | 239.1 | 0.858 | 0.546 |
| 8 | 595 s | 549 s | +46 s | 367.8 | 297.5 | 0.837 | 1.101 |
| 9 | 738 s | 857 s | −119 s | 256.2 | 292.9 | 0.853 | 0.890 |
| 10 | 806 s | 784 s | +22 s | 284.3 | 380.9 | 0.630 | 0.612 |

Negative delta favours control. Step 1 absorbs warmup in both arms, which is why its throughput is an order of magnitude below the rest.

Control's range is 411–806 s. NVCF's is 400–1,065 s, with a single 1,065 s excursion at step 6 that accounts for most of its aggregate deficit.

---

## Success and error rates

### Terminal outcomes: both arms lost nothing

| Counter | Control | NVCF |
|---|---|---|
| `rollout/committed_total` | 5,212 | 5,253 |
| `rollout/data_failures_total` | 0 | 0 |
| `rollout/infra_drops_total` | 0 | 0 |
| `rollout/max_consecutive_infra_drops` | 0 | 0 |
| `rollout/skipped_total` | 0 | 0 |
| `train/dropped_prompt_groups` | 0 | 0 |
| `train/evicted_stale_prompt_groups` | 0 | 0 |

**Eventual success was 100% in both arms.** No prompt was dropped, skipped or evicted, and every step assembled its full 512 groups. Neither arm logged a single retry-exhaustion or give-up marker at the rollout level.

That topline zero is not the same as "every error was retried away", and the distinction matters — see the retry-budget rows below. At the HTTP layer, 169 of NVCF's policy requests and 13 of control's used all three attempts without succeeding. Those requests did fail. What the zeros establish is that no *prompt group* was lost as a result, not that no request failed.

### Judge tier: the comparison that isolates the variable

This is the only error surface where the two arms actually differ by design, and it is one-sided.

| Judge-tier errors | Control | NVCF |
|---|---|---|
| HTTP 5xx | **0** | **73** (69 GenRM, 4 nl2bash) |
| Connection errors (`ClientOSError`) | **0** | 5 |
| Total retry events | **0** | 78 |
| Retry attempts that exhausted the budget | **0** | **0** |
| Requests with a measured status | 28,012, all HTTP 200 | not countable |

Control's judges returned **HTTP 200 on all 28,012 requests** — a measured error rate of 0.000%. NVCF's judges produced 73 HTTP 5xx and 5 connection errors. NVCF's are counts without a denominator, since its request total is unrecoverable, so no rate can be computed for it; but zero versus 78 needs no denominator to interpret.

In absolute terms NVCF's judge errors are mild, and they were genuinely retried away. Every one of GenRM's 69 carries `try=1 max_tries=4`, meaning all of them succeeded on the first retry with three attempts still in reserve. The only cost was the extra round trip's latency.

Their signature is specific and worth recording:

```
[model_retry url=http://127.0.0.1:1250/v1/chat/completions status=502 kind=rate_limit try=1 max_tries=4
 error_msg=h2-ping-sidecar: upstream error: http2: Transport: cannot retry err
 [http2: Transport received Server's graceful shutdown GOAWAY] after Request.Body was written]
```

Those are not vLLM errors. They come from NVCF's HTTP/2 front-end sidecar issuing a graceful-shutdown `GOAWAY` — the proxy recycling connections underneath in-flight requests — and the client classifies them `kind=rate_limit`. This is a property of the NVCF ingress path that the control arm's direct-to-load-balancer path structurally cannot produce.

### Policy tier: larger gap, but not attributable to judge hosting

| Policy-tier errors | Control | NVCF | Ratio |
|---|---|---|---|
| HTTP 5xx log lines | 84 | **1,067** | 12.7× |
| — distinct failing requests (`try=1`) | 51 | **556** | 10.9× |
| — needed a second attempt (`try=2`) | 20 | 342 | 17.1× |
| — **exhausted the budget (`try=3 max_tries=3`)** | **13** | **169** | 13.0× |
| Connection errors (`ClientOSError`) | 265 | 286 | 1.1× |
| Total retry events | 360 | **1,369** | 3.8× |

All of these are `status=500 kind=server_error` with `error_msg=Internal Server Error` in both arms — the same failure mode, differing only in volume.

The retry-budget row is the one that carries weight. Policy requests get `max_tries=3`, and **169 of NVCF's reached the third attempt and still failed**, against 13 of control's. Unlike the judge 5xx, these were not retried away. The rollout counters show no prompt group was lost as a result, but the counters do not reveal what became of those requests — whether they were re-issued higher up, or produced degraded trajectories that still committed. That is an open question, not a resolved one.

NVCF's arm logged 12.7× more policy-side 5xx. **This cannot be caused by judge hosting**, because the policy model is served identically in both arms — 32 local Slurm generation nodes, same TP, same `max_num_seqs`. The most plausible link is workload: NVCF's arm generated much longer sequences (13,327 tokens/sequence at peak against control's 9,255), which stresses the generation engines harder. That is a hypothesis consistent with the sequence-length correlations reported under Concurrency, not a demonstrated cause.

Also of note, NVCF logged 72 HTTP 502 and 1 HTTP 504 while control logged **none**. Gateway status codes of that kind come from a proxy or front-end layer, which the control arm's direct-to-load-balancer path does not have.

### Rollout level: control's only deficit

| Redispatch counter | Control | NVCF |
|---|---|---|
| `rollout/redispatch_total` | **109** | 12 |
| — `RolloutTimeout` | 109 | 10 |
| — `RayTaskError(GymTransportError)` | 0 | 2 |
| `rollout/gym_row_redispatch_total` | 63 | 60 |

Control's 109 against NVCF's 12 is the one error metric where control is worse, and it is a config artifact rather than a reliability difference — see the timeout discussion in the parity section.

### Net reading

Every arm-versus-arm error comparison either favours control or is explained away:

- **Terminal loss:** tied at zero. No prompt group lost in either arm.
- **Judge errors:** control 0, NVCF 78, all of NVCF's retried away on the first attempt. Control wins on the dimension under test, though NVCF's judge tier was not a material problem in absolute terms.
- **Policy errors:** control 51 distinct failures, NVCF 556, with 13 versus 169 exhausting the retry budget. Both arms serve the policy identically, so this reflects workload rather than hosting — but it is the largest unexplained asymmetry in the comparison.
- **Rollout redispatch:** control 109, NVCF 12 — control worse, caused by its own shorter timeout.

So the answer to "did NVCF have a high error rate that retries hid?" is: partly. Its *judge* tier — the thing being tested — had a small, cleanly-retried error rate against control's exact zero. The large error count sits in the *policy* tier, which is not NVCF-hosted, and a sixth of those exhausted their retries rather than being absorbed.

Counting caveat: HTTP and connection figures count matching log *lines* across each run's `nemo_gym/*.log` files. Because a retried request logs once per attempt, the `try=N` rows above are the trustworthy way to read them: `try=1` approximates distinct failing requests, and the top-line 5xx figure sums all attempts. The wandb counters in the first and last tables are exact.

---

## Request latency

### Headline: GenRM latency is equivalent across the two arms

| Percentile | NVCF-hosted GenRM | Slurm-hosted GenRM |
|---|---|---|
| p50 | ~5 min | **4.93 min** (296 s) |
| p90 | not supplied | **6.68 min** (401 s) |
| p95 | ~8 min | **7.32 min** (439 s) |
| p99 | not supplied | 8.57 min (514 s) |
| Max | not supplied | 14.23 min (854 s) |
| Mean | not supplied | 5.03 min (302 s) |

The p50 figures agree to within the 1-second resolution of the access log and the rounding of the NVCF figure — **the two judge tiers are performing the same**. At p95 Slurm is about 40 seconds faster, which should be held loosely given the NVCF figure is rounded; the fair reading is that p95 is comparable as well, perhaps marginally in Slurm's favour.

For a sense of the shape of the distribution: 47.5% of Slurm GenRM requests exceed 5 minutes but only 2.1% exceed 8 minutes. The distribution is tightly packed around its median with a thin tail, not bimodal.

Two caveats on this comparison, both of which matter for how much weight it can carry.

**Provenance differs.** The Slurm figures are measured from our own load balancer access logs, request by request, across all 11,842 requests. The NVCF figures were supplied from NVCF-side telemetry and are not reproducible from anything in our logs. They are not the same kind of measurement, and the two were not collected by the same instrument.

**Measurement point differs.** The Slurm numbers are server-side, taken at the load balancer in front of the pool, so they exclude network time between Gym and the balancer. If the NVCF figures are client-side, they include round-trip network to NVCF, which would make Slurm's true client-observed latency slightly worse than shown and NVCF's slightly better than its server-side equivalent.

**Scope matters.** The comparison above assumes the NVCF figures describe GenRM specifically. If they instead cover all judge traffic, the conclusion changes sharply: pooled across both tiers the Slurm p50 is 5 seconds, not 5 minutes, because nl2bash handles more requests than GenRM and answers in about a second. A ~5 min pooled p50 on the NVCF side would then mean NVCF is dramatically slower on the light tier. Worth confirming which scope the NVCF dashboard reports.

### Latency is not measurable for NVCF from our side

The NVCF figures above came from outside this run's artifacts, and that limitation is worth recording. The control arm's judges sit behind a load balancer that writes an access log, so every request's latency is recoverable. NVCF's judges are remote endpoints with no access log on our side, and Gym disables its own uvicorn access logging to keep 200 OKs out of the logs. **No latency percentile can be produced for the NVCF arm from our logs, at any request tier.**

The only timings recorded anywhere in the NVCF arm are on the retry path, which fires only after a request has already failed:

| NVCF retry-path timings | Value |
|---|---|
| Events with a recorded time, whole run | 311 |
| p50 time-to-failure | 16.1 s |
| p90 time-to-failure | 132.9 s |
| Max time-to-failure | 703.6 s |
| Largest single source | `policy_model.log` (244 of 311) |

These describe how long failing requests took before giving up, and 244 of the 311 come from policy generation rather than from judges. They are not comparable to the numbers below.

### Control arm judge latency

Parsed from `external_genrm/load_balancer.log` and `external_nl2bash/load_balancer.log`. Latency is response-write time minus request-receive time, so resolution is 1 second. All 28,012 parsed requests returned HTTP 200.

| Judge tier | Requests | p50 | p90 | p95 | p99 | Max | Mean | Median response size |
|---|---|---|---|---|---|---|---|---|
| **GenRM** (6 × TP8) | 11,842 | **296 s** | **401 s** | 439 s | 514 s | 854 s | 302 s | 34 KB |
| **nl2bash** (4 × TP4) | 16,170 | **1 s** | **6 s** | 9 s | 14 s | 155 s | 2 s | 792 B |
| All judge requests pooled | 28,012 | 5 s | 352 s | 392 s | 475 s | 854 s | 129 s | — |

The pooled row is included only to show why pooling misleads here: nl2bash serves more requests than GenRM and answers in about a second, so the combined p50 collapses to 5 seconds and describes neither tier. Compare tiers individually.

**GenRM is roughly 300× slower per request than nl2bash**, and at a 618 s mean step time a single GenRM verdict consumes about half the step. It returns 34 KB per response against nl2bash's 792 bytes, so the gap is tokens generated rather than protocol overhead. GenRM latency, not judge hosting, is the dominant term in this workload.

### Independent cross-check

Little's Law provides a check from unrelated data. 11,842 GenRM requests over 6,185 s of step time is 1.91 requests/s, and the pool held about 600 concurrent requests (6 replicas at a measured p50 of 100 each). That implies a mean latency of **313 s**, against the **301.7 s** mean and 296 s p50 read straight from the access log — agreement within 4%.

---

## Concurrency

### Rollout level — directly comparable

`rollout/inflight` is logged identically in both arms, making it the one concurrency measure that compares directly. One in-flight unit is a prompt group: one prompt with 16 generations.

In-flight prompt groups, warmup step excluded:

| | Control | NVCF | Delta |
|---|---|---|---|
| p50 | **455** | 435 | +20 |
| p90 | 493 | **499** | −7 |
| Peak | 499 | **501** | −2 |
| Min | **416** | 261 | +155 |
| Mean | **453.4** | 415.2 | +38 |
| Stdev | **29.8** | 87.2 | −57 |
| Configured cap (`max_inflight_prompts`) | 1024 | 1024 | — |
| Implied outstanding sequences (mean) | ~7,255 | ~6,644 | — |
| Engine running-sequence ceiling | 2,048 | 2,048 | — |

**The two arms are level at the top and differ at the floor.** NVCF edges control at p90 and peak, by 7 and 2 groups respectively, which is noise. Control's advantage is that it never drops below 416 while NVCF sinks to 261: its mean is 9% higher not because its ceiling is higher but because NVCF's mean is dragged down by three low steps (261, 327, 362). Control's standard deviation is a third of NVCF's, the same consistency pattern seen in step time.

**Neither arm is prompt-admission capped** — both sit at roughly 44% of the 1,024 cap.

### Is NVCF's lower floor a capacity problem?

No — it tracks sequence length, which is workload rather than infrastructure. Low in-flight is ambiguous on its own and has to be read alongside `exposed_generation`, the time the trainer sat idle waiting for rollouts.

| NVCF step | In-flight | Exposed gen | Step time | Tokens/seq | Reading |
|---|---|---|---|---|---|
| 6 | **261** | 399 s | 1,065 s | **13,327** | Starved |
| 4 | 327 | 0.2 s | **400 s** | 3,484 | Surplus — buffer full |
| 3 | 362 | 240 s | 857 s | 10,630 | Starved |
| 10 | 379 | 175 s | 784 s | 9,631 | Starved |

Step 4 has NVCF's second-lowest in-flight together with its fastest step and no idle time at all: concurrency was low because rollouts arrived faster than training consumed them, so the pump throttled. That is a healthy state, not a degraded one.

The three genuinely starved steps are precisely NVCF's three longest-sequence steps. For NVCF, `corr(tokens-per-sequence, step-time) = +0.95` and `corr(tokens-per-sequence, exposed-generation) = +0.86`. The policy generated longer rollouts on those steps, each held its slot longer, and the trainer waited. With 95% of NVCF's step-time variance explained by token volume, little is left to attribute to the endpoint. An NVCF contribution cannot be formally excluded — there is no NVCF-side telemetry — but the workload signal does not require one.

Two observations cut against a purely benign reading, and they partly offset each other:

**NVCF drew a rougher workload.** Its tokens-per-sequence spread is 3,143–13,327 (4.2×) against control's 3,097–9,255 (3.0×). Both arms ran the same recipe, data and seed, so this is stochastic divergence after step 1 rather than an effect of hosting. It does flatter control's step-time comparison.

**Control's step time is less explained by sequence length** (`corr = +0.62` versus NVCF's 0.95). That residual variance is consistent with control carrying overhead NVCF did not, most plausibly its 109 redispatched rollouts.

Two caveats on the percentile figures. First, the p50 comparison is sensitive to the warmup step: including step 0, control's p50 is 449 against NVCF's 454, marginally reversing the ordering, because control's step 0 reads 226 against NVCF's 473. Second, and more important, **these are not distributions.** `rollout/inflight` is logged exactly once per optimizer step, so each arm contributes 10 instantaneous readings and nothing in the driver logs samples it more finely. A p90 drawn from 10 points is interpolation near the second-highest value, and none of it is time-weighted. Treat the p50 and mean as indicative and the p90 as barely resolved.

The engine ceiling is 2,048 concurrently running sequences in both arms (32 generation engines × `max_num_seqs=64`), identical by construction. Since outstanding work averages ~7,000 sequences, roughly three-quarters of it is queued at any moment. `max_num_seqs=64` is therefore the binding constraint, and raising it is the throughput lever worth testing next.

### Engine level — measured for control only

From `Running:` / `Waiting:` counts in each pool's `vllm_*.log` (3,727 GenRM samples, 2,465 nl2bash samples). Per-replica figures.

| Pool | Replicas | Running p50 | Running p90 | Running peak | Waiting p50 | Waiting peak | Aggregate concurrent |
|---|---|---|---|---|---|---|---|
| GenRM | 6 | 100 | 150 | 150 | 0 | 13 | ~600 |
| nl2bash | 4 | **2** | 8 | 35 | 0 | 5 | ~8 |
| Safety (in-Gym) | 1 × DP4 | — | — | **3** | 0 | 0 | ~3 |

**The judge tiers are sized backwards.** GenRM runs 100 concurrent requests per replica with its waiting queue at zero — saturated but not backed up, simply slow per request. nl2bash, which received four whole TP4 nodes after the deadlock described below, runs at a measured p50 of **2 concurrent requests per replica**. The in-Gym safety judge peaks at 3 concurrent with zero waiting across 590 samples. The 16 GPUs spent on nl2bash bought reliability, not utilization, and are the obvious place to reclaim capacity for GenRM.

For the policy generation engines, engine-level concurrency is unknown in **both** arms: they do not emit vLLM's usual `Running:` stat lines, because `enable_vllm_metrics_logger` replaces the default stdout logger, and no running-sequence gauge reaches wandb.

---

## Tokens per second, per GPU and per request

Derived per step from `train/global_valid_toks`, `timing/train/total_step_time`, `rollout/inflight` and `timing/train/valid_tokens_per_sec_per_gpu`, steps 2–10 of each run, then averaged. Token counts are training-batch tokens (prompt plus response), not pure decode output.

| Metric | Control | NVCF | Delta |
|---|---|---|---|
| tok/s/GPU (wandb training metric) | **292.8** | 275.6 | +6.2% |
| Aggregate tokens/s across the job | **84,320** | 78,860 | +6.9% |
| Tokens/s per generation GPU | **658.7** | 616.1 | +6.9% |
| Tokens per sequence, mean | 6,205 | 6,723 | −7.7% |
| Tokens per sequence, peak | 9,255 | 13,327 | −30.6% |
| tok/s per outstanding request, p50 | **11.2** | 9.8 | +14.3% |
| tok/s per outstanding request, mean | 11.7 | **12.9** | −8.8% |
| tok/s per running engine slot | **41.2** | 38.5 | +6.9% |

Control leads on every aggregate view. Per request the answer depends on the denominator: dividing by outstanding sequences, control leads at p50 but trails on the mean, because NVCF's per-step token volume is more skewed — its tokens-per-sequence peaks at 13,327 against control's 9,255, dragging its mean up through a few very long steps.

The practical reading is that **per-request token rates sit within about 9% of each other in either direction**. No view of the data shows either arm at anything like half the other's per-request throughput.

Where a per-request decode rate was measured directly rather than derived — the GenRM pool's own engines — it comes out at **30.3 tok/s/request**, with nl2bash at 22.6 tok/s/request.

### Per-request rate distributions, control only

Generation throughput divided by running request count, per vLLM stat sample. This is a genuine distribution rather than a per-step average.

| Pool | Samples | p10 | p50 | p90 | p99 | Max | Mean |
|---|---|---|---|---|---|---|---|
| GenRM | 3,727 | 25.8 | **31.2** | **37.2** | 45.1 | 94.8 | 31.2 |
| nl2bash | 985 | 14.2 | **45.0** | **133.9** | 269.4 | 501.4 | 62.5 |

GenRM's distribution is tight — p10 to p90 spans only 25.8 to 37.2 tok/s — which is what an engine held at 100 concurrent requests with no queue should look like.

nl2bash's spread is far wider, and the high tail indicates *under*-utilization rather than efficiency: at a p50 of 2 concurrent requests, a lone request occasionally has a whole TP4 engine to itself and decodes at roughly 500 tok/s. Wide dispersion here is a symptom of an idle pool.

### Completed requests per minute, control only

Counted from load balancer access log completion timestamps, grouped by wall-clock minute.

| Pool | Minutes observed | Total requests | p50 | p90 | p99 | Peak |
|---|---|---|---|---|---|---|
| GenRM | 104 | 11,842 | **108** | **179** | 226 | 234 |
| nl2bash | 107 | 16,170 | **121** | **289** | 497 | 529 |

Neither of these two distributions has an NVCF counterpart. No vLLM logs means no per-request rate; no access logs means no completion rate. NVCF's GenRM logs contain only 69 failure records and no per-request entries, so even its request *count* is unrecoverable.

A per-step average can be constructed for NVCF only by assuming it issues the same 1,184 GenRM requests per step that control measurably did, giving a derived p50 of 106 and p90 of 175 against control's derived 119 and 151. **That derivation should not be used to judge NVCF.** It evaluates to 71,052 ÷ step-time, so it carries no information independent of the step-time comparison already presented, and the equal-request-count premise is exactly what would need verifying. Its means agreeing within 1% is a tautology, not a result.

As a magnitude check on the measured figures, control's measured per-minute p50 of 108 and its derived per-step figure of 119 differ by about 10% — the gap between counting real minutes, including ramp and idle ones, and assuming uniform arrival within a step.

---

## Why the control arm's judges are laid out this way

The preceding attempt, job 3656607, deadlocked and had to be cancelled. nl2bash's single-process in-Gym proxy accepted 594 concurrent judge requests and forwarded none of them: the vLLM engines behind it sat at `Running: 0 / Waiting: 0` while roughly 2,000 rollouts queued behind the graders, and the Gym `aiohttp` connector pool reached its 4,096-connection limit. Raising `num_workers` cannot fix this, because a `local_vllm_model` owns its engine and forking the server forks the engine.

Moving nl2bash to four independent TP4 external pool replicas behind the same load balancer GenRM already used removed the wedge. This run confirms the fix: 11 refits across 10 steps, no stall, no dropped prompt groups, and every step assembling its full 512 groups.

nl2bash's GPU budget is unchanged at 16. What changed is its shape: four independent DP=1 replicas, each owning a whole 4-GPU node, rather than one 16-GPU in-process deployment (`rlvr.yaml` sizes it TP4 × DP4). Whole-node replicas are a constraint of the external pool wrapper, not a tuning choice, and it cannot be avoided without teaching the wrapper to pack replicas per node.

How NVCF serves the same judges internally is not observable from our side, so no claim is made here about its judge topology.

---

## What this report does not establish

**Convergence.** Ten steps with reward scatter of 0.55–1.10 supports no convergence claim in either direction. The arms' mean rewards differ by 0.007, which is noise at this sample size.

**NVCF request latency, from our own artifacts.** Unmeasurable from our side, for the reasons given above. The NVCF p50 and p95 quoted in this report were supplied from NVCF-side telemetry; they are not reproducible from this run's logs, were not collected by the same instrument as the Slurm figures, and carry no stated sample size or measurement point. The conclusion that GenRM latency is equivalent across arms rests on that externally supplied pair of numbers agreeing with our measured distribution, which is suggestive but is not a controlled comparison.

**Engine-level generation concurrency.** Unknown in both arms. The ceiling of 2,048 is a config fact; how much of it was occupied at any instant was never logged.

**Resumability.** `checkpointing.enabled` is false in both arms and both `checkpoints/` directories are empty, so neither run can be extended from where it stopped. Enabling it on the single-controller path with a `ready_first` sampler additionally requires `checkpointing.save_data_plane=true`, or startup validation rejects the config.

---

## Recommended next actions

1. **Fix the timeout parity gap.** Pass `rollout_timeout_s=2400` from `submit_control_slurm_judges.sh` so the control arm stops paying for ~100 redundant rollouts per run.
2. **Rebalance the judge budget.** nl2bash and safety are provisioned for roughly 50× the concurrency they use. Moving GPUs from nl2bash to GenRM targets the one tier that is actually saturated and that consumes about half of every step.
3. **Test a higher generation `max_num_seqs`.** At 64 per engine the ceiling is 2,048 running sequences against ~7,000 outstanding, so generation is admission-limited in both arms. This is the clearest available throughput lever.
4. **Enable checkpointing** (with `save_data_plane=true`) if any future run may need to be extended or resumed.
5. **Run longer for convergence.** If a convergence claim is the goal, both arms need materially more than 10 steps, launched with the step budget set correctly from the start since it is fixed at launch.
