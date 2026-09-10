# judge-comparison-0910

Snapshot of the NVCF-vs-Slurm judge-hosting comparison, so the two arms can be
re-run without repeating the smoke runs, the topology tuning, or the deadlock
debugging.

| File | What it is |
|---|---|
| `RUNBOOK-judge-comparison.md` | How to re-run either arm. Start here. |
| `nvcf-vs-slurm-judges-report.md` | Results and analysis of the two reference runs. |
| `ledger.tsv` | Experiment ledger, three rows. |
| `submit/` | Snapshot copies of the launch scripts. |

The recipe itself is commit `547ff28eeafe6edd742354b6adfcf86af331e227` on
`rlvr-convergence-allfeatures`, tagged `judge-comparison-0910`. Nothing is
pushed; this filesystem is the only copy.

## Two caveats on this snapshot

**`submit/` is a copy, not the source of truth.** The live scripts are in
`/lustre/.../users/sauramishra/*.sh` and that is where you should edit and
launch from. Re-copy them here when they change, or the two will drift.

**Ledger row 1 is not reproducible.** Job 3656607 — the run that deadlocked
with nl2bash in-Gym — ran on parent `7f2947ae` plus a *different* set of
uncommitted edits than the two runs that followed. Externalising nl2bash was an
uncommitted change made between that job and job 3659144, so no commit captures
row 1's tree. It is recorded for the reasoning, not for replay; the deadlock
itself is described in the commit message of `547ff28` and in the runbook.

Rows 2 and 3 both correspond to `547ff28`.
