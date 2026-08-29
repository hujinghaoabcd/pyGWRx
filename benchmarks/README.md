# Architecture performance / memory baselines

This directory stores **observational** baselines captured before the pyGWRx 0.2 architecture rewrite.

The baseline harness lives at `tools/benchmarks/architecture_baseline.py`. It runs each scenario in a fresh Python process and records:

- wall-clock time;
- process CPU time;
- peak Python-tracked allocations (`tracemalloc`);
- process peak RSS and the increase above the post-import starting point when the platform exposes it;
- unique directly retained `n × n` NumPy buffers, grouping aliasing attributes, and their total size;
- scenario-specific structural checks and small numerical fingerprints.

## Required A4 scenarios

The harness covers the architecture constitution's minimum set and one extra GWR dense-hat comparison:

1. `gwr_manual_streaming` — manual fixed bandwidth, default streaming-style fit, no retained hat matrix;
2. `gwr_manual_hat` — same scale of manual GWR with a retained dense hat matrix;
3. `gwr_auto_bandwidth` — automatic adaptive CV bandwidth selection;
4. `mgwr_backfit` — representative repeated multiscale backfit;
5. `gtwr_fit` — representative spatiotemporal fit;
6. `mgtwr_small` — small representative multiscale spatiotemporal backfit;
7. `sgwr_weight_heavy` — SGWR with dense weight retention enabled;
8. `scalable_gwr_large_n` — large-n cKDTree/compressed ScaGWR path.

## Important: not a timing gate

`architecture_baseline.json` is a comparison record, **not a CI pass/fail timing threshold**. GitHub-hosted runner hardware and background load vary. Future refactor PRs should compare large regressions with the same harness and investigate them, but must not make statistical or architectural changes merely to satisfy a noisy wall-time number.

Structural regressions are different: for example, ScalableGWR acquiring an `n × n` retained matrix or streamed GWR unexpectedly retaining one is meaningful evidence and should be investigated independently of timing noise.

To regenerate manually:

```bash
python tools/benchmarks/architecture_baseline.py --output benchmarks/architecture_baseline.json
```

For a subset, repeat `--scenario`, for example:

```bash
python tools/benchmarks/architecture_baseline.py \
  --scenario gwr_manual_streaming \
  --scenario scalable_gwr_large_n \
  --output /tmp/pygwrx-baseline.json
```
