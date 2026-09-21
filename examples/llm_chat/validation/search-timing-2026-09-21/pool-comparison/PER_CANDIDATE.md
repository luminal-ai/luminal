# Per-candidate and per-trial search timing

Expanded from the saved 2026-09-21 baseline and final pooled traces; no new benchmark was run. Llama3-8B FP32 on H200; two generations of four attempts, one warmup and three timed trials per unique plan. All numbers are wall-clock seconds. Candidate numbering is generation.candidate, one-based.

Profiling shape is q=1, c=1, with planned capacity q=1..8, c=1..2048. These timings describe search profiling rather than a 128-input/128-output token generation benchmark. Trials include CPU staging, GPU transfers/kernels/synchronization, output unpacking, and profiling bookkeeping. GPU replay is not a kernel-only measurement.

The pooled run uses the experimental retained pinned-host CUDA pool with GPU READWRITE access. Both cached candidates (2.1 and 2.3) reuse candidate 1.1. Generation 2 differs between runs because measured rankings affect subsequent mutations; rows with the same attempt number are not necessarily the same plan.

Candidate totals exclude initial graph assembly/search analysis and final winner validation. Warmup includes compilation. Prepare includes plan installation and allocation/zeroing. Extract + plan includes fingerprinting. Other is the exact remaining wall time, so unrounded columns reconcile to each row total.

## Pooled candidate phases

| Candidate | Extract + plan | Prepare | Compile + warmup | Trial 1 | Trial 2 | Trial 3 | Release | Other | Total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.1 | 1.756 | 5.620 | 8.105 | 2.965 | 2.945 | 2.941 | 0.142 | 0.022 | 24.495 |
| 1.2 | 0.350 | 0.498 | 4.077 | 2.964 | 2.965 | 2.964 | 0.092 | 0.026 | 13.936 |
| 1.3 | 0.315 | 0.494 | 3.466 | 2.974 | 2.980 | 2.971 | 0.092 | 0.026 | 13.317 |
| 1.4 | 0.310 | 0.501 | 3.241 | 2.975 | 2.965 | 2.921 | 0.101 | 0.027 | 13.040 |
| 2.1 (cached 1.1) | 0.218 | 0.000 | 0.000 | — | — | — | 0.000 | 0.015 | 0.233 |
| 2.2 | 0.275 | 0.540 | 2.974 | 2.954 | 2.964 | 2.943 | 0.144 | 0.027 | 12.821 |
| 2.3 (cached 1.1) | 0.215 | 0.000 | 0.000 | — | — | — | 0.000 | 0.015 | 0.229 |
| 2.4 | 0.276 | 0.544 | 2.981 | 2.973 | 2.969 | 2.969 | 0.145 | 0.027 | 12.884 |

## Pooled execution details

Compilation (including graph rebuild), address rebind, staging, replay, and unpack columns use disjoint trace spans. Other includes profiling cleanup/bookkeeping and uninstrumented gaps. No nested spans are double-counted.

| Candidate | Execution | Compile + graph | Address rebind | CPU staging | GPU replay + sync | CPU unpack | Other | Total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.1 | warmup | 5.133 | 0.000 | 1.840 | 0.813 | 0.291 | 0.028 | 8.105 |
| 1.1 | trial 1 | 0.000 | 0.009 | 1.820 | 0.813 | 0.297 | 0.026 | 2.965 |
| 1.1 | trial 2 | 0.000 | 0.009 | 1.826 | 0.813 | 0.270 | 0.027 | 2.945 |
| 1.1 | trial 3 | 0.000 | 0.009 | 1.821 | 0.813 | 0.272 | 0.027 | 2.941 |
| 1.2 | warmup | 1.131 | 0.000 | 1.819 | 0.800 | 0.299 | 0.028 | 4.077 |
| 1.2 | trial 1 | 0.000 | 0.009 | 1.825 | 0.801 | 0.301 | 0.028 | 2.964 |
| 1.2 | trial 2 | 0.000 | 0.009 | 1.826 | 0.801 | 0.301 | 0.029 | 2.965 |
| 1.2 | trial 3 | 0.000 | 0.009 | 1.825 | 0.801 | 0.302 | 0.028 | 2.964 |
| 1.3 | warmup | 0.497 | 0.000 | 1.844 | 0.794 | 0.303 | 0.028 | 3.466 |
| 1.3 | trial 1 | 0.000 | 0.009 | 1.841 | 0.794 | 0.302 | 0.028 | 2.974 |
| 1.3 | trial 2 | 0.000 | 0.009 | 1.849 | 0.794 | 0.300 | 0.028 | 2.980 |
| 1.3 | trial 3 | 0.000 | 0.009 | 1.838 | 0.794 | 0.302 | 0.028 | 2.971 |
| 1.4 | warmup | 0.281 | 0.000 | 1.822 | 0.808 | 0.302 | 0.028 | 3.241 |
| 1.4 | trial 1 | 0.000 | 0.009 | 1.828 | 0.807 | 0.302 | 0.028 | 2.975 |
| 1.4 | trial 2 | 0.000 | 0.009 | 1.821 | 0.807 | 0.304 | 0.024 | 2.965 |
| 1.4 | trial 3 | 0.000 | 0.009 | 1.826 | 0.807 | 0.253 | 0.025 | 2.921 |
| 2.2 | warmup | 0.031 | 0.000 | 1.821 | 0.813 | 0.281 | 0.028 | 2.974 |
| 2.2 | trial 1 | 0.000 | 0.009 | 1.821 | 0.813 | 0.283 | 0.029 | 2.954 |
| 2.2 | trial 2 | 0.000 | 0.009 | 1.829 | 0.813 | 0.287 | 0.027 | 2.964 |
| 2.2 | trial 3 | 0.000 | 0.009 | 1.821 | 0.813 | 0.273 | 0.027 | 2.943 |
| 2.4 | warmup | 0.030 | 0.000 | 1.824 | 0.813 | 0.286 | 0.028 | 2.981 |
| 2.4 | trial 1 | 0.000 | 0.009 | 1.828 | 0.813 | 0.295 | 0.028 | 2.973 |
| 2.4 | trial 2 | 0.000 | 0.009 | 1.825 | 0.812 | 0.293 | 0.029 | 2.969 |
| 2.4 | trial 3 | 0.000 | 0.009 | 1.824 | 0.812 | 0.295 | 0.029 | 2.969 |

## Baseline candidate phases

| Candidate | Extract + plan | Prepare | Compile + warmup | Trial 1 | Trial 2 | Trial 3 | Release | Other | Total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.1 | 1.753 | 17.018 | 8.027 | 2.917 | 2.882 | 2.883 | 6.400 | 0.022 | 41.901 |
| 1.2 | 0.346 | 17.078 | 4.031 | 2.885 | 2.886 | 2.891 | 6.427 | 0.028 | 36.572 |
| 1.3 | 0.312 | 17.054 | 3.403 | 2.914 | 2.914 | 2.913 | 6.390 | 0.028 | 35.928 |
| 1.4 | 0.307 | 17.139 | 3.186 | 2.912 | 2.908 | 2.874 | 6.428 | 0.030 | 35.785 |
| 2.1 | 0.278 | 17.049 | 2.888 | 2.869 | 2.868 | 2.870 | 6.386 | 0.035 | 35.242 |
| 2.2 | 0.280 | 17.059 | 2.927 | 2.902 | 2.921 | 2.923 | 6.455 | 0.027 | 35.494 |
| 2.3 | 0.280 | 17.016 | 2.926 | 2.902 | 2.906 | 2.903 | 6.421 | 0.028 | 35.381 |
| 2.4 | 0.282 | 17.006 | 2.916 | 2.909 | 2.901 | 2.905 | 6.430 | 0.029 | 35.378 |

## Baseline execution details

Compilation (including graph rebuild), address rebind, staging, replay, and unpack columns use disjoint trace spans. Other includes profiling cleanup/bookkeeping and uninstrumented gaps. No nested spans are double-counted.

| Candidate | Execution | Compile + graph | Address rebind | CPU staging | GPU replay + sync | CPU unpack | Other | Total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.1 | warmup | 5.120 | 0.000 | 1.765 | 0.812 | 0.302 | 0.028 | 8.027 |
| 1.1 | trial 1 | 0.000 | 0.009 | 1.766 | 0.812 | 0.303 | 0.026 | 2.917 |
| 1.1 | trial 2 | 0.000 | 0.009 | 1.763 | 0.813 | 0.272 | 0.027 | 2.882 |
| 1.1 | trial 3 | 0.000 | 0.009 | 1.762 | 0.812 | 0.274 | 0.026 | 2.883 |
| 1.2 | warmup | 1.154 | 0.000 | 1.768 | 0.800 | 0.281 | 0.028 | 4.031 |
| 1.2 | trial 1 | 0.000 | 0.009 | 1.765 | 0.800 | 0.283 | 0.028 | 2.885 |
| 1.2 | trial 2 | 0.000 | 0.009 | 1.765 | 0.800 | 0.284 | 0.028 | 2.886 |
| 1.2 | trial 3 | 0.000 | 0.009 | 1.769 | 0.800 | 0.284 | 0.028 | 2.891 |
| 1.3 | warmup | 0.496 | 0.000 | 1.799 | 0.794 | 0.285 | 0.028 | 3.403 |
| 1.3 | trial 1 | 0.000 | 0.009 | 1.800 | 0.794 | 0.283 | 0.028 | 2.914 |
| 1.3 | trial 2 | 0.000 | 0.008 | 1.800 | 0.794 | 0.283 | 0.028 | 2.914 |
| 1.3 | trial 3 | 0.000 | 0.008 | 1.799 | 0.794 | 0.284 | 0.028 | 2.913 |
| 1.4 | warmup | 0.281 | 0.000 | 1.785 | 0.807 | 0.285 | 0.028 | 3.186 |
| 1.4 | trial 1 | 0.000 | 0.009 | 1.784 | 0.807 | 0.284 | 0.028 | 2.912 |
| 1.4 | trial 2 | 0.000 | 0.009 | 1.783 | 0.807 | 0.285 | 0.024 | 2.908 |
| 1.4 | trial 3 | 0.000 | 0.009 | 1.784 | 0.807 | 0.249 | 0.025 | 2.874 |
| 2.1 | warmup | 0.029 | 0.000 | 1.782 | 0.801 | 0.251 | 0.025 | 2.888 |
| 2.1 | trial 1 | 0.000 | 0.009 | 1.781 | 0.801 | 0.253 | 0.025 | 2.869 |
| 2.1 | trial 2 | 0.000 | 0.009 | 1.780 | 0.800 | 0.253 | 0.025 | 2.868 |
| 2.1 | trial 3 | 0.000 | 0.009 | 1.781 | 0.800 | 0.255 | 0.025 | 2.870 |
| 2.2 | warmup | 0.031 | 0.000 | 1.786 | 0.801 | 0.281 | 0.029 | 2.927 |
| 2.2 | trial 1 | 0.000 | 0.009 | 1.786 | 0.801 | 0.278 | 0.028 | 2.902 |
| 2.2 | trial 2 | 0.000 | 0.010 | 1.782 | 0.801 | 0.299 | 0.029 | 2.921 |
| 2.2 | trial 3 | 0.000 | 0.009 | 1.783 | 0.800 | 0.302 | 0.029 | 2.923 |
| 2.3 | warmup | 0.029 | 0.000 | 1.768 | 0.798 | 0.301 | 0.029 | 2.926 |
| 2.3 | trial 1 | 0.000 | 0.009 | 1.764 | 0.798 | 0.302 | 0.028 | 2.902 |
| 2.3 | trial 2 | 0.000 | 0.009 | 1.766 | 0.798 | 0.304 | 0.028 | 2.906 |
| 2.3 | trial 3 | 0.000 | 0.009 | 1.765 | 0.798 | 0.303 | 0.028 | 2.903 |
| 2.4 | warmup | 0.029 | 0.000 | 1.758 | 0.801 | 0.300 | 0.028 | 2.916 |
| 2.4 | trial 1 | 0.000 | 0.009 | 1.771 | 0.801 | 0.301 | 0.028 | 2.909 |
| 2.4 | trial 2 | 0.000 | 0.009 | 1.761 | 0.801 | 0.303 | 0.028 | 2.901 |
| 2.4 | trial 3 | 0.000 | 0.009 | 1.768 | 0.800 | 0.301 | 0.028 | 2.905 |

## Evidence

- [Baseline trace](../llama-context2048.trace.json.gz)
- [Pooled trace](trace.json.gz)
- [Comparison and methodology](RESULTS.md)
- [Candidate CSV, including every trial](per-candidate-with-trials.csv)
- [Every warmup/trial phase as CSV](per-execution-breakdown.csv)
