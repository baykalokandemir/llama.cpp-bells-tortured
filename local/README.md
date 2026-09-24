
## Experiment: --spec-draft-p-min x draft depth (2026-09-24)

Config: qsa-slim build, HCQ8 IQ3_XXS, 64k Q8_0, 240 slots, -ub 2048 op offload, pinned, MTP
shared-Q4_K_M on CUDA1, LLAMA_DRAFT_UBATCH=256. Workload (local/bench/pmin.py): 3x prose + 3x code
short prompts (300 tokens), 2x real-text 8k, 1x 32k (200 tokens). Median decode tok/s:

| arm | prose | code | 8k | 32k | tokens/verify pass (prose/code) |
|---|---:|---:|---:|---:|---|
| n2 p0 (baseline) | 55.96 | 46.60 | 47.96 | 49.95 | 2.42 / 2.34 |
| n2 p0.75 | 41.40 | 43.07 | 48.02 | 39.47 | 2.06 / 2.12 |
| n3 p0.5 | 47.40 | 42.92 | 56.62 | 48.56 | 2.59 / 2.51 |
| n3 p0.75 | 43.67 | 42.13 | 51.09 | 44.64 | 2.28 / 2.18 |
| n3 p0.9 | 41.85 | 41.75 | 45.56 | 38.75 | 1.97 / 1.94 |
| n4 p0.75 | 40.37 | 42.95 | 51.38 | 44.98 | 2.36 / 2.37 |
| n4 p0.9 | 40.91 | 41.57 | 49.68 | 43.33 | 2.03 / 2.05 |

Result: every p_min > 0 arm is slower than the baseline at shallow depth, although acceptance
rises (0.91-0.99 vs 0.68-0.72). Cause: a draft that stops early changes the verify batch size,
and llama.cpp keeps only the last built graph. "graphs reused" per 300-token request falls from
~125 (baseline, ~every pass) to ~55-60, so about half the passes rebuild and re-allocate the graph
(plus CUDA graph re-capture). Per-pass time rises (~43 -> ~50 ms at n2) instead of falling.
p_min is not usable until graphs for several batch shapes are cached. Single-sample depth points
(8k/32k) are noisy; the shallow baseline itself spreads 39.6-57.8 across prompts.
Full table: local/results/pmin-summary.txt.
