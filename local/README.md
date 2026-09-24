
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
# graph-cache branch notes (2026-09-24)

Branch `graph-cache` = qsa-slim + two CUDA commits:
- `GGML_CUDA_GRAPH_STATS=1` prints eager/capture/replay counts of graph_compute calls (debug).
- CUDA graphs keyed by node count + first/last node shape instead of nodes[0] alone
  (`GGML_CUDA_GRAPH_KEY_LEGACY=1` restores the old key).

Workload: local/bench/graphtest.sh (64k Q8_0, 240 slots, -ub 2048, pinned, MTP shared-Q4_K_M on
CUDA1, 3x prose + 3x code short prompts, 300 tokens). Results in local/results/g*.json.

| arm | replay | eager | decode prose/code (median) | tokens/pass | ms/pass | overall tok/s |
|---|---:|---:|---|---:|---:|---:|
| old key, n2 p0 | 94.5% | 3.6% | 56.6 / 56.0 (dips 45.6, 41.3) | 2.38 | 46.4 | 51.3 |
| new key, n2 p0 | 95.6% | 2.5% | 57.1 / 56.5 (all 56.1-58.0) | 2.38 | 41.9 | 56.8 |
| old key, n2 p0.75 | 20.9% | 59.2% | 41.1 / 42.8 | 2.09 | 49.5 | 42.3 |
| new key, n2 p0.75 | 90.0% | 5.4% | 47.1 / 50.2 | 2.09 | 46.5 | 45.0 |

Conclusions:
- The shape key removes the slow runs of the baseline: +10.7% overall throughput, same median.
- With the old key, variable verify sizes (p_min) made 59% of graph_compute calls run eagerly.
- p_min still cannot win here: a verify pass costs ~the same for 2 or 3 tokens (~42 ms), so
  stopping drafts early only loses tokens (2.09 vs 2.38 per pass). A llama.cpp-level per-shape
  graph cache would only help p_min, so it was not built.
- The very first request after load still prefills slowly (13.9 tok/s for 31 tokens); unrelated.

## PLE page cache and the graph-key confound (2026-09-24)

local/bench/pletest.sh + pagecache.py (mincore residency, /proc majflt + read_bytes per request).
- `-lzm on`: pinned loading evicts the 27 GB PLE shard (0-7.5% resident after load, even if pre-read).
  Cold 8k prefill 159.6 tok/s (81,490 major faults, 318 MiB); same 8k again 339.0; 32k new 181.7.
- `-lzm off`: table mapped and resident after load; 8k 334.6 / 338.3, 32k new 308.2, 0 faults.
- Clean ABAB of CUDA graph key (legacy vs shape, same binary, -lzm off): no baseline difference
  (prose/code medians 53.6/47.2, 48.2/47.2, 48.9/47.3, 48.3/47.2). The earlier +10.7% was PLE/order.
- Decode is bimodal run to run (~47 vs ~56 tok/s) regardless of flags. Suspect CPU threadpool
  contention (-t 32 on 32 vCPUs while decode needs almost no CPU work); next test.
Results: local/results/{ple-*,ab-*,pw-*,d-*}.
