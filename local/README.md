# Local notes for the kv-f16 branch (wintermute, 2026-09-24)

Branch `kv-f16` is `qsa-slim` (chunked qwen4exp indexer + LLAMA_DRAFT_UBATCH) plus a debug
log: `GGML_CUDA_SPARSE_LOG=1` prints the first flash-attention kernel decisions to stderr.

## Experiment: does an F16 KV cache (which enables the CUDA sparse-FA path) beat Q8_0?

Config: HCQ8 IQ3_XXS, BELLS pinned 240 slots, -ts 28,20, -c 65536, -ub 2048 op offload,
MTP shared-Q4_K_M head on CUDA1 (depth 2), LLAMA_DRAFT_UBATCH=256. Real-text prompts
(llama.cpp docs), 2 reps per depth, 200 generated tokens. Scripts in local/bench,
raw rows in local/results. Run: local/bench/dc-batch1.sh.

| depth | F16 decode | Q8_0 decode | F16 prefill | Q8_0 prefill |
|---:|---:|---:|---:|---:|
| shallow | 59.1 / 59.7 | 59.1 / 59.5 | - | - |
| 8k | 48.0 / 48.4 | 57.7 / 56.3 | 159 / 172 | 370 / 380 |
| 16k | 51.5 / 53.2 | 54.8 / 52.5 | 367 / 289 | 369 / 380 |
| 32k | 51.6 / 46.6 | 51.6 / 47.2 | 273 / 301 | 338 / 361 |
| 60k | 45.3 / 42.3 | 40.3 / 41.5 | 274 / 338 | 341 / 349 |

Kernel log (verified): F16 decode takes the sparse MMA path (sparse=1, n_kv_max=2051);
Q8_0 decode always selects the dense VEC kernel (quantized KV never consults the sparse
check). Normalised by MTP acceptance, time per verification pass at 60k is ~62.5 ms (F16)
vs ~65 ms (Q8_0): ~4%. F16 prefill is slower (F16 arm ran first after a rebuild, so part
of that may be warm-up). VRAM at load: F16 616/1194 MiB free, Q8_0 ~1000/1290 MiB free.

Conclusion: at <= 60k the sparse FA kernel is not where the depth cost is; stay on Q8_0.
Per-pass cost still grows ~40 -> ~65 ms from shallow to 60k; the next step is an Nsight
profile at depth to attribute it (indexer scoring, draft-layer dense attention, FA).
