# llama.cpp-qsa: Qwen3.8-Flash-Next at 60+ tok/s on two 16 GB GPUs

This is a llama.cpp fork tuned for one model on one machine: **Qwen3.8-Flash-Next** (arch
`qwen4exp`, ~177B MoE with 512 experts / top-10, DeltaNet + sparse-attention (QSA) layers,
hyper-connections, a 27 GB per-layer-embedding table), quantized as ISTA-DASLab GSQ-RCO IQ3_XXS,
running on **2x RTX 5060 Ti 16 GB + an EPYC 7K62 (Zen 2) with 94 GB RAM**, batch 1.

It stacks three things on top of llama.cpp master:

1. **BELLS**, a per-layer VRAM expert cache for MoE models (by danielguckert4-droid; its original
   README is kept in [docs/BELLS.md](docs/BELLS.md)).
2. **MTP speculative decoding** from upstream PR #28243, with the Unsloth shared MTP head placed
   entirely on the second GPU.
3. About a dozen local patches and cherry-picks, each measured on its own (listed below).

| | Start (2026-09-21, static expert placement) | Now (`main`, 64k context) |
|---|---:|---:|
| Decode, short prompt | 34.8 tok/s | **62-65 tok/s** |
| Decode at 32k / 61k context | - | 57-63 / 55-60 tok/s |
| Prefill (8k-61k prompt) | 162-246 tok/s | 340-380 tok/s |

## Where to look

| Path | What it is |
|---|---|
| [local/README.md](local/README.md) | The lab notebook: **a table of every attempt** (kept and rejected) with its decode, prefill and VRAM effect, footnotes, the costs/downsides of each kept change, and a recap. Start here. |
| [local/bench/](local/bench) | Benchmark and profiling scripts (A/B drivers, nsys kernel attribution, depth curves). They hardcode the author's model paths; `depthcurve2.sh` takes `BIN`, `MODEL`, `OUT`, `FILLER` env vars. |
| [local/results/](local/results) | Raw outputs: per-run JSON, server logs, summaries. |
| `git log --author=god main` | The local commits. Code commits are prefixed by subsystem (`cuda:`, `qwen4exp`, `bells :`), notes and results by `local:`. |

## Local changes on `main`

| Change | Commit / switch | Effect (decode unless noted) |
|---|---|---|
| AVX2 `Q2_0` vec_dot for Zen 2 (no VNNI) | 251288429 | +14-21% when experts run on CPU |
| `BELLS_HOST_ONLY`: cache only host-resident layers | cb92ae21e | neutral, kept as an option |
| Chunked QSA indexer scoring | 7e6b26317, `LLAMA_QSA_CHUNK` | 128k prefill ~3x, compute buffer -1.6 GB |
| Cap MTP draft-context ubatch | 60a79a76e, `LLAMA_DRAFT_UBATCH` | lets 128k + `-ub 2048` + MTP fit |
| CUDA graphs keyed by node count and first/last shape | b2d148fb2 | removes re-captures with variable batch shapes |
| mmvf for 16-512-row weights that mmf rejects | 9f14ee5b7, `GGML_CUDA_MMVF_FALLBACK_MIN_ROWS` | +3.5% (fixed-token bench) |
| FR-Spec: MTP drafts over the 64k most frequent tokens | 3b47dd11e, `LLAMA_MTP_VOCAB`, `LLAMA_MTP_VOCAB_N` | +3.0-3.5% |
| Incremental pooled-key cache for the QSA indexer (upstream PR #28699) | ac3af2fc1, `LLAMA_QSA_NO_POOLED_CACHE=1` disables | +13.8% at 61k |
| Sparse flash attention with Q8_0 KV: convert only the selected rows to F16 | fa-sparse-q8 merge, `GGML_CUDA_FA_SPARSE_ALL_ROWS=1` disables | +12.5% at 61k |
| One-element-per-thread `get_rows` for rows of <= 32 elements | getrows-small merge, `GGML_CUDA_GET_ROWS_SMALL=0` disables | -6% ms/pass at 61k |

Rejected attempts (tensor parallel, n-gram drafting, draft p_min, `-ub 4096`, F16 KV, split
GPU/CPU experts, and others) are in the table in local/README.md with the reason for each.

## Build

```sh
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Only sm_120 (Blackwell consumer) has been measured. The CPU side assumes AVX2.

## Run (the 64k test config)

```sh
LLAMA_DRAFT_UBATCH=256 \
LLAMA_MTP_VOCAB=local/results/fr/rank.ids LLAMA_MTP_VOCAB_N=65536 \
build/bin/llama-server -m Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-HCQ8-00001-of-00002.gguf \
  -md mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf --spec-type draft-mtp --spec-draft-n-max 2 \
  -devd CUDA1 -ngld 99 \
  -ngl 99 -sm layer -ts 28,20 --cpu-moe-pinned --bells-slots 240 \
  -c 65536 -fa on -ctk q8_0 -ctv q8_0 -ub 2048 -b 2048 \
  -lm mmap -lzm off -t 32 -tb 32 --fit off --parallel 1 --jinja
```

- `--bells-slots 240`: experts per layer kept in VRAM (~98% hit rate here). Set it by hand; auto-sizing
  left 7 GB unused.
- `--cpu-moe-pinned` locks ~41 GB of host RAM and makes loading take ~1 minute.
- `-lzm off` keeps the 27 GB PLE table in page cache, so RAM, not disk, is the limit.
- HCQ8 is a local requant of the shipped IQ3_XXS file with hyper-connection weights in Q8_0
  instead of BF16 (+9.7% decode, -560 MiB VRAM). Its quality (KLD) has not been measured.
- `rank.ids` is the FR-Spec token ranking for this tokenizer (English prose and code weighted).
- Inside the VM, `vm.compaction_proactiveness=0` removed a bimodal decode speed (see the notebook).

## Caveats

- Numbers are from one machine, batch 1, greedy decoding, and move a few percent between sessions
  (page cache, run order); the notebook says which comparisons were same-session A/Bs.
- The pooled-key cache is an unmerged upstream PR and works for a single stream only.
- The notebook cites a private knowledge base ("claim NN") and paths on the author's hosts; those
  are not included.

## Upstream

- llama.cpp: https://github.com/ggml-org/llama.cpp (MIT, see LICENSE)
- BELLS original README: [docs/BELLS.md](docs/BELLS.md)
