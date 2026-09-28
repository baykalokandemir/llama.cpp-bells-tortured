# llama.cpp-qsa: Qwen3.8-Flash-Next at ~70 tok/s on two 16 GB GPUs

This is a llama.cpp fork tuned for one model on one machine: **Qwen3.8-Flash-Next** (arch
`qwen4exp`, ~177B MoE with 512 experts / top-10, DeltaNet + sparse-attention (QSA) layers,
hyper-connections, a 27 GB per-layer-embedding table), quantized as ISTA-DASLab GSQ-RCO IQ3_XXS,
running on **2x RTX 5060 Ti 16 GB + an EPYC 7K62 (Zen 2) with 94 GB RAM**, batch 1.

It is a fork of [DGuckert/llama.cpp-BELLS](https://github.com/DGuckert/llama.cpp-BELLS), itself a
fork of llama.cpp, and stacks three things:

1. **BELLS**, a per-layer VRAM expert cache for MoE models (by danielguckert4-droid), taken unchanged
   from its `bells-next` branch except for one local patch (below); its README is kept in
   [docs/BELLS.md](docs/BELLS.md).
2. **MTP speculative decoding** from upstream PR #28243, with the Unsloth shared MTP head placed
   entirely on the second GPU.
3. About a dozen local patches and cherry-picks, each measured on its own (listed below).

| | Start (2026-09-21, static expert placement) | Now (`main`, 64k context) |
|---|---:|---:|
| Decode, short prompt | 34.8 tok/s | **72 tok/s** |
| Decode at 32k / 61k context | - | 69 / 64 tok/s |
| Prefill (8k-61k prompt) | 162-246 tok/s | 367-401 tok/s |

## Who did this

The experiments, patches, profiling and write-ups in this fork were done by **Claude** (Anthropic's
AI model, running in Claude Code) working over SSH on the owner's machine. The owner,
[baykalokandemir](https://github.com/baykalokandemir), set the direction, decided which changes to
keep and supplied the hardware. Commits Claude wrote carry a `Co-Authored-By: Claude` trailer. The
goal was to hyper-optimize one model for this one setup, not to produce general-purpose upstream
patches; treat the code accordingly.

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
| AVX2 `Q2_0` vec_dot for Zen 2 (no VNNI) | 4412c6b96 | +14-21% when experts run on CPU |
| Chunked QSA indexer scoring | 499bd2c23, `LLAMA_QSA_CHUNK` | 128k prefill ~3x, compute buffer -1.6 GB |
| Cap MTP draft-context ubatch | 5866be973, `LLAMA_DRAFT_UBATCH` | lets 128k + `-ub 2048` + MTP fit |
| CUDA graphs keyed by node count and first/last shape | 204b78fa1 | removes re-captures with variable batch shapes |
| mmvf for 16-512-row weights that mmf rejects | bba5e909b, `GGML_CUDA_MMVF_FALLBACK_MIN_ROWS` | +3.5% (fixed-token bench) |
| FR-Spec: MTP drafts over the 64k most frequent tokens | 542a10f22, `LLAMA_MTP_VOCAB`, `LLAMA_MTP_VOCAB_N` | +3.0-3.5% |
| Incremental pooled-key cache for the QSA indexer (upstream PR #28699) | 1bb8441ff, `LLAMA_QSA_NO_POOLED_CACHE=1` disables | +13.8% at 61k |
| Sparse flash attention with Q8_0 KV: convert only the selected rows to F16 | 3681909aa, `GGML_CUDA_FA_SPARSE_ALL_ROWS=1` disables | +12.5% at 61k |
| One-element-per-thread `get_rows` for rows of <= 32 elements | 58309c90f, `GGML_CUDA_GET_ROWS_SMALL=0` disables | -6% ms/pass at 61k |
| BELLS: skip unchanged slot-table uploads (the one-readback-per-layer half is upstream now) | 24f32a1fa, `BELLS_UPLOAD_ALWAYS=1` disables | part of +1.5-3% |
| Stable uid for the graph views the BELLS callback creates (skips CUDA graph re-checks) | 0b414a72f, `GGML_SCHED_VIEW_UID=0` disables | +2-3% |
| GPU token sampling (upstream flag) plus no scheduler re-reserve per request | `--backend-sampling`; 7653ffe69, 72e817c2b, `LLAMA_SAMPLER_ALWAYS_RESERVE=1` disables | +5%; without the fix short prompts lose ~0.5 s |
| Radix top-k kernel for the QSA indexer at prefill (upstream issue #29326) | 9077b21d5, `GGML_CUDA_TOPK_RADIX=0` disables | prefill +6-9% |
| Upstream cherry-picks #29393 (RMS_NORM+SCALE fusion) and #29298 (sparse-FA fix) | c34a39d35, 7d728b975 | neutral here |

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
  -lm mmap -lzm off -t 32 -tb 32 --fit off --parallel 1 --jinja --backend-sampling
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
- The pooled-key cache is an unmerged upstream PR and works for a single stream only: with `--parallel > 1` it corrupts across sequences instead of falling back, so set `LLAMA_QSA_NO_POOLED_CACHE=1` there.
- The notebook cites a private knowledge base ("claim NN") and paths on the author's hosts; those
  are not included.

## History layout

`main` is linear on top of two upstream points, so a future rebase is one command:

    bells-next (DGuckert, = llama.cpp + BELLS) -> merge of PR #28243 head 6fcaa16f4 (qwen4exp MTP)
      -> our commits in the order they were made (code and local/ experiment log interleaved)

Every replayed commit carries `(cherry picked from commit <old>)`, so hashes cited in local/README.md
and older notes resolve through that trailer; the pre-restack history is kept under the tag
`archive/pre-restack-2026-09-28`. Upstream llama.cpp PRs we carry ahead of time (#28699, #29298,
#29393, and the radix top-k kernel for #29326) name the PR in their subject, so they can be dropped
once the BELLS base contains them.

## Upstream

- llama.cpp: https://github.com/ggml-org/llama.cpp (MIT, see LICENSE)
- BELLS: https://github.com/DGuckert/llama.cpp-BELLS (branch `bells-next`), README in [docs/BELLS.md](docs/BELLS.md)
