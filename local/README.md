## Everything we tried (2026-09-04 .. 2026-09-25)

Model: ISTA-DASLab Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS (from 2026-09-22 the local HCQ8 variant), arch
qwen4exp, on wintermute (2x RTX 5060 Ti 16 GB, EPYC 7K62 Zen2, 94 GB RAM). Numbers are decode tok/s,
batch 1, unless stated. Sources: kb topics bells-mtp-flashnext, flashnext-iq3-placement-and-hc-quant,
cpu-moe-flashnext-decode, moe-expert-cache-b1, flash-next-audit-2026-09-04; kb claims 82-105;
magi:/root/llm-optimization-shit/QWEN38FN-{HANDOFF-2026-09-21,IQ3-XXS-HANDOFF-2026-09-22*,
IQ3-XXS-RESULTS-2026-09-22,BELLS-MTP-BRIEF-2026-09-24}.md; the sections below and local/results/.
Numbers from different sessions are not directly comparable (page-cache regime, footnote 3).

| Attempt | Branch/commit or flag | Decode effect | Prefill effect | Context / free VRAM effect | Verdict |
|---|---|---|---|---|---|
| **Before GSQ-RCO (other quants, ik_llama)** | | | | | |
| ik_llama CPU-MoE keeper (AtomicChat Q4_K_M, -ncmoe 30) | ik, -sm graph | 27.80 code / 27.77 prose, 24.73 at 32k | flat 183-208 tok/s to 92k | 96k ctx | superseded |
| PR #27861 LRU expert cache ported into ik | --moe-expert-cache 96, -ncmoe 36 | +7.2 to +12.0% over that keeper | not measured | not measured | superseded |
| MTP with experts mostly on CPU (ik) | draft-mtp | short-code only; -13 to -30% at depth | not measured | not measured | rejected [1] |
| **Q2_0 / IQ3_XXS bring-up on mainline (2026-09-21/22)** | | | | | |
| AVX2 Q2_0 vec_dot kernel (Zen2 has no VNNI) | 251288429 | IQ3_XXS +14..21% (28.3 -> 34.4 at -ncmoe 26); Q2_0 file +36% | +1.5% (noise, Q2_0 1.3k prompt) | none | adopted |
| Leaner-unpack variant of that kernel | local patch | 43.30 vs 43.07 (Q2_0), flat | not measured | none | rejected |
| Q2_0 file instead of IQ3_XXS | model choice | 43.56 vs 34.37 | not measured | lower -ncmoe possible (10 vs 24) | rejected (quality 89.07 vs 92.57 task avg) |
| n-gram spec decode (map-k / mod / cache) on Q2_0 | --spec-type ngram-* | copy 28.75 -> 20.00 / 20.10 / 23.26; prose 26.71 -> 29.49 best | not measured | none | rejected |
| Tensor parallel | -sm tensor | ~28% slower than -sm layer at matched -ncmoe (e.g. 40.88 vs 32.01) | not measured | fits fewer GPU banks | rejected |
| Hand-picked bank placement + -ub 128 (2026-09-21 keeper) | -ot, -ncmoe | 34.77 vs 32.43 contiguous -ncmoe 26 | not measured | 400 / 214 MiB free | superseded |
| -ub 256 vs -ub 128 | -ub | 34.51 vs 34.77 | not measured | 364/154 vs 400/214 MiB free | rejected (ub 128 kept) |
| -ncmoe before -ot silently voids -ot | flag order | 33.16 verbatim vs 34.19 with -ncmoe 0 | not measured | 14,218/14,400 vs 15,490/15,676 MiB used | diagnostic [2] |
| Uniform LRU expert cache (PR #27861) on mainline ff0dbb9 | llama-cache-current | 31.03 vs 34.77 keeper (-10.8%) at same VRAM | not measured | same VRAM | rejected |
| Bank placement knapsack (cost per VRAM byte) | place2-4.py, -ot, -ts 18,30 | 36.56 -> 38.67 (contiguous control 36.67) | not measured | same budget | adopted (superseded by BELLS) |
| Knapsack allowing cross-device banks | -ts 34,14 | 33.06 vs 34.19 (75-115 us per crossing) | not measured | same | rejected |
| Layer split alone, no re-selection | -ts 20,28 | 36.67 vs 36.56 | not measured | same | neutral |
| BF16 -> Q8_0 hyper-connection requant (HCQ8) | gen-tt.py + placeholder imatrix | 38.67 -> 42.41 (+9.7%) | not measured | -560 MiB VRAM | adopted [4] |
| Placement re-solved after HCQ8 (L=21) | -ts 20,28 | 42.52 vs 42.26 (+0.6%, one high sample) | not measured | CUDA1 free 188 vs 642 MiB | rejected |
| "Shallow speed" placement profile | -ot variant | 42.87 vs 42.48 | not measured | 260 / 142 MiB free | rejected (too tight) |
| CUDA env knobs | GGML_CUDA_{GRAPH_OPT,PDL,DISABLE_FUSION,DISABLE_GRAPHS} | +0.3 / -0.7 / -3.3 / -7.3% vs defaults | not measured | none | diagnostic (defaults kept) |
| 8-warp Q8_0 MMVQ for HC projections | local, reverted | 40.89 vs 42.48 (-3.7%) | not measured | none | rejected |
| Decode-time op offload off (hybrid) | --no-op-offload | 42.48 both | not measured (hybrid 162-246 was with op offload on) | CUDA0 -670 MiB (compute buf 795 -> 124 MiB) | adopted (hybrid era) |
| ik_llama vs mainline (Q2_0 ported to ik) | ik q2_0-port | shipped file ik +3.7% (39.92 vs 38.44); HCQ8 mainline +5.5% (42.48 vs 40.26); all-CPU tie | not measured | ik 536 vs 280 MiB free CUDA0 | rejected (mainline + HCQ8) |
| Placement re-solved for ik capacities | place5.py | 40.36 vs 40.26 (noise) | not measured | CUDA1 headroom 130 MiB | rejected |
| Parallel slots on hybrid | --parallel 4 | 16.47 per stream, 63.24 aggregate (+49%) | not measured | no extra VRAM | diagnostic |
| **BELLS expert cache (2026-09-24, 32k, HCQ8)** | | | | | |
| BELLS instead of static placement | --cpu-moe --bells-slots 330 -ts 26,22 | 40.88 -> 44.78 (98.1% hit); 280 slots 42.36; plain --cpu-moe 27.86 | not measured | 15,410 / 15,502 MiB used | adopted |
| BELLS auto-sizing | --bells | 35.58 (chose 170 slots) | not measured | left 7.1 GiB unused | rejected |
| Split experts GPU/CPU per token | --bells-split 8/6/4 | 31.55 / 29.29 / 27.19 vs 44.78 | not measured | same | rejected |
| Whole banks resident + BELLS on the rest | BELLS_HOST_ONLY=1 (cb92ae21e) | 6 banks + 309 slots 44.38 (tie); 12 + 276 43.20 | not measured | same | neutral |
| Mechanism overhead probe | --bells-passive | 26.64 (~34 us/layer/token) | - | - | diagnostic |
| Refresh / L2 victim cache | --bells-refresh, --bells-l2-slots | not measured | not measured | not measured | rejected untested [5] |
| **MTP (PR #28243 + BELLS cherry-pick, Unsloth shared-Q4_K_M head, 32k)** | | | | | |
| MTP on hybrid, head experts on CPU | -cmoed, depth 2 | 40.88 -> 49.67 (+21.5%); depth 3 49.84 | not measured | not measured | superseded |
| MTP on BELLS, head experts on CPU | 310 slots | 44.78 -> 50.47 (+12.7%) | not measured | -20 slots | superseded [6] |
| MTP head fully on GPU1 | -devd CUDA1 -ngld 99 | 50.47 -> 56.91 (275 slots) | not measured | -35 slots | adopted |
| Split rebalanced for the head | -ts 28,20, 298 slots | 59.2 | not measured | +23 slots | adopted |
| Pinned host experts | --cpu-moe-pinned | 59.2 -> 61.62 (+4%; copy 155 -> 20 us/layer-call) | not measured | ~41 GB locked RAM, load 9 -> 54 s | adopted |
| Q8_0 MTP head | head file | 58.13 vs 59.2 | not measured | -13 slots (285) | rejected |
| Draft depth 3 | --spec-draft-n-max 3 | 59.63 vs 59.2 (prose 63.7 / code 55.6); 56.12 with pinned + copy stream | not measured | none | neutral |
| Separate copy stream | BELLS_COPY_STREAM=1 | 59.45 vs 59.2; pinned 61.37 vs 61.62 | not measured | none | neutral |
| Draft confidence cutoff | --spec-draft-p-min 0.5-0.9, depth 2-4 | all slower: n2 p0.75 41.40/43.07 vs 55.96/46.60 prose/code | not measured | none | rejected [7] |
| Cap MTP draft ubatch | LLAMA_DRAFT_UBATCH=256 (60a79a76e) | not isolated | not isolated | needed for 128k + -ub 2048 + MTP to fit | adopted |
| FR-Spec draft vocab, top 64k | LLAMA_MTP_VOCAB(_N=65536) (3b47dd11e) | prose 57.4-58.3 -> 59.9-60.1, code 58.9-59.0 -> 60.9-61.1 (+3.0/+3.5%), 64k ctx | not measured | ~110 MiB on GPU1 (~1.4 slots) | adopted |
| FR-Spec draft vocab, top 32k / 16k | LLAMA_MTP_VOCAB_N | gain erased (-7 / -19 accepted drafts, prose) | not measured | not measured | rejected |
| **Prefill and compute buffer** | | | | | |
| Op offload on + -ub 2048 | drop --no-op-offload, -ub 2048 | shallow 47-59 vs 53-64 (32k) | 8k: 70-92 -> 363-375 | ~43 fewer slots (255) | adopted (prefill config) |
| -ub 512 / 1024 | -ub | shallow 44-60 / 49 | 8k: 166-169 / 253-260 | 260 / 255 slots | rejected (2048 better) |
| -ub 4096 | -ub 4096 | ~-10% (53-58 at 8k) | 8k: 482-490 | ~40 fewer slots (215) | rejected |
| Split 29,19 at -ub 4096 | -ts 29,19 | inside noise | 451-501 | +12 slots | neutral |
| Chunked QSA indexer (qsa-slim) | LLAMA_QSA_CHUNK=256 (7e6b26317) | decode graphs unchanged | 110k prompt ~83 -> 263 (op offload now fits at 128k) | 128k/-ub 2048 compute buf 3,872/3,815 -> 2,280/1,895 MiB | adopted [8] |
| 128k context on the 32k decode config | -c 131072, 270 slots | 59.01 shallow; suite 40.1/35.4/32.7/27.9/28.3 at 7.6k-127k | ~80-90 (no op offload) | ~1.1 GB per GPU vs 32k | diagnostic [9] |
| Op-offload min batch for slow short prefill | GGML_OP_OFFLOAD_MIN_BATCH=192 | none | 31-token prompt still ~18 tok/s at times | none | neutral (anomaly open) |
| **KV type** | | | | | |
| F16 KV to enable sparse FA (kv-f16) | -ctk/-ctv f16, 22cd92d43 | F16 vs Q8_0: 48/57 at 8k, 52/54 16k, 49/49 32k, 44/41 60k | slower | higher | rejected |
| **CUDA graphs and kernel dispatch** | | | | | |
| Graph stats counter | GGML_CUDA_GRAPH_STATS=1 (46ad58f0e) | debug only | - | - | diagnostic |
| Shape-keyed CUDA graph cache (graph-cache) | b2d148fb2 | no baseline change in clean ABAB; p_min 0.75 41.1/42.8 -> 47.1/50.2 | none | none | neutral (kept, harmless) [10] |
| mmvf for thin weights (HC inject [10240,4]) | GGML_CUDA_MMVF_THIN_ROWS (1517cf129) | -2% on code (mmvf 7.1 us vs cuBLAS split-K 5.8 us) | not measured | none | rejected |
| mmvf fallback for 16-512 rows not %32 (ssm_alpha/beta) | GGML_CUDA_MMVF_FALLBACK_MIN_ROWS (9f14ee5b7) | batched-bench npl 3: 88.6/89.8 -> 92.3/92.6 (+3.5%); server prose 57.0-57.2 -> 58.2-58.3 | not measured | none | adopted [11] |
| **PLE (27 GB n-gram table)** | | | | | |
| Keep PLE mapped and resident | -lzm off (vs on) | cold -lzm on: first 2 requests 53.9/56.2 ms/pass vs ~42; overall 51.1 vs 56.9 | cold 8k 159.6 -> 334.6; 32k new text 181.7 -> 308.2 | ~27 GB page cache | adopted |
| PLE history after rejected drafts (ik PR #2460) | ple-trace, LLAMA_PLE_TRACE=1 | 907 accepted tokens, 0 mismatches | - | - | diagnostic (no bug) |
| Direct-read lazy mode (PRs #28136/#29030) | --lazy-mode on-direct | not measured | not measured | not measured | not tried |
| **System / VM** | | | | | |
| Guest proactive compaction off | vm.compaction_proactiveness=0 | bimodal ~40/48/58 ms/pass -> 40.1-41.5 ms/pass (57.4-59.4 tok/s) after 1st request | not measured | none | adopted [12] |
| VM 32 -> 48 vCPU | Proxmox | fast state ~42 -> ~40 ms/pass (not isolated) | not measured | none | adopted (not isolated) |
| Thread count | -t 4/8/32/48, -tb 32 | irrelevant: -t 8 56.9, -t 48 56.0 vs -t 32 56.9 | not measured | none | neutral |
| **In progress** | | | | | |
| Sparse FA with Q8_0 KV: convert only listed rows | fa-sparse-q8 (merged), GGML_CUDA_FA_SPARSE_ALL_ROWS=1 disables | 61k 49.6/50.9 -> 55.8/57.3 (+12.5%), 32k +5%, 16k +4% (64k ctx, 1 pair) | unchanged | none (full F16 buffer still allocated) | adopted [13] |
| Indexer top-k over blocks -> replaced by short-row get_rows kernel | getrows-small (merged), GGML_CUDA_GET_ROWS_SMALL=0 disables | ms/pass 61k 48.3 -> 45.3 (-6%), 32k -2.7%, <=16k none | unchanged | none | adopted [14] |
| PR #28699 pooled-key cache | LLAMA_QSA_NO_POOLED_CACHE=1 disables (ac3af2fc1) | 61k 43.8 -> 49.8 (+13.8%), 32k +4.1%, 8-16k +2-3% (64k ctx, 2 pairs) | unchanged (345-380) | ~50 MiB per GPU | adopted |

Footnotes:

1. From the 2026-09-04 audit and 2026-09-10 CPU-MoE work on other quants: MTP only paid with experts
   mostly in VRAM. It became worthwhile here once BELLS kept ~98% of routing on the GPU.
2. `-ncmoe` and `-ot` share one override list, first match wins (claim 82). The 2026-09-21 keeper
   record was marked SUPERSEDED for this reason.
3. Page-cache regime: identical configs measured 36.0 before and 38.3-38.8 after a load that evicted
   other cache (claim 86). The hybrid baseline is 42.48 in a warm regime and 40.88 in the 2026-09-24
   session; the BELLS/MTP ladder uses the 40.88 session.
4. HCQ8 output is byte-identical at temperature 0 on the probes, but KLD/perplexity against the
   shipped file has not been measured. Q6_K/Q5_K hyper-connections were planned, not run.
5. Not run by design: `--bells-refresh` zero-slots unobserved experts (lossy; the author marks it
   research-only); an L2 hit costs two PCIe hops through host staging.
6. Verify batches multiply cache misses: per-layer copy time 37 -> 113 us (claim 95).
7. Early-stopped drafts change the verify batch size and defeat graph reuse, and a verify pass costs
   about the same for 2 or 3 tokens (claim 99). Still a loss with the graph cache (47.0 overall).
8. PPL at 8k/ub 2048 2.0374 -> 2.0351 (+-0.031). The runtime buffer grows ~80 MiB past its
   reservation at full depth; keep >= 300 MiB free.
9. The benchdb suite depth sweep used synthetic filler and single samples; real-text depth decode on
   the 64k config is 59/57/54/50/41 at 0/8k/16k/32k/60k. Needle and concurrency axes need a rerun.
10. Claim 100's +10.7% baseline gain was retracted: it came from PLE page-cache warmth and run order
    (claim 101), not from the graph key.
11. Server A/B is confounded because new numerics change the text and MTP acceptance; the fixed-token
    batched-bench number is the clean one. The ik PR #2373 inject kernel does not carry over to
    mainline (inject costs only ~5.8 us here).
12. kcompactd0 ran after each server load and stalled the latency-bound host thread (claim 103).
    Persisted in /etc/sysctl.d/99-llm-decode.conf in the guest. It explains the "bimodal decode"
    that thread-count, graph-key and -lzm tests had been chasing.

13. The 3-query MTP verify pass takes the MMA flash-attention kernel, which needs F16 K/V; launch_fattn
    converted the whole Q8_0 cache of each QSA layer every call (O(n_kv)) and the sparse kernel then read
    only the ~2048 listed rows. The patch computes the index lists first and converts only those rows.
    Depth profile at 61k: this conversion was 5.1 ms of the ~10 ms per-pass growth over 8k (claim 107).
    test-backend-ops FLASH_ATTN_EXT 3986/3986 on and off, incl. new Qwen-QSA Q8_0 nb 3/4 kv 16k/32k cases.
14. The profile showed top-k itself flat (0.32 ms/pass); the cost was the get_rows expanding block scores
    to every cell (3-float rows, one thread block per row: 2.5 ms/pass at 61k). A one-element-per-thread
    kernel for rows <= 32 is bit-identical. Block-level top-k was dropped (blk_cells cannot express the
    incomplete tail block). GET_ROWS tests 252/252.

### Costs and downsides of what we kept

Only adopted (or about-to-be-adopted) changes are listed; rejected attempts cost nothing.

| Change | Cost / downside |
|---|---|
| BELLS expert cache | Out-of-tree fork: every upstream rebase is work. Prompts over ~n_slot/10 tokens bypass the cache (prefill runs experts on CPU without op offload). Auto-sizing is unusable; slots must be set by hand. |
| MTP head on GPU1 | ~35 fewer slots. Verify batches multiply cache misses (37 -> 113 us copy per layer-call). Speed varies with how predictable the text is. |
| --cpu-moe-pinned | ~41 GB locked RAM, load 9 -> 54 s, some swap during load; the copy evicts other page cache (it pushed out PLE until -lzm off). |
| -lzm off | PLE table (~27 GB) stays in page cache: RAM, not disk, sets the floor. A larger (e.g. BF16) PLE would not fit this way. |
| Op offload + -ub 2048 | ~43 fewer slots than the decode-only config; shallow decode a few percent lower (~59 vs 61.6 at 32k). Bigger compute buffer (keep >= 300 MiB free). |
| Chunked QSA indexer | Extra small kernels per chunk; PPL 2.0374 vs 2.0351 (inside error). Local patch to carry. |
| LLAMA_DRAFT_UBATCH | Draft-context prefill with a smaller ubatch; its own cost not measured. Env var to remember. |
| HCQ8 requant | Quality never measured (no KLD/PPL vs the shipped file). A second model file to maintain. |
| AVX2 Q2_0 kernel | None known (CPU-only path). |
| Shape-keyed CUDA graph cache | Keeps more graphs alive (a little memory); no speed effect at fixed draft depth. |
| mmvf fallback (16-512 rows) | Changes numerics, so outputs differ from earlier builds (precision not worse: activations stay F32). Global dispatch change, measured only on sm_120. |
| FR-Spec draft vocab (top 64k) | ~110 MiB on GPU1 (1-2 slots). English/code-weighted ranking: other languages likely draft worse. Needs a ranking file matching the tokenizer plus two env vars. |
| Pooled-key cache (PR #28699) | Unmerged upstream PR with non-trivial rollback bookkeeping. Single-stream only (--parallel > 1 falls back to full recompute). ~50 MiB per GPU. Output identity not provable by hash (depth output is nondeterministic anyway). |
| Sparse FA Q8_0 row conversion | No VRAM saving (full F16 buffer still allocated). Q8_0 only. Correctness relies on the sparse kernel reading only listed rows: the rest of the F16 buffer is stale, so an upstream change that reads more rows would give wrong output (tests cover our shapes). |
| Short-row get_rows | None functional (bit-identical). Applies to every model's short-row gathers; measured only on our case and sm_120. |
| Guest compaction off | System-wide VM setting: memory fragments more over time; on-demand compaction stalls still occur (compact_stall 435 -> 503 over 2026-09-25). |
| VM 48 vCPU | Host cores reserved for the VM; effect not isolated. |
| Overall | ~10 local patches plus 3 unmerged upstream PRs on top of the BELLS fork, and ~8 env vars in the serving config: rebases get harder and a dropped env var silently loses a gain. |

### Recap

- Current build: `main` at 51a7b1b1d = BELLS + PR #28243 MTP + AVX2 Q2_0 + chunked QSA indexer +
  LLAMA_DRAFT_UBATCH + shape-keyed CUDA graphs + mmvf fallback + FR-Spec draft vocab.
- Current test config (not yet in llama-swap): HCQ8, 64k Q8_0 KV, 240 slots, -ub 2048 with op offload, --cpu-moe-pinned,
  MTP head on CUDA1 depth 2, -lzm off, FR-Spec top 64k, guest compaction off.
- Its numbers: shallow decode ~60-61 tok/s (FR-Spec arms); real-text depth 59/57/54/50/41 at
  0/8k/16k/32k/60k (measured before hc-mmvf, FR-Spec and the compaction fix); prefill 340-380 tok/s.
- 128k works at 230 slots: 8k prefill 369-372, 110k prompt 263 tok/s, decode 56-61 at 8k, 25.8 at 110k.
- Decode-only 32k variant (298 slots, -ub 128, no op offload): 61.62 tok/s, prefill ~80-90.
- Changes that mattered most: BELLS over static placement (+9.5%), MTP with the head on GPU1
  (+27% over BELLS alone), HCQ8 (+9.7%), AVX2 Q2_0 (+14-21%), chunked indexer (128k prefill ~3x),
  -lzm off (cold prefill ~2x).
- Smaller adopted wins: pinned experts +4%, mmvf fallback +3.5% (fixed tokens), FR-Spec +3.0-3.5%.
- Noise or confounds: claim 100's +10.7% graph-key gain (retracted, PLE warmth/order); the bimodal
  decode (guest compaction, not threads or flags); page-cache regime shifts of ~7%; server A/Bs of
  numerics changes (text and acceptance move).
- Open: pooled-key cache (PR #28699); per-pass cost growth with depth (~40 -> ~65 ms at 60k) not
  attributed; HCQ8 quality (KLD) unmeasured; needle/concurrency suite reruns; slow short-prompt
  prefill anomaly; depth curve re-measure on the current build.


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
- Thread count (-t 32/8/4/8/32, -tb 32, -lzm off) does not explain the bimodal decode: run means
  48.1 / 47.1 / 52.6 / 50.9 / 51.4 tok/s; repeats of the same -t differ as much as different -t.
  The first request on a fresh server is often slow (39-41), consistent with BELLS cache warm-up.

## PLE n-gram history after rejected drafts (ik PR #2460 check, 2026-09-24)

Not affected. qwen4exp PLE takes n-gram predecessors from the attention KV cells by position
(llama_kv_cache::get_prev_tokens -> seq_pos_tok_le), so a rejected drafts cells are simply rewritten.

## PLE n-gram history after rejected drafts (ik PR #2460 check, 2026-09-24)

Not affected. qwen4exp PLE takes n-gram predecessors from the attention KV cells by position
(llama_kv_cache::get_prev_tokens -> seq_pos_tok_le), so the cells of a rejected draft are simply
rewritten. Trace build on branch `ple-trace` (LLAMA_PLE_TRACE=1 prints pos, token and predecessors
for ubatches of at most 8 tokens); checker local/bench/ple_trace_check.py. MTP n2, prose + code +
8k doc: 907 accepted tokens checked, 166 of them rewritten after a rejection, 0 mismatches. The MTP
draft context never runs the PLE path (one model object in the trace). Greedy text with MTP on vs
off diverges early (prose char 249, code char 72) from batch-size numerics; without MTP the 8k-doc
prompt is itself not reproducible (two runs diverge at char 121) while short prompts are.
Results: local/results/{ex-*,trace-mtp2-*}.

## Bimodal decode root cause (2026-09-25)

Decode varied ~40 / ~48 / ~58 ms per MTP pass for the same prompt with identical acceptance.
Cause: guest proactive memory compaction. kcompactd0 runs for the first requests after each server
load (the load fragments memory: 43 GB file read + 41 GB pinned + 27 GB PLE) and its page migration
stalls the latency-bound host thread. Fix: `sysctl -w vm.compaction_proactiveness=0` in the guest
(root, via qm guest exec). After: all requests after the first at 40.1-41.5 ms/pass (57-59 tok/s).
Tools: local/bench/bimodal-probe.sh (per-request ms/pass + GPU telemetry, NREQ/N/EVICT_MAIN env),
threadmon.py; host-side vCPU sampler at magi:/root/vcpu-sampler.py; all-thread guest sampler
/tmp/allthreads.py (copied to local/bench/). Ruled out: GPU clocks/PCIe, thread count, GPU IRQs,
kswapd, main-thread migration, evicting the main shard page cache.

## Re-check with proactive compaction off (2026-09-25)

local/bench/recheck-batch.sh, graphtest workload (64k Q8_0, 240 slots, -ub 2048, MTP n2), ms per pass:
- baseline (-lzm off, shape key, -t 32): 41.3-42.5, overall 56.9 tok/s
- legacy graph key: 41.4-43.2, 56.4 (no difference at fixed draft depth)
- -lzm on (PLE cold): 53.9, 56.2 for the first two requests, then 41.8-42.7; overall 51.1
  (cold PLE also slows early decode; -lzm off has no decode cost)
- -t 8: 41.3-42.4 (56.9); -t 48: 41.4-45.8 (56.0): thread count irrelevant for decode
- p_min 0.75: 2.09 tokens/pass, 41.1-51.6 ms, overall 47.0 (still a loss)
- baseline repeat: 41.4-48.6 (three passes at 44.6-48.6): residual noise remains, much reduced.
