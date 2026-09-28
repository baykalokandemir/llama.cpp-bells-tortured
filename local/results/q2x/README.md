# Qwen3.8-Flash-Next on the vLLM "2x3090" stack, ported to 2x RTX 5060 Ti 16 GB (2026-09-28)

Status: PARKED. Baseline measured; container stopped; files kept. Resume instructions and the
list of gaps to close are at the end. Results are also in llama.cpp-qsa local/README.md
(section "vLLM 2x3090 stack on 2x 16 GB", commit 475ed226b) with raw JSON in local/results/q2x/.

## What it is

- Repo: https://github.com/DominikBucko/qwen38-flash-next-2x3090 (cloned at b395412, v0.3.0),
  local clone /home/god/dev/qwen38-flash-next-2x3090.
- Image: ghcr.io/dominikbucko/qwen38-flash-next-2x3090:v0.3.0
  (sha256:8dffc80d90327fb4baf1327def3da9317226a81e0c9ef7f6cb583f2016b5dfc1, 29.3 GB). vLLM day-0
  qwen38 build + humming kernels + a 27-file overlay.
- Checkpoint: albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE rev ef554143 (the rev their repo pins),
  /opt/stacks/llm-stack/models/qwen38fn-w4a16-fp8ple (121 GiB). Intel AutoRound W4A16 (INT4 g128)
  routed experts and linears, BF16 "sensitive" layers, FP8 E4M3 PLE table (51.2B params, ~48 GB),
  INT4-g32 MTP draft (3.9 GiB).
- Serving design: TP2 + EP2 (each GPU owns 256 of 512 experts per layer), all experts in pinned host
  RAM read over UVA, a per-GPU hot-expert cache (static ranking + dynamic LRU), PLE served by a
  separate CPU worker process, MTP depth 3, BF16 KV.
- Their numbers (2x 3090 24 GB, 128 GB RAM, Threadripper): 104.5 tok/s decode, 2,752 tok/s prefill
  at 131k, with 84 hot experts per layer per GPU and streaming prefill on.

## Host changes made for this (temporary; revert steps below)

1. VM 100 RAM 96 -> 112 GiB (qm set 100 --memory 114688; VM uses 2 MB hugepages).
2. 64 GB swap disk on the WD SN810 (local-lvm:vm-100-disk-0, attached as scsi1 = /dev/sdb in the
   guest), mkswap label plesw, fstab entry UUID=a96cd908-... pri=10,discard. The old 8 GB
   /swap.img stays at pri -2.
3. llama-swap retired for good (separate user decision): compose service has
   profiles: ["retired"]; backup compose.yaml.bak-2026-09-28.

## Config that runs (local/results/q2x/wintermute-16gb.env, also configs/ in the clone)

64k context, KV_CACHE_MEMORY_BYTES=1560281088 (1.45 GiB/GPU; 64k needs 1.31 GiB incl. MTP layer and
DeltaNet state), VLLM_WNA16_STATIC_HOT_CACHE_SIZE=40, DISABLE_CUSTOM_ALL_REDUCE=1 (no SM-level P2P
across our two root complexes), QWEN38_PLE_PREFAULT=0 (so the PLE can live in swap),
QWEN38_STREAM_STAGE=0 and QWEN38_STAGE_OVERLAP=0, async scheduling on, restricted MTP draft vocab.

Three capacity guards (64 <= slots <= 96) are relaxed to 32 by bind-mounting patched copies of
tiered_runtime.py, lru_warmup.py and staging_runtime.py (local/results/q2x/patch/). The LRU kernel is
capacity-generic and batches that need more than the capacity fall back to their slower path.
The clone's scripts/docker_serve.sh got one line: ${EXTRA_DOCKER_ARGS:-} for the extra mounts.

Launch:

    cd /home/god/dev/qwen38-flash-next-2x3090
    D=/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe
    M=""; for f in tiered_runtime lru_warmup staging_runtime; do M="$M -v /home/god/dev/q2x-bench/patch/$f.py:$D/$f.py:ro"; done
    (set -a; . configs/wintermute-16gb.env; set +a
     export MODEL_DIR=/opt/stacks/llm-stack/models/qwen38fn-w4a16-fp8ple \
            IMAGE=ghcr.io/dominikbucko/qwen38-flash-next-2x3090:v0.3.0 EXTRA_DOCKER_ARGS="$M"
     nohup ./scripts/docker_serve.sh > /home/god/dev/q2x-bench/serveN.log 2>&1 < /dev/null &)

Load takes ~13 min (weights 231 s, then MTP draft, profiling, KV, graph capture). Serves on :8210,
model name Qwen3.8-Flash-Next. Benchmark: python3 /home/god/dev/q2x-bench/vllm_depthcurve.py 8210
out.json 8192,32768,61440 1 (same prompts as llama.cpp depthcurve2.py; token counts from the usage
block, never from SSE chunks).

## How we got there (8 launches; each failure only shows at its own startup stage)

| # | change | result |
|---|---|---|
| 1 | hot 40, stream stage on | RuntimeError "unchecked tiered expert geometry" (guard 64..96) |
| 2 | stream stage off | same: the guard is on the dynamic-LRU path, which their launcher forces on |
| 3 | hot 64 (inside the guard) | CUDA OOM building the MTP draft: 343 MiB free, draft + KV + graphs need ~3-4 GB more per GPU |
| 4 | hot 32, tiered guard relaxed | "unchecked LRU warmup geometry" (second copy of the guard) |
| 5 | hot 40, all 3 guards relaxed, MAX_TOKENS 4 | "unchecked LRU token threshold" (warmup requires MAX_TOKENS == 16) |
| 6 | MAX_TOKENS back to 16 | stream stage: "borrowed pages too small 256 MiB < 400 MiB" |
| 7 | stream stage off | KV: 64k needs 1.31 GiB, 1.12 allotted |
| 8 | KV 1.45 GiB | runs: 15.1 of 16 GB per GPU |

VRAM stages in vLLM: weights + hot cache (12.2 GB/GPU at hot 40), then MTP draft (~2 GB/GPU), activation
profiling, KV, CUDA graphs (0.18 GiB). One hot slot = 116 MiB per GPU across 48 layers.

Swap: the kernel moved the PLE into swap by itself during load: PLE worker VmSwap 48.6 GB, RSS 2.1 GB;
engine workers ~1 GB each swapped at load, paged back on first use; no swap-in traffic during decode.
The planned process_madvise tool (local/results/q2x/pageout.py) was not needed.

## Results (pass 2 of 2; pass 1 within noise)

| | shallow | 8k | 32k | 61k | prefill 8k / 32k / 61k | short-prompt TTFT |
|---|---:|---:|---:|---:|---|---:|
| vLLM stack, hot 40, no stream stage | 30-32 | 36.0 | 37.9 | 34.9 | 667 / 1,086 / 1,356 | ~0.95 s |
| llama.cpp-qsa main, BELLS 240 slots | 72.5 | 70-71 | 69-71 | 62-64 | 400 / 390 / 367 | ~0.5 s |
| their 2x 3090, hot 84, stream stage | ~104 (131k run) | | | | 2,752 (131k) | |

MTP3 acceptance: shallow ~52% of drafted tokens, depth 69-77%. Output coherent.
Reading: decode half of ours (16% of each GPU's experts resident vs their 33%; misses over PCIe x8);
prefill 1.7-3.7x ours even without streaming prefill. Prefill is the transferable win.

## Gaps to their performance (work list; target ~66% of theirs = ~69 decode, ~1,800 prefill)

Ordered by what to do first. Items 1-4 need no hardware; 1 decides whether the rest is worth it.

1. Profile a decode step (nsys, CUDA graphs on) at hot 40: split time into GPU compute, UVA miss
   reads, all-reduce, host. Also get the hot-cache hit rate (look for stats/logging in
   compressed_tensors_moe_wna16.py / lru_map). Without this the items below are guesses.
2. Streaming prefill at low slot counts: give the stage a dedicated buffer instead of borrowing
   (QWEN38_STREAM_STAGE_BORROW=0; not in docker_serve.sh's pass-through list, add it), or lower
   QWEN38_STREAM_STAGE_BORROW_PER_SLOT. Costs ~400 MiB/GPU (~3-4 slots). Measures the 2,752 path.
3. Re-rank the static hot set for our traffic: configs/static_hot_cache_rankings.json comes from
   their workload; at 40 slots the choice matters more. Their LRU already adapts, the ranking is the
   starting set.
4. Short-prompt TTFT ~0.95 s for 31 tokens: find the fixed cost (PLE worker round trip, scheduler,
   graph selection).
5. More slots per GB (each 116 MiB/GPU): FP8 KV (--kv-cache-dtype fp8, ~0.7 GiB -> ~6 slots; breaks
   their BF16-KV quality contract, needs a check), MAX_NUM_BATCHED_TOKENS 4096 -> 2048, 32k context
   (~5 slots), measure the BF16 "sensitive" layers' GPU footprint and consider FP8 for them, and check
   whether the MTP draft's routed experts (~2 GB/GPU) can use the same host-UVA offload as the target.
6. Decode miss path: 5060 Ti runs PCIe Gen4 x8 (checked: nvidia-smi link gen 4 width 8) vs 3090
   Gen4 x16. If item 1 shows misses dominate, look at prefetching the next layer's likely experts
   (their Sep-25 DMA-ahead is prefill-only) and at the humming kernels' sm_120 tuning;
   QWEN38_TRITON_SKINNY is sm_86-only (no-op here), a sm_120 port may matter for small GEMMs.
7. Not available to us: their P2P custom all-reduce (needs SM-level peer access; our cards sit on
   different EPYC root complexes).
8. RTX 3060 12 GB (planned third card): TP3 is invalid (2 KV heads); the open question is whether
   vLLM can place something useful on it (pipeline stage, the MTP draft, or more expert cache).
   Research item, not a config flip.

## Resume / revert

- Resume: the launch block above. The GPUs must be free (nothing else serves by default now).
- Revert host changes when done: swapoff /dev/sdb, remove its fstab line, qm set 100 --delete scsi1,
  lvremove pve/vm-100-disk-0; RAM back with qm set 100 --memory 98304 (needs a VM restart).
- Disk: checkpoint 121 GiB + image 29.3 GB; delete both to reclaim (ask the user first).
