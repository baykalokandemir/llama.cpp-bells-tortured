#!/usr/bin/env bash
# fixed-token A/B: batched-bench with 3 sequences (same matmul batch as a 3-token MTP verify), no sampling
B=/home/god/dev/llama.cpp-hcmmvf/build/bin
M=/opt/stacks/llm-stack/models/qwen38-flash-next-gsq/IQ3_XXS_HCQ8/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-HCQ8-00001-of-00002.gguf
O=/home/god/dev/llama.cpp-hcmmvf/local/results/bb; mkdir -p $O
C="-ngl 99 -sm layer -ts 28,20 -c 16384 -lm mmap -lzm off -t 32 -tb 32 -fa on -ctk q8_0 -ctv q8_0 --fit off --cpu-moe-pinned -ub 2048 -b 2048 --bells-slots 240 -npp 1024 -ntg 128 -npl 1,3"
for r in 1 2; do
  for arm in off on; do
    E=""; [ $arm = off ] && E="GGML_CUDA_MMVF_FALLBACK_MIN_ROWS=-1"
    env $E LD_LIBRARY_PATH=$B $B/llama-batched-bench -m $M $C > $O/$arm$r.log 2>&1
    echo "== $arm$r"; grep -E "^\|\s+[0-9]" $O/$arm$r.log
  done
done
echo ALLDONE
