#!/usr/bin/env bash
# MoE weighted-reduction fusion acceptance A/B on accwork.py (8 distinct short prompts + 2x 8k); off = GGML_CUDA_MOE_WR_FUSION=0
cd /home/god/dev/iq3-explore
R=/home/god/dev/llama.cpp-qsa/local/results/fr/rank.ids
H=/opt/stacks/llm-stack/models/qwen38-flash-next-mtp/MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf
BASE_ENV="GGML_CUDA_MOE_WR_LOG=1 LLAMA_ARG_BACKEND_SAMPLING=1 LLAMA_DRAFT_UBATCH=256 LLAMA_MTP_VOCAB=$R LLAMA_MTP_VOCAB_N=65536"
ARGS="-ngl 99 -sm layer -ts 28,20 -c 65536 -lm mmap -lzm off -t 32 -tb 32 -fa on -ctk q8_0 -ctv q8_0 --fit off --parallel 1 --jinja --cpu-moe-pinned -ub 2048 -b 2048 --bells-slots 240 -md $H --spec-type draft-mtp -devd CUDA1 -ngld 99 --spec-draft-n-max 2"
for arm in ${ARMS:-on1 off1 off2 on2}; do
  E="$BASE_ENV"; case $arm in off*) E="$E GGML_CUDA_MOE_WR_FUSION=0";; esac
  BIN=/home/god/dev/llama.cpp-moewr/build/bin CLIENT=/home/god/dev/llama.cpp-qsa/local/bench/accwork.py EXTRA_ENV="$E" /home/god/dev/llama.cpp-qsa/local/bench/depthcurve2.sh acc-$arm 0 1 $ARGS 2>&1 | cut -c1-240
done
echo ALLDONE
