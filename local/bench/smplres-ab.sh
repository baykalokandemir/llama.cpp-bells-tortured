#!/usr/bin/env bash
# sampler-reserve A/B with --backend-sampling in both arms; off = LLAMA_SAMPLER_ALWAYS_RESERVE=1
cd /home/god/dev/iq3-explore
R=/home/god/dev/llama.cpp-qsa/local/results/fr/rank.ids
H=/opt/stacks/llm-stack/models/qwen38-flash-next-mtp/MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf
BASE_ENV="LLAMA_ARG_BACKEND_SAMPLING=1 LLAMA_DRAFT_UBATCH=256 LLAMA_MTP_VOCAB=$R LLAMA_MTP_VOCAB_N=65536"
ARGS="-ngl 99 -sm layer -ts 28,20 -c 65536 -lm mmap -lzm off -t 32 -tb 32 -fa on -ctk q8_0 -ctv q8_0 --fit off --parallel 1 --jinja --cpu-moe-pinned -ub 2048 -b 2048 --bells-slots 240 -md $H --spec-type draft-mtp -devd CUDA1 -ngld 99 --spec-draft-n-max 2"
for arm in ${ARMS:-on1 off1 off2 on2}; do
  E="$BASE_ENV"; case $arm in off*) E="$E LLAMA_SAMPLER_ALWAYS_RESERVE=1";; esac
  BIN=/home/god/dev/llama.cpp-smplres/build/bin EXTRA_ENV="$E" ./depthcurve2.sh sr-$arm ${DEPTHS:-8192,32768} 1 $ARGS 2>&1 | cut -c1-240
done
echo ALLDONE
