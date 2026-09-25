#!/usr/bin/env bash
# #12 first step: GenerelSchwerz moe-cache fork vs our BELLS main, 64k config, same slots per layer.
cd /home/god/dev/iq3-explore
H=/opt/stacks/llm-stack/models/qwen38-flash-next-mtp/MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf
COMMON="-ngl 99 -sm layer -ts 28,20 -c 65536 -lzm off -t 32 -tb 32 -fa on -ctk q8_0 -ctv q8_0 --fit off --parallel 1 --jinja -ub 2048 -b 2048 -md $H --spec-type draft-mtp -devd CUDA1 -ngld 99 --spec-draft-n-max 2"
SLOTS=${SLOTS:-240}
for arm in ${ARMS:-mc mcx bells}; do
  case $arm in
    bells) B=/home/god/dev/llama.cpp-qsa/build/bin; E="LLAMA_DRAFT_UBATCH=256"
           A="$COMMON -lm mmap --cpu-moe-pinned --bells-slots $SLOTS" ;;
    mc)    B=/home/god/dev/llama.cpp-moecache/build/bin; E=""
           A="$COMMON ${LM:--lm none} --cpu-moe --moe-expert-cache-size $SLOTS --moe-expert-cache-host-pinned-mb 0 -ubd 256 $XARGS" ;;
    mclru) B=/home/god/dev/llama.cpp-moecache/build/bin; E="GGML_CUDA_MOE_FREQUENCY=0"
           A="$COMMON ${LM:--lm none} --cpu-moe --moe-expert-cache-size $SLOTS --moe-expert-cache-host-pinned-mb 0 -ubd 256 $XARGS" ;;
    mcx)   B=/home/god/dev/llama.cpp-moecache/build/bin; E=""
           A="$COMMON ${LM:--lm none} --cpu-moe --moe-expert-cache-size $SLOTS --moe-expert-cache-host-pinned-mb 0 -ubd 256 --moe-early-router --decode-overlap --decode-boundary-overlap $XARGS" ;;
  esac
  BIN=$B EXTRA_ENV="$E" ./depthcurve2.sh mc-$arm-$SLOTS ${DEPTHS:-8192,32768} 1 $A 2>&1 | cut -c1-200
  { grep "moe-grouped-paths" dc-mc-$arm-$SLOTS-server.log | grep -v "decode_grouped=0 " | tail -1; grep "bells_timing" dc-mc-$arm-$SLOTS-server.log | tail -1; } | cut -c1-400 | sed "s/^/$arm LOG /"
done
echo ALLDONE
