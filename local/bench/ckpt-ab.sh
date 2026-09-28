#!/usr/bin/env bash
# Prompt-cache reuse A/B for configurable checkpoints (upstream PR #29463, server part only).
# Arms: def = stock layout (4+n_ubatch, 4); fine = LLAMA_CKPT_OFFSETS=u,512,128,4.
# Client: local/bench/cachetest.py (same 8k document, alternating questions, cache_prompt on).
cd /home/god/dev/iq3-explore
B=/home/god/dev/llama.cpp-ck/build/bin
M=/opt/stacks/llm-stack/models/qwen38-flash-next-gsq/IQ3_XXS_HCQ8/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-HCQ8-00001-of-00002.gguf
R=/home/god/dev/llama.cpp-qsa/local/results/fr/rank.ids
H=/opt/stacks/llm-stack/models/qwen38-flash-next-mtp/MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf
O=/home/god/dev/llama.cpp-qsa/local/results/ckpt; mkdir -p $O
BASE_ENV="LLAMA_ARG_BACKEND_SAMPLING=1 LLAMA_DRAFT_UBATCH=256 LLAMA_MTP_VOCAB=$R LLAMA_MTP_VOCAB_N=65536"
ARGS="-ngl 99 -sm layer -ts 28,20 -c 65536 -lm mmap -lzm off -t 32 -tb 32 -fa on -ctk q8_0 -ctv q8_0 --fit off --parallel 1 --jinja --cpu-moe-pinned -ub 2048 -b 2048 --bells-slots 240 -md $H --spec-type draft-mtp -devd CUDA1 -ngld 99 --spec-draft-n-max 2"
for arm in ${ARMS:-def1 fine1 fine2 def2}; do
  E="$BASE_ENV"; case $arm in fine*) E="$E LLAMA_CKPT_OFFSETS=u,512,128,4";; lean*) E="$E LLAMA_CKPT_OFFSETS=u,128,4";; esac
  [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && { echo "GPU busy before $arm"; exit 3; }
  env LD_LIBRARY_PATH=$B $E $B/llama-server -m $M --host 127.0.0.1 --port 8094 $ARGS > $O/$arm-server.log 2>&1 &
  P=$!
  for i in $(seq 1 300); do [ "$(curl -s -m 2 -o /dev/null -w %{http_code} http://127.0.0.1:8094/health)" = 200 ] && break; kill -0 $P 2>/dev/null || { echo "$arm LOADFAIL"; exit 2; }; sleep 2; done
  echo "=== $arm"
  python3 /home/god/dev/llama.cpp-qsa/local/bench/cachetest.py 8094 $P 2>&1 | sed "s/^/$arm /"
  grep -c "created context checkpoint\|create_checkpoint" $O/$arm-server.log | sed "s/^/$arm checkpoint-log-lines /"
  kill $P; wait $P 2>/dev/null; sleep 5
done
echo ALLDONE
