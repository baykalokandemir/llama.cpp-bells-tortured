#!/usr/bin/env bash
# upstream BELLS (master 0275669ec + PR #28243 head, branch bells-upstream-mtp) vs our main, test config.
# Our-only env hooks (LLAMA_DRAFT_UBATCH, LLAMA_MTP_VOCAB) are no-ops upstream; kept identical per arm.
cd /home/god/dev/iq3-explore
R=/home/god/dev/llama.cpp-qsa/local/results/fr/rank.ids
H=/opt/stacks/llm-stack/models/qwen38-flash-next-mtp/MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf
BASE_ENV="LLAMA_ARG_BACKEND_SAMPLING=1 LLAMA_DRAFT_UBATCH=256 LLAMA_MTP_VOCAB=$R LLAMA_MTP_VOCAB_N=65536"
ARGS="-ngl 99 -sm layer -ts 28,20 -c 65536 -lm mmap -lzm off -t 32 -tb 32 -fa on -ctk q8_0 -ctv q8_0 --fit off --parallel 1 --jinja --cpu-moe-pinned -ub 2048 -b 2048 --bells-slots ${SLOTS:-240} -md $H --spec-type draft-mtp -devd CUDA1 -ngld 99 --spec-draft-n-max 2"
O=/home/god/dev/llama.cpp-qsa/local/results/bellsup; mkdir -p $O
for arm in ${ARMS:-up1 ours1 ours2 up2}; do
  case $arm in up*) B=/home/god/dev/llama.cpp-bellsup/build/bin;; next*) B=/home/god/dev/llama.cpp-bellsnext/build/bin;; port*) B=/home/god/dev/llama.cpp-bnport/build/bin;; restack*) B=/home/god/dev/llama.cpp-restack/build/bin;; *) B=/home/god/dev/llama.cpp-qsa/build/bin;; esac
  [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && { echo "GPU busy before $arm"; exit 3; }
  OUT=$O BIN=$B EXTRA_ENV="$BASE_ENV" ./../llama.cpp-qsa/local/bench/depthcurve2.sh bu-$arm${SLOTS:+-s$SLOTS} ${DEPTHS:-8192,32768,61440} 1 $ARGS 2>&1 | cut -c1-240
  sleep 5
done
echo ALLDONE
