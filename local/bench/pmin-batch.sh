#!/usr/bin/env bash
# p_min x draft-depth sweep on the qsa-slim build, 64k Q8_0, 240 slots, -ub 2048, MTP on CUDA1
B=/home/god/dev/llama.cpp-qsa/build/bin
M=/opt/stacks/llm-stack/models/qwen38-flash-next-gsq/IQ3_XXS_HCQ8/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-HCQ8-00001-of-00002.gguf
H=/opt/stacks/llm-stack/models/qwen38-flash-next-mtp/MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf
O=/home/god/dev/iq3-explore; R=/home/god/dev/llama.cpp-qsa/local/results
C="-ngl 99 -sm layer -ts 28,20 -c 65536 -lm mmap -lzm on -t 32 -tb 32 -fa on -ctk q8_0 -ctv q8_0 --fit off --parallel 1 --jinja --cpu-moe-pinned -ub 2048 -b 2048 --bells-slots 240 -md $H --spec-type draft-mtp -devd CUDA1 -ngld 99"
for arm in "2 0" "2 0.75" "3 0.5" "3 0.75" "3 0.9" "4 0.75" "4 0.9"; do
  set -- $arm; N=$1; P=$2; L=pmin-n$N-p$P
  env LLAMA_DRAFT_UBATCH=256 LD_LIBRARY_PATH=$B $B/llama-server -m $M --host 127.0.0.1 --port 8094 $C --spec-draft-n-max $N --spec-draft-p-min $P > $O/$L-server.log 2>&1 &
  PID=$!
  for i in $(seq 1 300); do [ "$(curl -s -m 2 -o /dev/null -w %{http_code} http://127.0.0.1:8094/health)" = 200 ] && break; kill -0 $PID 2>/dev/null || break; sleep 2; done
  python3 /home/god/dev/llama.cpp-qsa/local/bench/pmin.py 8094 $R/$L.json > $O/$L.out 2>&1 || echo "$L FAILED" >> $O/pmin-batch.log
  kill $PID; wait $PID 2>/dev/null
  echo "$L done" >> $O/pmin-batch.log
done
echo ALLDONE >> $O/pmin-batch.log
