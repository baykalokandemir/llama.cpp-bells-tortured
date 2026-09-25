#!/usr/bin/env bash
# perf CPU profile of the decode thread during shallow MTP decode (needs kernel.perf_event_paranoid <= 1).
# usage: hostprof.sh TAG [extra env...]
TAG=${1:-main}; shift
B=${BIN:-/home/god/dev/llama.cpp-qsa/build/bin}
M=/opt/stacks/llm-stack/models/qwen38-flash-next-gsq/IQ3_XXS_HCQ8/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-HCQ8-00001-of-00002.gguf
H=/opt/stacks/llm-stack/models/qwen38-flash-next-mtp/MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf
R=/home/god/dev/llama.cpp-qsa/local/results/fr/rank.ids
O=/home/god/dev/llama.cpp-qsa/local/results/hostprof-$TAG; mkdir -p $O
C="-ngl 99 -sm layer -ts 28,20 -c 65536 -lm mmap -lzm off -t 32 -tb 32 -fa on -ctk q8_0 -ctv q8_0 --fit off --parallel 1 --jinja --cpu-moe-pinned -ub 2048 -b 2048 --bells-slots 240 -md $H --spec-type draft-mtp -devd CUDA1 -ngld 99 --spec-draft-n-max 2"
env LLAMA_DRAFT_UBATCH=256 LLAMA_MTP_VOCAB=$R LLAMA_MTP_VOCAB_N=65536 "$@" LD_LIBRARY_PATH=$B \
  $B/llama-server -m $M --host 127.0.0.1 --port 8094 $C > $O/server.log 2>&1 &
NP=$!
for i in $(seq 1 300); do [ "$(curl -s -m 2 -o /dev/null -w %{http_code} http://127.0.0.1:8094/health)" = 200 ] && break; sleep 2; done
REQ='{"prompt":"<|im_start|>user\nExplain how non-uniform tensor quantization trades model size, accuracy, and inference speed.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n","n_predict":N,"temperature":0,"cache_prompt":false}'
for n in 64 300; do curl -s http://127.0.0.1:8094/completion -d "${REQ/N/$n}" > /dev/null; done
if [ -n "$SHORT" ]; then
  ( for i in $(seq 1 12); do curl -s http://127.0.0.1:8094/completion -d "${REQ/N/1}" > $O/req-short-$i.json; done ) &
  CP=$!; sleep 0.2
  perf record -F 4000 --call-graph dwarf,16384 -p $NP -o $O/perf.data -- sleep 6 2> $O/perf-record.log
  wait $CP
  python3 -c "import json,glob;[print(f, round(json.load(open(f))[\"timings\"][\"prompt_ms\"],1)) for f in sorted(glob.glob(\"$O/req-short-*.json\"))]"
  pkill -INT -x llama-server; wait $NP; exit 0
fi
curl -s http://127.0.0.1:8094/completion -d "${REQ/N/600}" > $O/req.json &
CP=$!
sleep 1
perf record -F 4000 --call-graph dwarf,16384 -p $NP -o $O/perf.data -- sleep 4 2> $O/perf-record.log
wait $CP
python3 -c "import json;t=json.load(open('$O/req.json'))['timings'];print('decode tok/s', round(t['predicted_per_second'],2))"
pkill -INT -x llama-server; wait $NP
