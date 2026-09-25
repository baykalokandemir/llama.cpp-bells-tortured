#!/usr/bin/env bash
# nsys kernel profile of MTP decode, to attribute the hyper-connection inject matmuls
B=/home/god/dev/llama.cpp-hcmmvf/build/bin
M=/opt/stacks/llm-stack/models/qwen38-flash-next-gsq/IQ3_XXS_HCQ8/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-HCQ8-00001-of-00002.gguf
H=/opt/stacks/llm-stack/models/qwen38-flash-next-mtp/MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf
O=/home/god/dev/llama.cpp-hcmmvf/local/results/hcprof-${TAG:-on}
mkdir -p $O
C="-ngl 99 -sm layer -ts 28,20 -c 65536 -lm mmap -lzm off -t 32 -tb 32 -fa on -ctk q8_0 -ctv q8_0 --fit off --parallel 1 --jinja --cpu-moe -ub 2048 -b 2048 --bells-slots 240 -md $H --spec-type draft-mtp -devd CUDA1 -ngld 99 --spec-draft-n-max 2"
env LLAMA_DRAFT_UBATCH=256 GGML_CUDA_DISABLE_GRAPHS=1 LD_LIBRARY_PATH=$B nsys profile -t cuda,nvtx -s none --cpuctxsw=none -o $O/mtp -f true \
  $B/llama-server -m $M --host 127.0.0.1 --port 8094 $C > $O/server.log 2>&1 &
for i in $(seq 1 300); do [ "$(curl -s -m 2 -o /dev/null -w %{http_code} http://127.0.0.1:8094/health)" = 200 ] && break; sleep 2; done
python3 - <<"PY"
import json, urllib.request
def post(b):
    r = urllib.request.Request("http://127.0.0.1:8094/completion", data=json.dumps(b).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=3600))
p = "<|im_start|>user\nExplain how non-uniform tensor quantization trades model size, accuracy, and inference speed.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
for n in (64, 300):
    t = post({"prompt": p, "n_predict": n, "temperature": 0, "cache_prompt": False})["timings"]
    print(n, t["predicted_per_second"], t["predicted_n"], t.get("draft_n"), t.get("draft_n_accepted"))
PY
pkill -INT -f "llama-server -m $M" ; sleep 30
ls -la $O
