#!/usr/bin/env bash
# nsys timeline of MTP decode with CUDA graphs on (node-level trace): where does the GPU sit idle,
# and how much of that idle time sits between a BELLS routing readback (D2H) and the next launch.
# usage: bellsprof.sh TAG [extra env...]; analysis: bellsprof.py DIR
TAG=${1:-main}; shift
B=${BIN:-/home/god/dev/llama.cpp-qsa/build/bin}
M=/opt/stacks/llm-stack/models/qwen38-flash-next-gsq/IQ3_XXS_HCQ8/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-HCQ8-00001-of-00002.gguf
H=/opt/stacks/llm-stack/models/qwen38-flash-next-mtp/MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf
R=/home/god/dev/llama.cpp-qsa/local/results/fr/rank.ids
O=/home/god/dev/llama.cpp-qsa/local/results/bellsprof-$TAG; mkdir -p $O
C="-ngl 99 -sm layer -ts 28,20 -c 65536 -lm mmap -lzm off -t 32 -tb 32 -fa on -ctk q8_0 -ctv q8_0 --fit off --parallel 1 --jinja --cpu-moe-pinned -ub 2048 -b 2048 --bells-slots ${SLOTS:-240} -md $H --spec-type draft-mtp -devd CUDA1 -ngld 99 --spec-draft-n-max 2"
env LLAMA_DRAFT_UBATCH=256 LLAMA_MTP_VOCAB=$R LLAMA_MTP_VOCAB_N=65536 "$@" LD_LIBRARY_PATH=$B \
  nsys profile -t cuda,osrt -s none --cpuctxsw=none --cuda-graph-trace=${GTRACE:-node} -o $O/prof -f true \
  $B/llama-server -m $M --host 127.0.0.1 --port 8094 $C > $O/server.log 2>&1 &
NP=$!
for i in $(seq 1 300); do [ "$(curl -s -m 2 -o /dev/null -w %{http_code} http://127.0.0.1:8094/health)" = 200 ] && break; sleep 2; done
python3 - "$O/windows.json" <<"PY"
import json, sys, time, urllib.request
def post(path, b):
    r = urllib.request.Request("http://127.0.0.1:8094" + path, data=json.dumps(b).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=7200))
eog = [[t, False] for t in (248044, 248046, 248063, 248064, 248065)]
short = "<|im_start|>user\nExplain how non-uniform tensor quantization trades model size, accuracy, and inference speed.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
out = []
def run(label, prompt, n):
    r = post("/completion", {"prompt": prompt, "n_predict": n, "temperature": 0, "cache_prompt": False, "logit_bias": eog})
    end = time.time_ns(); t = r["timings"]
    w = {"case": label, "end_ns": end, "decode_ms": t["predicted_ms"], "n_gen": t["predicted_n"],
         "draft_n": t.get("draft_n"), "draft_acc": t.get("draft_n_accepted"), "decode_tps": t["predicted_per_second"], "prompt_n": t["prompt_n"]}
    out.append(w); print(json.dumps(w), flush=True)
run("warmup", short, 64)
run("shallow", short, 300)
run("shallow2", short, 300)
json.dump(out, open(sys.argv[1], "w"), indent=1)
PY
pkill -INT -x llama-server; wait $NP   # one interrupt only: a second one makes the server hang in exit
nsys export -t sqlite -f true -o $O/prof.sqlite $O/prof.nsys-rep >/dev/null 2>&1
python3 /home/god/dev/llama.cpp-qsa/local/bench/bellsprof.py $O; python3 /home/god/dev/llama.cpp-qsa/local/bench/bellsgap.py $O
