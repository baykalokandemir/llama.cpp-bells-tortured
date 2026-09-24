#!/usr/bin/env bash
# mtpexact.sh LABEL extra-args... : greedy texts for prose/code/8k-doc prompts, saved as JSON
L=$1; shift
B=/home/god/dev/llama.cpp-qsa/build/bin
M=/opt/stacks/llm-stack/models/qwen38-flash-next-gsq/IQ3_XXS_HCQ8/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-HCQ8-00001-of-00002.gguf
O=/home/god/dev/llama.cpp-qsa/local/results
env LLAMA_DRAFT_UBATCH=256 LD_LIBRARY_PATH=$B $B/llama-server -m $M --host 127.0.0.1 --port 8094 -ngl 99 -sm layer -ts 28,20 -c 65536 -lm mmap -lzm off -t 32 -tb 32 -fa on -ctk q8_0 -ctv q8_0 --fit off --parallel 1 --jinja --cpu-moe-pinned -ub 2048 -b 2048 --bells-slots 240 "$@" > $O/$L-server.log 2>&1 &
PID=$!
for i in $(seq 1 300); do [ "$(curl -s -m 2 -o /dev/null -w %{http_code} http://127.0.0.1:8094/health)" = 200 ] && break; kill -0 $PID 2>/dev/null || { echo LOADFAIL; exit 2; }; sleep 2; done
python3 - $O/$L.json <<"PY"
import json, sys, urllib.request
def post(path, b):
    r = urllib.request.Request("http://127.0.0.1:8094" + path, data=json.dumps(b).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=3600))
w = lambda u: "<|im_start|>user\n" + u + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
toks = post("/tokenize", {"content": open("/home/god/dev/iq3-explore/filler3.txt").read()})["tokens"]
P = {"prose": w("Explain how non-uniform tensor quantization trades model size, accuracy, and inference speed."),
     "code": w("Implement a bounded thread-safe work queue in Python and explain the invariants."),
     "doc8k": w("Here is some documentation:\n\n" + post("/detokenize", {"tokens": toks[:8000]})["content"] + "\n\nSummarize the key points of the documentation above.")}
eog = [[t, False] for t in (248044, 248046, 248063, 248064, 248065)]
out = {}
for k, p in P.items():
    r = post("/completion", {"prompt": p, "n_predict": 300, "temperature": 0, "cache_prompt": False, "logit_bias": eog, "n_probs": 0})
    out[k] = {"text": r["content"], "draft_n": r["timings"].get("draft_n"), "draft_acc": r["timings"].get("draft_n_accepted")}
json.dump(out, open(sys.argv[1], "w"), indent=1)
PY
kill $PID; wait $PID 2>/dev/null
