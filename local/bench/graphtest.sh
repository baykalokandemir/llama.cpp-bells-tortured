#!/usr/bin/env bash
# usage: graphtest.sh LABEL BIN extra-server-args...   (short prose/code workload, graph stats on)
L=$1; B=$2; shift 2
M=/opt/stacks/llm-stack/models/qwen38-flash-next-gsq/IQ3_XXS_HCQ8/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-HCQ8-00001-of-00002.gguf
H=/opt/stacks/llm-stack/models/qwen38-flash-next-mtp/MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf
O=/home/god/dev/llama.cpp-graph/local/results
C="-ngl 99 -sm layer -ts 28,20 -c 65536 -lm mmap -lzm on -t 32 -tb 32 -fa on -ctk q8_0 -ctv q8_0 --fit off --parallel 1 --jinja --cpu-moe-pinned -ub 2048 -b 2048 --bells-slots 240 -md $H --spec-type draft-mtp -devd CUDA1 -ngld 99"
env LLAMA_DRAFT_UBATCH=256 GGML_CUDA_GRAPH_STATS=1 $EXTRA_ENV LD_LIBRARY_PATH=$B $B/llama-server -m $M --host 127.0.0.1 --port 8094 $C "$@" > $O/$L-server.log 2>&1 &
PID=$!
for i in $(seq 1 300); do [ "$(curl -s -m 2 -o /dev/null -w %{http_code} http://127.0.0.1:8094/health)" = 200 ] && break; kill -0 $PID 2>/dev/null || { echo LOADFAIL; exit 2; }; sleep 2; done
python3 - "$O/$L.json" <<"PY"
import json, sys, urllib.request, statistics as st
def post(b):
    r = urllib.request.Request("http://127.0.0.1:8094/completion", data=json.dumps(b).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=3600))
w = lambda u: "<|im_start|>user\n" + u + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
P = {"prose": w("Explain how non-uniform tensor quantization trades model size, accuracy, and inference speed."),
     "code": w("Implement a bounded thread-safe work queue in Python and explain the invariants.")}
eog = [[t, False] for t in (248044, 248046, 248063, 248064, 248065)]
body = lambda p, n: {"prompt": p, "n_predict": n, "temperature": 0, "cache_prompt": False, "logit_bias": eog}
post(body(P["prose"], 64))
rows = []
for i in range(3):
    for k, p in P.items():
        t = post(body(p, 300))["timings"]
        rows.append({"case": k, "decode_tps": round(t["predicted_per_second"], 2), "n_gen": t["predicted_n"], "draft_n": t.get("draft_n"), "draft_acc": t.get("draft_n_accepted")})
json.dump(rows, open(sys.argv[1], "w"), indent=1)
for k in P:
    v = [r["decode_tps"] for r in rows if r["case"] == k]
    print(k, "median", st.median(v), v)
PY
grep "cuda_graph_stats" $O/$L-server.log | tail -1
grep "graphs reused" $O/$L-server.log | tail -1 | sed "s/.*graphs/graphs/"
kill $PID; wait $PID 2>/dev/null
