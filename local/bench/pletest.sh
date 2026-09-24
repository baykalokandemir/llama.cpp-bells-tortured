#!/usr/bin/env bash
# PLE page-cache / fault accounting around prefill requests (main build, 64k, 240 slots, ub 2048, MTP)
B=${BIN:-/home/god/dev/llama.cpp-qsa/build/bin}
D=/opt/stacks/llm-stack/models/qwen38-flash-next-gsq/IQ3_XXS_HCQ8
M=$D/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-HCQ8-00001-of-00002.gguf
PLE=$D/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-HCQ8-00002-of-00002.gguf
H=/opt/stacks/llm-stack/models/qwen38-flash-next-mtp/MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf
L=${LABEL:-ple}; O=/home/god/dev/llama.cpp-qsa/local/results; PC=/home/god/dev/llama.cpp-qsa/local/bench/pagecache.py
python3 $PC evict $PLE
if [ -n "$PREWARM" ]; then t0=$(date +%s); cat $PLE > /dev/null; echo "prewarm took $(($(date +%s)-t0)) s"; fi
env LLAMA_DRAFT_UBATCH=256 LD_LIBRARY_PATH=$B $B/llama-server -m $M --host 127.0.0.1 --port 8094 -ngl 99 -sm layer -ts 28,20 -c 65536 -t 32 -tb 32 -fa on -ctk q8_0 -ctv q8_0 --fit off --parallel 1 --jinja --cpu-moe-pinned -ub 2048 -b 2048 --bells-slots 240 -md $H --spec-type draft-mtp --spec-draft-n-max 2 -devd CUDA1 -ngld 99 "$@" > $O/$L-server.log 2>&1 &
PID=$!
for i in $(seq 1 300); do [ "$(curl -s -m 2 -o /dev/null -w %{http_code} http://127.0.0.1:8094/health)" = 200 ] && break; kill -0 $PID 2>/dev/null || { echo LOADFAIL; exit 2; }; sleep 2; done
SPID=$(pgrep -f "llama-server -m $M" | head -1)
stat() { awk "{print \$12}" /proc/$SPID/stat; }
rb() { awk "/^read_bytes/{print \$2}" /proc/$SPID/io; }
echo "after load: $(python3 $PC resident $PLE)"
python3 - $SPID $PC $PLE <<"PY"
import json, sys, urllib.request, subprocess, time
pid, pc, ple = sys.argv[1], sys.argv[2], sys.argv[3]
def post(path, b, timeout=7200):
    r = urllib.request.Request("http://127.0.0.1:8094" + path, data=json.dumps(b).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=timeout))
def counters():
    majflt = int(open(f"/proc/{pid}/stat").read().split(")")[1].split()[9])
    rb = int([l for l in open(f"/proc/{pid}/io") if l.startswith("read_bytes")][0].split()[1])
    return majflt, rb
toks = post("/tokenize", {"content": open("/home/god/dev/iq3-explore/filler3.txt").read()})["tokens"]
w = lambda u: "<|im_start|>user\n" + u + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
doc = lambda n, off: w("Here is some documentation:\n\n" + post("/detokenize", {"tokens": toks[off:off+n]})["content"] + "\n\nSummarize the key points of the documentation above.")
eog = [[t, False] for t in (248044, 248046, 248063, 248064, 248065)]
for label, n, off in [("8k-cold", 8000, 0), ("8k-again", 8000, 0), ("32k-new", 32000, 20000)]:
    f0, r0 = counters(); t0 = time.time()
    t = post("/completion", {"prompt": doc(n, off), "n_predict": 32, "temperature": 0, "cache_prompt": False, "logit_bias": eog})["timings"]
    f1, r1 = counters()
    res = subprocess.run(["python3", pc, "resident", ple], capture_output=True, text=True).stdout.strip().split(": ")[1]
    print(f"{label:9s} prompt {t["prompt_n"]:6d} tok  {t["prompt_per_second"]:6.1f} tok/s  majflt +{f1-f0:7d}  disk read +{(r1-r0)/2**20:8.1f} MiB  PLE {res}", flush=True)
PY
kill $PID; wait $PID 2>/dev/null
