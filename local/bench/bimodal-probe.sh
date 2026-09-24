#!/usr/bin/env bash
# restart the server N times; per request record ms/pass plus GPU telemetry sampled during decode
N=${N:-4}
B=/home/god/dev/llama.cpp-qsa/build/bin
M=/opt/stacks/llm-stack/models/qwen38-flash-next-gsq/IQ3_XXS_HCQ8/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-HCQ8-00001-of-00002.gguf
H=/opt/stacks/llm-stack/models/qwen38-flash-next-mtp/MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf
O=/home/god/dev/llama.cpp-qsa/local/results
for inst in $(seq 1 $N); do
  env LLAMA_DRAFT_UBATCH=256 LD_LIBRARY_PATH=$B $B/llama-server -m $M --host 127.0.0.1 --port 8094 -ngl 99 -sm layer -ts 28,20 -c 65536 -lm mmap -lzm off -t 32 -tb 32 -fa on -ctk q8_0 -ctv q8_0 --fit off --parallel 1 --jinja --cpu-moe-pinned -ub 2048 -b 2048 --bells-slots 240 -md $H --spec-type draft-mtp --spec-draft-n-max 2 -devd CUDA1 -ngld 99 > $O/bm-inst$inst-server.log 2>&1 &
  PID=$!
  for i in $(seq 1 300); do [ "$(curl -s -m 2 -o /dev/null -w %{http_code} http://127.0.0.1:8094/health)" = 200 ] && break; sleep 2; done
  python3 /home/god/dev/llama.cpp-qsa/local/bench/threadmon.py $PID $O/threads-inst$inst.txt &
  if [ -n "$EVICT_MAIN" ]; then python3 /home/god/dev/llama.cpp-qsa/local/bench/pagecache.py evict $M; sleep 5; fi
  grep -E "^pgscan_kswapd|^compact_daemon_wake" /proc/vmstat > $O/vmstat-inst$inst-before.txt
  python3 - $inst <<"PY"
import json, sys, subprocess, urllib.request, statistics as st
inst = sys.argv[1]
w = lambda u: "<|im_start|>user\n" + u + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
prose = w("Explain how non-uniform tensor quantization trades model size, accuracy, and inference speed.")
eog = [[t, False] for t in (248044, 248046, 248063, 248064, 248065)]
q = "index,clocks.sm,clocks.mem,pstate,power.draw,pcie.link.gen.current,pcie.link.width.current,clocks_throttle_reasons.active"
import time
for k in range(int(__import__("os").environ.get("NREQ","5"))):
    t_start = time.time()
    mon = subprocess.Popen(["nvidia-smi", "--query-gpu=" + q, "--format=csv,noheader,nounits", "-lms", "250"], stdout=subprocess.PIPE, text=True)
    r = urllib.request.Request("http://127.0.0.1:8094/completion", data=json.dumps({"prompt": prose, "n_predict": 300, "temperature": 0, "cache_prompt": False, "logit_bias": eog}).encode(), headers={"Content-Type": "application/json"})
    t = json.load(urllib.request.urlopen(r, timeout=3600))["timings"]
    mon.terminate(); out = mon.communicate()[0].strip().splitlines()
    g = {0: [], 1: []}
    for l in out:
        f = [x.strip() for x in l.split(",")]
        try: g[int(f[0])].append(f)
        except Exception: pass
    t_end = time.time()
    passes = t["predicted_n"] - t["draft_n_accepted"]
    ms = 1000 * t["predicted_n"] / t["predicted_per_second"] / passes
    summ = []
    for d in (0, 1):
        rows = g[d][2:-1] or g[d]
        sm = st.median(int(x[1]) for x in rows); mem = st.median(int(x[2]) for x in rows)
        gen = sorted(set(x[5] for x in rows)); ps = sorted(set(x[3] for x in rows)); thr = sorted(set(x[7] for x in rows))
        pw = st.median(float(x[4]) for x in rows)
        summ.append(f"g{d}: sm {sm} mem {mem} {"/".join(ps)} {pw:.0f}W pcie gen{"/".join(gen)} thr {"/".join(thr)}")
    print(f"{t_start:.2f} {t_end:.2f} inst{inst} req{k}: {t["predicted_per_second"]:6.2f} tok/s  {ms:5.1f} ms/pass | " + " | ".join(summ), flush=True)
PY
  grep -E "^pgscan_kswapd|^compact_daemon_wake" /proc/vmstat > $O/vmstat-inst$inst-after.txt
  echo "inst$inst kswapd scanned $(( $(awk "/pgscan/{print \$2}" $O/vmstat-inst$inst-after.txt) - $(awk "/pgscan/{print \$2}" $O/vmstat-inst$inst-before.txt) )) pages during requests"
  kill $PID; wait $PID 2>/dev/null
done
