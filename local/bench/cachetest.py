# Prompt-cache check: same 8k document with different questions, cache_prompt on.
# Prints prompt_n (tokens actually processed), timings, output hash, VRAM and server RSS per request.
import json, sys, urllib.request, hashlib, subprocess, os
port, pid = sys.argv[1], sys.argv[2]
url = f"http://127.0.0.1:{port}"
def post(path, body, timeout=7200):
    r = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=timeout))
def mem():
    v = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.split()
    rss = int(open(f"/proc/{pid}/status").read().split("VmRSS:")[1].split()[0]) // 1024
    return f"vram {'/'.join(v)} MiB, rss {rss} MiB"
eog = [[t, False] for t in (248044, 248046, 248063, 248064, 248065)]
toks = post("/tokenize", {"content": open(os.environ.get("FILLER", "filler3.txt")).read()})["tokens"]
doc = post("/detokenize", {"tokens": toks[:8000]})["content"]
wrap = lambda q: "<|im_start|>user\nHere is some documentation:\n\n" + doc + "\n\n" + q + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
qa = "Summarize the key points of the documentation above."
qb = "List every command-line flag mentioned in the documentation above."
print("start", mem(), flush=True)
for label, q in (("A", qa), ("B", qb), ("A-again", qa), ("B-again", qb)):
    r = post("/completion", {"prompt": wrap(q), "n_predict": 120, "temperature": 0, "cache_prompt": True, "logit_bias": eog})
    t = r["timings"]
    print(json.dumps({"req": label, "prompt_n": t["prompt_n"], "prompt_ms": round(t["prompt_ms"]), "decode_tps": round(t["predicted_per_second"], 2),
                      "cached": r.get("tokens_cached"), "sha": hashlib.sha1(r["content"].encode()).hexdigest()[:12]}), mem(), flush=True)
