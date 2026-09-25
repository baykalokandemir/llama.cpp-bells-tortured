import os, json, sys, urllib.request, hashlib
port, out = sys.argv[1], sys.argv[2]
depths = [int(x) for x in sys.argv[3].split(",")]
reps = int(sys.argv[4]) if len(sys.argv) > 4 else 2
url = f"http://127.0.0.1:{port}"
def post(path, body, timeout=7200):
    r = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=timeout))
eog = [[t, False] for t in (248044, 248046, 248063, 248064, 248065)]
toks = post("/tokenize", {"content": open(os.environ.get("FILLER", "filler3.txt")).read()})["tokens"]
short = "<|im_start|>user\nExplain how non-uniform tensor quantization trades model size, accuracy, and inference speed.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
rows = []
def run(label, prompt, n):
    r = post("/completion", {"prompt": prompt, "n_predict": n, "temperature": 0, "cache_prompt": False, "logit_bias": eog})
    t = r["timings"]
    row = {"case": label, "prompt_n": t["prompt_n"], "prompt_tps": round(t["prompt_per_second"], 1),
           "decode_tps": round(t["predicted_per_second"], 2), "draft_n": t.get("draft_n"), "draft_acc": t.get("draft_n_accepted"), "sha": hashlib.sha1(r["content"].encode()).hexdigest()[:12]}
    rows.append(row); print(json.dumps(row), flush=True)
run("warmup", short, 64)
for i in range(3):
    run("shallow", short, 300)
for d in depths:
    for k in range(reps):
        off = (k * 7919) % max(1, len(toks) - d)   # different text per rep
        body = post("/detokenize", {"tokens": toks[off:off + d]})["content"]
        p = "<|im_start|>user\nHere is some documentation:\n\n" + body + "\n\nSummarize the key points of the documentation above.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        run(f"depth{d}", p, 200)
json.dump(rows, open(out, "w"), indent=1)
