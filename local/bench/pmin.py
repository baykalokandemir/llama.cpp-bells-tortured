# p_min sweep workload: prose and code short prompts, real-text 8k and 32k prompts
import json, sys, urllib.request
port, out = sys.argv[1], sys.argv[2]
url = f"http://127.0.0.1:{port}"
def post(path, body, timeout=7200):
    r = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=timeout))
eog = [[t, False] for t in (248044, 248046, 248063, 248064, 248065)]
wrap = lambda u: "<|im_start|>user\n" + u + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
prose = wrap("Explain how non-uniform tensor quantization trades model size, accuracy, and inference speed.")
code = wrap("Implement a bounded thread-safe work queue in Python and explain the invariants.")
toks = post("/tokenize", {"content": open("/home/god/dev/iq3-explore/filler3.txt").read()})["tokens"]
def doc(n, off):
    return wrap("Here is some documentation:\n\n" + post("/detokenize", {"tokens": toks[off:off + n]})["content"] + "\n\nSummarize the key points of the documentation above.")
rows = []
def run(label, prompt, n):
    r = post("/completion", {"prompt": prompt, "n_predict": n, "temperature": 0, "cache_prompt": False, "logit_bias": eog})
    t = r["timings"]
    row = {"case": label, "prompt_n": t["prompt_n"], "prompt_tps": round(t["prompt_per_second"], 1),
           "decode_tps": round(t["predicted_per_second"], 2), "n_gen": t["predicted_n"],
           "draft_n": t.get("draft_n"), "draft_acc": t.get("draft_n_accepted")}
    rows.append(row); print(json.dumps(row), flush=True)
run("warmup", prose, 64)
for i in range(3):
    run("prose", prose, 300)
    run("code", code, 300)
run("depth8k", doc(8000, 0), 200)
run("depth8k", doc(8000, 7919), 200)
run("depth32k", doc(32000, 0), 200)
json.dump(rows, open(out, "w"), indent=1)
