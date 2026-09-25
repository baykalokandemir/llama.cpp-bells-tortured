# Acceptance workload: 8 distinct short prompts (300 tokens each) + 2 real-text 8k prompts (200 tokens).
# Used as CLIENT for depthcurve2.sh: accwork.py PORT OUT [ignored...]. Prints one JSON row per request.
import os, json, sys, urllib.request, hashlib
port, out = sys.argv[1], sys.argv[2]
url = f"http://127.0.0.1:{port}"
def post(path, body, timeout=7200):
    r = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=timeout))
eog = [[t, False] for t in (248044, 248046, 248063, 248064, 248065)]
wrap = lambda u: "<|im_start|>user\n" + u + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
short = [
    ("prose",     "Explain how non-uniform tensor quantization trades model size, accuracy, and inference speed."),
    ("code",      "Implement a bounded thread-safe work queue in Python and explain the invariants."),
    ("math",      "Prove that the square root of 2 is irrational, then generalise the argument to any prime p."),
    ("list",      "Give a numbered checklist for migrating a PostgreSQL 14 database to PostgreSQL 17 with minimal downtime."),
    ("letter",    "Write a polite but firm letter to a landlord about a heating system that has been broken for three weeks."),
    ("translate", "Translate into German and then French: 'The committee postponed the vote because two members were ill.' Explain any grammar choices."),
    ("refactor",  "Refactor this JavaScript to use async/await and add error handling:\nfunction load(u, cb){ fetch(u).then(r=>r.json()).then(d=>cb(null,d)).catch(e=>cb(e)); }"),
    ("story",     "Write the opening of a short science-fiction story set on a generation ship where the crew has forgotten it is a ship."),
]
toks = post("/tokenize", {"content": open(os.environ.get("FILLER", "filler3.txt")).read()})["tokens"]
def doc(n, off):
    return wrap("Here is some documentation:\n\n" + post("/detokenize", {"tokens": toks[off:off + n]})["content"] + "\n\nSummarize the key points of the documentation above.")
rows = []
def run(label, prompt, n):
    r = post("/completion", {"prompt": prompt, "n_predict": n, "temperature": 0, "cache_prompt": False, "logit_bias": eog})
    t = r["timings"]
    row = {"case": label, "prompt_n": t["prompt_n"], "prompt_tps": round(t["prompt_per_second"], 1),
           "decode_tps": round(t["predicted_per_second"], 2), "n_gen": t["predicted_n"], "decode_ms": round(t["predicted_ms"], 1),
           "draft_n": t.get("draft_n"), "draft_acc": t.get("draft_n_accepted"), "sha": hashlib.sha1(r["content"].encode()).hexdigest()[:12]}
    rows.append(row); print(json.dumps(row), flush=True)
run("warmup", wrap(short[0][1]), 64)
for lab, u in short:
    run(lab, wrap(u), 300)
run("doc8k-a", doc(8000, 0), 200)
run("doc8k-b", doc(8000, 40000), 200)
S = [r for r in rows if r["case"] != "warmup"]
tot = {"case": "TOTAL", "n_gen": sum(r["n_gen"] for r in S), "decode_ms": round(sum(r["decode_ms"] for r in S), 1),
       "draft_n": sum(r["draft_n"] or 0 for r in S), "draft_acc": sum(r["draft_acc"] or 0 for r in S)}
tot["decode_tps"] = round(tot["n_gen"] / tot["decode_ms"] * 1000, 2)
tot["acc_rate"] = round(tot["draft_acc"] / max(1, tot["draft_n"]), 4)
rows.append(tot); print(json.dumps(tot), flush=True)
json.dump(rows, open(out, "w"), indent=1)
