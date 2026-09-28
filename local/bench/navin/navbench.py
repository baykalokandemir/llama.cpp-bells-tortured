#!/usr/bin/env python3
# Client for the Navin AD-4.27 apples-to-apples run: two 2048-token prompts, 192 generated
# tokens, greedy, prompt cache off. Prints one JSON line per request.
import json, sys, time, urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080"
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
N_IN, N_OUT = 2048, 192

def post(path, body, timeout=900):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

def build(system, ask, filler):
    # Grow/shrink the filler until the templated prompt is exactly N_IN tokens.
    lo, hi = 0, len(filler)
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": filler[:mid] + "\n\n" + ask}]
        text = post("/apply-template", {"messages": msgs})["prompt"]
        toks = post("/tokenize", {"content": text, "add_special": False})["tokens"]
        if len(toks) <= N_IN:
            best = toks; lo = mid + 1
        else:
            hi = mid - 1
    assert best is not None and len(best) >= N_IN - 8, len(best or [])
    return best

code = open("/home/god/dev/llama.cpp-bellsup/src/llama-sampler.cpp").read()
plan = open("/opt/stacks/llm-stack/AGENTS.md").read()

prompts = {
    "coding": build("You are a senior C++ engineer.",
                    "Review the code above. Point out the two most serious bugs or risks and show a corrected version of the affected function.",
                    code),
    "backup": build("You are a careful infrastructure engineer.",
                    "Using the environment described above, write a concrete backup plan: what to back up, how often, where to, and how to test restores.",
                    plan),
}

def run(name, toks, n_out):
    t0 = time.time()
    r = post("/completion", {"prompt": toks, "n_predict": n_out, "temperature": 0, "top_k": 1,
                             "cache_prompt": False, "ignore_eos": True, "seed": 1})
    t = r.get("timings", {})
    out = {"prompt": name, "n_in": len(toks), "wall_s": round(time.time() - t0, 2),
           "pp_tps": round(t.get("prompt_per_second", 0), 2), "n_pp": t.get("prompt_n"),
           "tg_tps": round(t.get("predicted_per_second", 0), 2), "n_tg": t.get("predicted_n"),
           "draft_n": t.get("draft_n"), "draft_acc": t.get("draft_n_accepted"),
           "head": r.get("content", "")[:80]}
    print(json.dumps(out), flush=True)
    return out

run("warmup", prompts["coding"][:64], 16)
for i in range(REPS):
    for name in ("coding", "backup"):
        run(name, prompts[name], N_OUT)
