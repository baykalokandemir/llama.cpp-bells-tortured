#!/usr/bin/env python3
# vLLM counterpart of llama.cpp-qsa local/bench/depthcurve2.py: same short prompt, same filler text and
# per-depth slicing, greedy, EOS suppressed. Prefill = prompt_tokens / TTFT; decode = (completion_tokens - 1)
# / (t_last_chunk - t_first_chunk), token counts from the usage block (never from SSE chunk counts, which
# undercount with speculative decoding). Draft acceptance from /metrics deltas.
# usage: vllm_depthcurve.py PORT OUT.json DEPTHS REPS [MODEL]
import json, os, re, sys, time, urllib.request

port, out = sys.argv[1], sys.argv[2]
depths = [int(x) for x in sys.argv[3].split(",") if x]
reps = int(sys.argv[4]) if len(sys.argv) > 4 else 1
model = sys.argv[5] if len(sys.argv) > 5 else "Qwen3.8-Flash-Next"
url = f"http://127.0.0.1:{port}"


def post(path, body, timeout=7200):
    r = urllib.request.Request(url + path, data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=timeout))


def metrics():
    txt = urllib.request.urlopen(url + "/metrics", timeout=30).read().decode()
    vals = {}
    for key in ("vllm:spec_decode_num_draft_tokens_total", "vllm:spec_decode_num_accepted_tokens_total",
                "vllm:spec_decode_num_drafts_total"):
        m = re.findall(r"^" + re.escape(key) + r"(?:\{[^}]*\})? ([0-9.e+]+)$", txt, re.M)
        vals[key.split(":")[1].replace("spec_decode_", "")] = sum(float(x) for x in m) if m else None
    return vals


def run(label, prompt, n):
    m0 = metrics()
    body = {"model": model, "prompt": prompt, "max_tokens": n, "temperature": 0, "ignore_eos": True,
            "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(url + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); t_first = t_last = None; usage = None; text = []
    with urllib.request.urlopen(req, timeout=7200) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            if d.get("usage"):
                usage = d["usage"]
            ch = d.get("choices") or []
            if ch and ch[0].get("text"):
                now = time.time()
                t_first = t_first or now
                t_last = now
                text.append(ch[0]["text"])
    m1 = metrics()
    pn, cn = usage["prompt_tokens"], usage["completion_tokens"]
    row = {"case": label, "prompt_n": pn, "ttft_s": round(t_first - t0, 3),
           "prompt_tps": round(pn / (t_first - t0), 1),
           "decode_tps": round((cn - 1) / (t_last - t_first), 2) if cn > 1 and t_last > t_first else None,
           "completion_n": cn,
           "draft_n": (m1["num_draft_tokens_total"] - m0["num_draft_tokens_total"]) if m0["num_draft_tokens_total"] is not None else None,
           "draft_acc": (m1["num_accepted_tokens_total"] - m0["num_accepted_tokens_total"]) if m0["num_accepted_tokens_total"] is not None else None,
           "head": "".join(text)[:60]}
    print(json.dumps(row), flush=True)
    return row


rows = []
filler = open(os.environ.get("FILLER", "/home/god/dev/iq3-explore/filler3.txt")).read()
toks = post("/tokenize", {"model": model, "prompt": filler, "add_special_tokens": False})["tokens"]
short = ("<|im_start|>user\nExplain how non-uniform tensor quantization trades model size, accuracy, and "
         "inference speed.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
rows.append(run("warmup", short, 64))
for i in range(3):
    rows.append(run("shallow", short, 300))
for d in depths:
    for k in range(reps):
        off = (k * 7919) % max(1, len(toks) - d)
        body = post("/detokenize", {"model": model, "tokens": toks[off:off + d]})["prompt"]
        p = ("<|im_start|>user\nHere is some documentation:\n\n" + body +
             "\n\nSummarize the key points of the documentation above.<|im_end|>\n<|im_start|>assistant\n"
             "<think>\n\n</think>\n\n")
        rows.append(run(f"depth{d}", p, 200))
json.dump(rows, open(out, "w"), indent=1)
