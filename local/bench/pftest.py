import json, sys, time, urllib.request
port, out, ntok = sys.argv[1], sys.argv[2], int(sys.argv[3])
url = f"http://127.0.0.1:{port}"
def post(path, body, timeout=3600):
    r = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=timeout))
filler = open("/home/god/dev/iq3-explore/filler.txt").read()
toks = post("/tokenize", {"content": filler})["tokens"]
body = post("/detokenize", {"tokens": toks[:ntok]})["content"]
eog = [[t, False] for t in (248044, 248046, 248063, 248064, 248065)]
short = "<|im_start|>user\nExplain how non-uniform tensor quantization trades model size, accuracy, and inference speed.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
longp = "<|im_start|>user\nHere is some documentation:\n\n" + body + "\n\nSummarize the key points of the documentation above.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
res = {}
post("/completion", {"prompt": short, "n_predict": 64, "temperature": 0, "cache_prompt": False, "logit_bias": eog})
seq = [("short", short, 300), ("short_b", short, 300), ("long", longp, 200), ("long2", longp, 200), ("short2", short, 300), ("short2_b", short, 300)]
for name, p, n in seq:
    r = post("/completion", {"prompt": p, "n_predict": n, "temperature": 0, "cache_prompt": False, "logit_bias": eog})
    t = r["timings"]
    res[name] = {"prompt_n": t["prompt_n"], "prompt_tps": round(t["prompt_per_second"], 1), "decode_tps": round(t["predicted_per_second"], 2),
                 "draft_n": t.get("draft_n"), "draft_acc": t.get("draft_n_accepted")}
    print(name, res[name], flush=True)
json.dump(res, open(out, "w"))
