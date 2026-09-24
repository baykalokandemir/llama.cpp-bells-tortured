import json, sys, urllib.request
port, ntok = sys.argv[1], int(sys.argv[2])
url = f"http://127.0.0.1:{port}"
def post(path, body, timeout=3600):
    r = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=timeout))
toks = post("/tokenize", {"content": open("/home/god/dev/iq3-explore/filler3.txt").read()})["tokens"]
body = post("/detokenize", {"tokens": toks[:ntok]})["content"]
p = "<|im_start|>user\nHere is some documentation:\n\n" + body + "\n\nSummarize the key points of the documentation above.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
eog = [[t, False] for t in (248044, 248046, 248063, 248064, 248065)]
r = post("/completion", {"prompt": p, "n_predict": 200, "temperature": 0, "cache_prompt": False, "logit_bias": eog})
t = r["timings"]
print(json.dumps({"prompt_n": t["prompt_n"], "prompt_tps": round(t["prompt_per_second"], 1), "decode_tps": round(t["predicted_per_second"], 2), "draft_n": t.get("draft_n"), "draft_acc": t.get("draft_n_accepted"), "text": r["content"][:120]}))
