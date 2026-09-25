import sys, json
sys.path.insert(0, "/home/god/dev/llama.cpp-qsa/gguf-py")
from gguf import GGUFReader
r = GGUFReader("/opt/stacks/llm-stack/models/qwen38-flash-next-gsq/IQ3_XXS_HCQ8/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-HCQ8-00001-of-00002.gguf")
def arr(name):
    f = r.fields[name]
    return [bytes(f.parts[i]).decode("utf-8") for i in f.data]
toks = arr("tokenizer.ggml.tokens"); merges = arr("tokenizer.ggml.merges")
tj = json.load(open("/opt/stacks/llm-stack/models/vllm/Qwen3.8-27B-INT5-Flat/tokenizer.json"))
hv = {v: k for k, v in tj["model"]["vocab"].items()}
for a in tj["added_tokens"]:
    hv[a["id"]] = a["content"]
hm = [m if isinstance(m, str) else " ".join(m) for m in tj["model"]["merges"]]
diff = [i for i in range(len(toks)) if hv.get(i) != toks[i]]
print("gguf tokens", len(toks), "hf", len(hv), "id mismatches", len(diff), diff[:5], [(toks[i], hv.get(i)) for i in diff[:5]])
print("merges", len(merges), len(hm), merges == hm)
print("pre", r.fields.get("tokenizer.ggml.pre") and bytes(r.fields["tokenizer.ggml.pre"].parts[-1]).decode())
