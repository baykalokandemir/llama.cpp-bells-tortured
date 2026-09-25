# token frequency ranking for FR-Spec drafting: SlimPajama text + code, Qwen3.8 tokenizer
# usage: frcount.py OUT_PREFIX [slimpajama_docs] [code_mb]
import sys, glob, os, json, collections
import pyarrow.parquet as pq
from tokenizers import Tokenizer

out = sys.argv[1]
n_docs = int(sys.argv[2]) if len(sys.argv) > 2 else 200000
code_mb = int(sys.argv[3]) if len(sys.argv) > 3 else 60
tok = Tokenizer.from_file("/opt/stacks/llm-stack/models/vllm/Qwen3.8-27B-INT5-Flat/tokenizer.json")
cnt = collections.Counter()
ntok = {"web": 0, "code": 0}

def add(texts, key):
    for enc in tok.encode_batch(texts, add_special_tokens=False):
        cnt.update(enc.ids)
        ntok[key] += len(enc.ids)

pf = pq.ParquetFile("/home/god/moon/archeval-slop/data_processing/slimpajama_2B_shuffled/part-00000-of-00004.parquet")
col = [c for c in pf.schema_arrow.names if c in ("text", "content")][0]
seen = 0
for b in pf.iter_batches(batch_size=2000, columns=[col]):
    add([t for t in b.column(0).to_pylist() if t], "web")
    seen += b.num_rows
    if seen >= n_docs:
        break

# code: source trees on this machine (C/C++/CUDA/Python/JS/Rust/Go/shell/markdown)
exts = (".py", ".c", ".cc", ".cpp", ".h", ".hpp", ".cu", ".cuh", ".js", ".ts", ".rs", ".go", ".sh", ".md", ".java", ".json", ".yaml", ".toml")
roots = ["/home/god/dev/llama.cpp-qsa", "/usr/lib/python3.12", "/home/god/moon", "/opt/stacks/llm-stack"]
budget = code_mb << 20
buf = []
for r in roots:
    for dp, dn, fn in os.walk(r):
        dn[:] = [d for d in dn if not d.startswith(".") and d not in ("build", "node_modules", "models", "__pycache__", "data_processing", "results")]
        for f in fn:
            if not f.endswith(exts):
                continue
            p = os.path.join(dp, f)
            try:
                if os.path.getsize(p) > 400000:
                    continue
                t = open(p, encoding="utf-8").read()
            except Exception:
                continue
            buf.append(t)
            budget -= len(t)
            if len(buf) >= 500:
                add(buf, "code"); buf = []
            if budget <= 0:
                break
        if budget <= 0:
            break
    if budget <= 0:
        break
if buf:
    add(buf, "code")

# rank by frequency; tokens never seen are appended in id order so the list is a full ranking
ranked = [i for i, _ in cnt.most_common()]
seen_ids = set(ranked)
vocab_n = tok.get_vocab_size(with_added_tokens=True)
ranked += [i for i in range(vocab_n) if i not in seen_ids]
open(out + ".ids", "w").write("\n".join(map(str, ranked)) + "\n")
tot = sum(cnt.values())
cov = {}
acc = 0
for k, (i, c) in enumerate(cnt.most_common(), 1):
    acc += c
    if k in (4096, 8192, 16384, 32768, 65536):
        cov[k] = round(acc / tot, 5)
json.dump({"tokens": ntok, "distinct": len(cnt), "vocab": vocab_n, "coverage_on_corpus": cov}, open(out + ".json", "w"), indent=1)
print(ntok, len(cnt), vocab_n, cov)
