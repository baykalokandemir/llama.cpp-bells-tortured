import glob, re, sys
sys.path.insert(0, "/home/god/dev/llama.cpp-qsa/gguf-py")
from gguf import GGUFReader
per = {}; other = {}
for f in sorted(glob.glob("/opt/stacks/llm-stack/models/navin-ad427/*-main-*.gguf")):
    r = GGUFReader(f)
    for t in r.tensors:
        n = t.name; b = int(t.n_bytes)
        m = re.match(r"blk\.(\d+)\.", n)
        if m:
            il = int(m.group(1))
            cpu = 14 <= il <= 47 and re.search(r"ffn_(up|down|gate|gate_up)_(ch_|)exps", n)
            per.setdefault(il, [0, 0])[1 if cpu else 0] += b
        else:
            other[n] = b
G = 2**20
for il in sorted(per): print(il, round(per[il][0]/G), round(per[il][1]/G))
for n, b in other.items(): print(n, round(b/G))
print("gpu layer total", round(sum(v[0] for v in per.values())/G))
