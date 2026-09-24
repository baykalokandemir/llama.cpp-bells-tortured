import json, glob, statistics as st
print(f"{'arm':14s} {'prose':>7s} {'code':>7s} {'d8k':>7s} {'d32k':>7s}  acc prose/code/8k/32k   tok/pass prose/code/8k/32k   spread prose|code")
for f in sorted(glob.glob("pmin-n*.json")):
    rows = [r for r in json.load(open(f)) if r["case"] != "warmup"]
    g = {}
    for r in rows:
        g.setdefault(r["case"], []).append(r)
    med = lambda c: st.median(r["decode_tps"] for r in g[c])
    acc = lambda c: sum(r["draft_acc"] for r in g[c]) / max(1, sum(r["draft_n"] for r in g[c]))
    tpp = lambda c: sum(r["n_gen"] for r in g[c]) / max(1, sum(r["n_gen"] - r["draft_acc"] for r in g[c]))
    sp = lambda c: f"{min(r['decode_tps'] for r in g[c]):.1f}-{max(r['decode_tps'] for r in g[c]):.1f}"
    cs = ["prose", "code", "depth8k", "depth32k"]
    print(f"{f[:-5]:14s} " + " ".join(f"{med(c):7.2f}" for c in cs) + "  " + "/".join(f"{acc(c):.2f}" for c in cs) + "   " + "/".join(f"{tpp(c):.2f}" for c in cs) + f"   {sp('prose')}|{sp('code')}")
