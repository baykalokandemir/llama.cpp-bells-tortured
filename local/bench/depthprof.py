# kernel time per MTP pass inside each request's decode window (see depthprof.sh)
import json, sqlite3, sys, collections
O = sys.argv[1]
W = json.load(open(O + "/windows.json"))
db = sqlite3.connect(O + "/prof.sqlite")
t0 = int(db.execute("select utcEpochNs from TARGET_INFO_SESSION_START_TIME").fetchone()[0])
K = db.execute("""select k.start, k.end, k.deviceId, s.value from CUPTI_ACTIVITY_KIND_KERNEL k
                  join StringIds s on s.id = k.shortName""").fetchall()
def bucket(n):
    n2 = n.lower()
    for key, lab in (("flash_attn", "attention (FA)"), ("argsort", "top-k sort"), ("top_k", "top-k sort"),
                     ("get_rows", "get_rows"), ("mul_mat_vec_q_moe", "MoE experts"), ("mul_mat_id", "MoE experts"),
                     ("mul_mat_vec_q", "dense matvec (q)"), ("mul_mat_vec_f", "dense matvec (f)"), ("mul_mat_f", "dense mm (f)"),
                     ("gemm", "cuBLAS"), ("kernel2", "cuBLAS"), ("splitk", "cuBLAS"), ("gated_delta", "DeltaNet"),
                     ("ssm_conv", "DeltaNet"), ("concat", "concat"), ("cpy", "copy/convert"), ("convert", "copy/convert"),
                     ("set_rows", "set_rows"), ("add", "elementwise"), ("mul", "elementwise"), ("scale", "elementwise"),
                     ("norm", "norms"), ("rope", "rope"), ("soft_max", "softmax"), ("sum", "reductions"), ("fill", "fill")):
        if key in n2:
            return lab
    return "other"
for w in W:
    if w["case"] == "warmup":
        continue
    end = w["end_ns"] - t0; beg = end - int(w["decode_ms"] * 1e6)
    passes = max(1, w["n_gen"] - (w["draft_acc"] or 0))
    by = collections.Counter(); byk = collections.Counter(); dev = collections.Counter()
    for s, e, d, n in K:
        if s >= beg and e <= end:
            by[bucket(n)] += e - s; byk[n] += e - s; dev[d] += e - s
    tot = sum(by.values())
    print(f"== {w['case']} prompt {w['prompt_n']}  decode {w['decode_tps']:.1f} tok/s  passes {passes}  wall/pass {w['decode_ms']/passes:.2f} ms  kernel/pass {tot/passes/1e6:.2f} ms  per-dev " + " ".join(f"{d}:{v/passes/1e6:.2f}" for d, v in sorted(dev.items())))
    for lab, v in by.most_common(14):
        print(f"   {lab:22s} {v/passes/1e6:7.3f} ms/pass")
    print("   top kernels:")
    for n, v in byk.most_common(10):
        print(f"     {v/passes/1e6:7.3f}  {n[:90]}")
