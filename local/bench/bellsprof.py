# GPU idle-gap attribution per MTP pass (see bellsprof.sh). Both GPUs run in sequence under -sm layer,
# so the union of their activity is the critical path; every hole in it is time no GPU works.
import json, sqlite3, sys, collections
O = sys.argv[1]
W = json.load(open(O + "/windows.json"))
db = sqlite3.connect(O + "/prof.sqlite")
t0 = int(db.execute("select utcEpochNs from TARGET_INFO_SESSION_START_TIME").fetchone()[0])
tabs = {r[0] for r in db.execute("select name from sqlite_master where type='table'")}
act = []  # (start, end, dev, kind, bytes)
for s, e, d in db.execute("select start, end, deviceId from CUPTI_ACTIVITY_KIND_KERNEL"):
    act.append((s, e, d, "K", 0))
if "CUPTI_ACTIVITY_KIND_MEMCPY" in tabs:
    for s, e, d, k, b in db.execute("select start, end, deviceId, copyKind, bytes from CUPTI_ACTIVITY_KIND_MEMCPY"):
        act.append((s, e, d, {1: "H2D", 2: "D2H", 8: "D2D", 10: "P2P"}.get(k, f"C{k}"), b))
act.sort()
rt = []
if "CUPTI_ACTIVITY_KIND_RUNTIME" in tabs:
    rt = db.execute("""select r.start, r.end, s.value from CUPTI_ACTIVITY_KIND_RUNTIME r
                       join StringIds s on s.id = r.nameId""").fetchall()
def lab(a):
    k, b = a[3], a[4]
    if k in ("H2D", "D2H"):
        return f"{k}{'<4K' if b < 4096 else '<1M' if b < (1 << 20) else '>=1M'}"
    return k
for w in W:
    if w["case"] == "warmup":
        continue
    end = w["end_ns"] - t0; beg = end - int(w["decode_ms"] * 1e6)
    passes = max(1, w["n_gen"] - (w["draft_acc"] or 0))
    A = [a for a in act if a[0] >= beg and a[1] <= end]
    # union + gaps, remembering the activity that ended last before each gap and the one after it
    busy = 0; gaps = collections.Counter(); gapn = collections.Counter(); hist = collections.Counter()
    cur_s, cur_e, last = None, None, None
    for a in A:
        if cur_e is None:
            cur_s, cur_e, last = a[0], a[1], a
            continue
        if a[0] > cur_e:
            busy += cur_e - cur_s
            g = a[0] - cur_e
            key = (lab(last), lab(a), "same-dev" if last[2] == a[2] else "dev-switch")
            gaps[key] += g; gapn[key] += 1
            hist["<5us" if g < 5000 else "5-20us" if g < 20000 else "20-100us" if g < 100000 else ">=100us"] += g
            cur_s, cur_e, last = a[0], a[1], a
        else:
            if a[1] > cur_e:
                cur_e, last = a[1], a
    if cur_e is not None:
        busy += cur_e - cur_s
    wall = end - beg
    kt = sum(a[1] - a[0] for a in A if a[3] == "K")
    mc = collections.Counter(); mb = collections.Counter()
    for a in A:
        if a[3] != "K":
            mc[lab(a)] += 1; mb[lab(a)] += a[1] - a[0]
    P = passes
    print(f"== {w['case']}  decode {w['decode_tps']:.1f} tok/s  passes {P}  wall/pass {wall/P/1e6:.2f} ms  "
          f"GPU-busy(union)/pass {busy/P/1e6:.2f} ms  idle/pass {(wall-busy)/P/1e6:.2f} ms  kernel-sum/pass {kt/P/1e6:.2f} ms")
    print("   copies per pass: " + "  ".join(f"{k} n={mc[k]/P:.1f} t={mb[k]/P/1e3:.0f}us" for k in sorted(mc)))
    print("   idle by gap size: " + "  ".join(f"{k} {v/P/1e6:.2f}ms" for k, v in sorted(hist.items())))
    print("   idle by (before -> after):")
    for k, v in gaps.most_common(12):
        print(f"     {v/P/1e6:6.2f} ms/pass  n={gapn[k]/P:6.1f}/pass  avg {v/gapn[k]/1e3:6.1f} us  {k[0]:>8s} -> {k[1]:<8s} {k[2]}")
    R = collections.Counter(); Rn = collections.Counter()
    for s, e, n in rt:
        if s >= beg and e <= end:
            R[n] += e - s; Rn[n] += 1
    print("   host CUDA API time per pass:")
    for n, v in R.most_common(8):
        print(f"     {v/P/1e6:6.2f} ms  n={Rn[n]/P:6.1f}  {n[:60]}")
