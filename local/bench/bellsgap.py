# Decompose each BELLS round-trip (see bellsprof.sh, GTRACE=graph): from the end of a routing
# readback (small D2H) to the next GPU work, split into host wake-up after the sync, host work until
# the next launch (with the CUDA API time spent inside it), the launch call, and launch-to-start.
import json, sqlite3, sys, bisect, statistics as st, collections
O = sys.argv[1]
W = [w for w in json.load(open(O + "/windows.json")) if w["case"] != "warmup"]
db = sqlite3.connect(O + "/prof.sqlite")
t0 = int(db.execute("select utcEpochNs from TARGET_INFO_SESSION_START_TIME").fetchone()[0])
tabs = {r[0] for r in db.execute("select name from sqlite_master where type='table'")}
gpu = [(s, e) for s, e in db.execute("select start, end from CUPTI_ACTIVITY_KIND_KERNEL")]
first_by_corr = {}
for s, c in db.execute("select start, correlationId from CUPTI_ACTIVITY_KIND_KERNEL"):
    first_by_corr[c] = min(s, first_by_corr.get(c, s))
if "CUPTI_ACTIVITY_KIND_GRAPH_TRACE" in tabs:
    for s, e, c in db.execute("select start, end, correlationId from CUPTI_ACTIVITY_KIND_GRAPH_TRACE"):
        gpu.append((s, e)); first_by_corr[c] = min(s, first_by_corr.get(c, s))
mem = db.execute("select start, end, copyKind, bytes, correlationId from CUPTI_ACTIVITY_KIND_MEMCPY").fetchall()
gpu += [(s, e) for s, e, k, b, c in mem]
gpu.sort(); gstarts = [g[0] for g in gpu]
rt = db.execute("""select r.start, r.end, s.value, r.correlationId, r.globalTid from CUPTI_ACTIVITY_KIND_RUNTIME r
                   join StringIds s on s.id = r.nameId order by r.start""").fetchall()
by_corr = {r[3]: r for r in rt}
by_tid = collections.defaultdict(list)
for r in rt:
    by_tid[r[4]].append(r)
tid_starts = {t: [r[0] for r in v] for t, v in by_tid.items()}
LAUNCH = ("cudaGraphLaunch", "cudaLaunchKernel")
for w in W:
    end = w["end_ns"] - t0; beg = end - int(w["decode_ms"] * 1e6)
    P = max(1, w["n_gen"] - (w["draft_acc"] or 0))
    rows = []; nmiss_launch = [0]
    for s, e, k, b, c in mem:
        if k != 2 or b > 16384 or s < beg or e > end or c not in by_corr:
            continue
        call = by_corr[c]; tid = call[4]; L = by_tid[tid]; S = tid_starts[tid]
        i = bisect.bisect_left(S, call[0])
        # the sync that returns after the copy lands
        j = i
        while j < len(L) and not ("Synchronize" in L[j][2] and L[j][1] >= e):
            j += 1
        if j >= len(L):
            continue
        t_ret = L[j][1]
        m = j + 1; api = 0; h2d_big = False
        while m < len(L) and not L[m][2].startswith(LAUNCH):
            api += L[m][1] - L[m][0]
            m += 1
        if m >= len(L):
            continue
        ls, le = L[m][0], L[m][1]
        nxt = first_by_corr.get(L[m][3])
        if nxt is None or nxt - e > 5e6:
            nmiss_launch[0] += 1
            continue
        # any large H2D (expert copy) between readback and next launch?
        h2d_big = any(k2 == 1 and b2 >= 4096 and e <= s2 <= ls for s2, e2, k2, b2, c2 in mem)
        rows.append({"wake": t_ret - e, "host": ls - t_ret - api, "api": api, "launch": le - ls,
                     "start": max(0, nxt - le) if nxt >= le else 0, "gap": nxt - e, "miss": h2d_big,
                     "g_before_launch": nxt < ls})
    if not rows:
        print(w["case"], "no rows"); continue
    print(f"== {w['case']}  decode {w['decode_tps']:.1f} tok/s  passes {P}  wall/pass {w['decode_ms']/P:.2f} ms  "
          f"round-trips/pass {len(rows)/P:.1f}  (launch without GPU record: {nmiss_launch[0]})")
    for lab, sel in (("all", rows), ("no copy", [r for r in rows if not r["miss"]]), ("with copy", [r for r in rows if r["miss"]])):
        if not sel:
            continue
        f = lambda key: st.median(r[key] for r in sel) / 1e3
        tot = sum(r["gap"] for r in sel) / P / 1e6
        print(f"   {lab:9s} n/pass {len(sel)/P:5.1f}  gap total {tot:5.2f} ms/pass  median us: gap {f('gap'):6.1f} = "
              f"wake {f('wake'):5.1f} + host {f('host'):5.1f} + api-in-between {f('api'):5.1f} + launch {f('launch'):5.1f} + to-start {f('start'):5.1f}"
              f"  ")
    # which API calls run between the sync return and the launch
    c = collections.Counter(); ct = collections.Counter()
    for s, e, k, b, cc in mem:
        pass
    print()
