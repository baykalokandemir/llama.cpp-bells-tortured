import collections
reqs = []
for l in open("/tmp/probe-long.out"):
    p = l.split()
    if len(p) > 5 and p[2].startswith("inst"):
        reqs.append((float(p[0]), float(p[1]), p[2] + " " + p[3], float(p[6])))
S = [l.split() for l in open("/tmp/allthreads-long.txt")]
for t0, t1, name, ms in reqs:
    c = collections.Counter()
    for ts, pid, tid, comm, cpu, b in S:
        ts = float(ts)
        if t0 + 0.5 <= ts <= t1:
            c[(comm, tid)] += float(b)
    nwin = max(1, int((t1 - t0 - 0.5) / 0.5))
    busy = sorted(((v / nwin, k) for k, v in c.items() if v / nwin > 0.15), reverse=True)
    print(f"{name} {ms:5.1f} ms/pass | " + " | ".join(f"{k[0]}[{k[1]}] {v:.2f}" for v, k in busy[:5]))
