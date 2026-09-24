# ple_trace_check.py SERVER_LOG : verify PLE n-gram predecessors against the final token sequence
import re, sys, collections
lines = [l for l in open(sys.argv[1]) if l.startswith("ple_trace ")]
pat = re.compile(r"ctx=(\S+) pos=(-?\d+) tok=(-?\d+) prev=([-\d,]*)")
by_ctx = collections.defaultdict(list)
for l in lines:
    m = pat.search(l)
    if m:
        by_ctx[m.group(1)].append((int(m.group(2)), int(m.group(3)), [int(x) for x in m.group(4).split(",") if x]))
for ctx, rows in by_ctx.items():
    # split into requests: a new request restarts at a lower position
    reqs, cur, last = [], [], -1
    for r in rows:
        if cur and r[0] < last - 64:
            reqs.append(cur); cur = []
        cur.append(r); last = r[0]
    reqs.append(cur)
    n_lines = n_checked = n_bad = n_after_reject = 0
    examples = []
    for req in reqs:
        final = {}
        for pos, tok, _ in req:
            final[pos] = tok                       # last write wins
        first = min(final)
        seen_pos = set()
        last_idx = {}
        for idx, (pos, tok, prev) in enumerate(req):
            last_idx[pos] = idx                    # the final write of each position
        for idx, (pos, tok, prev) in enumerate(req):
            n_lines += 1
            if last_idx[pos] != idx:
                seen_pos.add(pos)
                continue                           # superseded: a rejected draft, or a draft chain that was redone
            n = len(prev)
            want = [final.get(pos - (n - j)) for j in range(n)]
            ok = all(w is None or w == p for w, p in zip(want, prev))
            covered = any(w is not None for w in want)
            if not covered:
                continue
            n_checked += 1
            rewrite = pos in seen_pos              # this position was written before: post-rejection rewrite
            n_after_reject += rewrite
            if not ok:
                n_bad += 1
                if len(examples) < 5:
                    examples.append((pos, tok, prev, want, rewrite))
            seen_pos.add(pos)
    print(f"ctx {ctx}: requests {len(reqs)}, trace lines {n_lines}, accepted tokens checked {n_checked} "
          f"(of which post-rejection rewrites {n_after_reject}), MISMATCHES {n_bad}")
    for e in examples:
        print("   mismatch pos=%d tok=%d prev=%s expected=%s rewrite=%s" % e)
