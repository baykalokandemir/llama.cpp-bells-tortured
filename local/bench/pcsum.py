import json
D = "/home/god/dev/iq3-explore/dc-pc-%s.json"
arms = ["off1", "off2", "on1", "on2"]
rows = {a: json.load(open(D % a)) for a in arms}
keys = [(r["case"], i) for i, r in enumerate(rows["off1"])]
print("%-12s %s" % ("case", "  ".join("%-22s" % a for a in arms)))
for i, (c, _) in enumerate(keys):
    cells = []
    for a in arms:
        r = rows[a][i]
        cells.append("%6.2f %5.1f %s" % (r["decode_tps"], r["prompt_tps"], r["sha"][:6]))
    print("%-12s %s" % (c, "  ".join("%-22s" % x for x in cells)))
def med(a, c):
    v = sorted(r["decode_tps"] for r in rows[a] if r["case"] == c)
    return sum(v) / len(v)
print()
for c in ["shallow", "depth8192", "depth16384", "depth32768", "depth61440"]:
    off = (med("off1", c) + med("off2", c)) / 2
    on = (med("on1", c) + med("on2", c)) / 2
    print("%-12s off %.2f on %.2f  %+.1f%%" % (c, off, on, 100 * (on / off - 1)))
