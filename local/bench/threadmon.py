# threadmon.py PID OUT : per-thread CPU share of a process every 0.5 s (name, tid, busy)
import os, sys, time
pid, out = sys.argv[1], open(sys.argv[2], "w")
hz = os.sysconf("SC_CLK_TCK"); prev = {}
while os.path.exists(f"/proc/{pid}"):
    now = time.time()
    for tid in os.listdir(f"/proc/{pid}/task"):
        try:
            comm = open(f"/proc/{pid}/task/{tid}/comm").read().strip()
            f = open(f"/proc/{pid}/task/{tid}/stat").read().rsplit(")", 1)[1].split()
        except (FileNotFoundError, ProcessLookupError):
            continue
        t = (int(f[11]) + int(f[12])) / hz
        if tid in prev:
            b = (t - prev[tid][0]) / (now - prev[tid][1])
            if b > 0.2:
                out.write(f"{now:.2f} {tid} {comm} {b:.2f}\n")
        prev[tid] = (t, now)
    out.flush(); time.sleep(0.5)
