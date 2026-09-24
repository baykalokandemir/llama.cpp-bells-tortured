# all-threads sampler: every 0.5 s, any task (user or kernel) above 20% CPU, with its comm and cpu
import os, sys, time
out = open(sys.argv[1], "w"); dur = float(sys.argv[2])
hz = os.sysconf("SC_CLK_TCK"); prev = {}; t_end = time.time() + dur
while time.time() < t_end:
    now = time.time()
    for pid in os.listdir("/proc"):
        if not pid.isdigit(): continue
        try: tids = os.listdir(f"/proc/{pid}/task")
        except Exception: continue
        for tid in tids:
            try:
                comm = open(f"/proc/{pid}/task/{tid}/comm").read().strip()
                f = open(f"/proc/{pid}/task/{tid}/stat").read().rsplit(")", 1)[1].split()
            except Exception: continue
            t = (int(f[11]) + int(f[12])) / hz
            k = (pid, tid)
            if k in prev:
                b = (t - prev[k][0]) / (now - prev[k][1])
                if b > 0.2: out.write(f"{now:.2f} {pid} {tid} {comm.replace(chr(32),chr(95))} cpu{f[36]} {b:.2f}\n")
            prev[k] = (t, now)
    out.flush(); time.sleep(0.5)
