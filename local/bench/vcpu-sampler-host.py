# sample busy KVM vCPU threads of VM 100: host CPU they ran on, every 0.5 s
import os, time, sys
pid = open("/var/run/qemu-server/100.pid").read().strip()
hz = os.sysconf("SC_CLK_TCK")
out = open(sys.argv[1], "w")
prev = {}
while True:
    now = time.time()
    for tid in os.listdir(f"/proc/{pid}/task"):
        try:
            comm = open(f"/proc/{pid}/task/{tid}/comm").read().strip()
            if "KVM" not in comm:
                continue
            f = open(f"/proc/{pid}/task/{tid}/stat").read().rsplit(")", 1)[1].split()
        except FileNotFoundError:
            continue
        cpu_t = (int(f[11]) + int(f[12])) / hz
        cpu = int(f[36])
        if tid in prev:
            busy = (cpu_t - prev[tid][0]) / (now - prev[tid][1])
            if busy > 0.3:
                out.write(f"{now:.2f} {comm.split('/')[0]} {cpu} {busy:.2f}\n")
        prev[tid] = (cpu_t, now)
    out.flush()
    time.sleep(0.5)
