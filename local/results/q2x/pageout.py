#!/usr/bin/env python3
# Page out one process's large private anonymous mappings to swap: process_madvise(MADV_PAGEOUT).
# Meant for the PLE offload worker, so only its table goes to swap and the engine processes stay resident.
# Locked/pinned pages are skipped by the kernel. Run as root.  usage: pageout.py PID [MIN_MB]
import ctypes, os, sys

SYS_pidfd_open, SYS_process_madvise, MADV_PAGEOUT = 434, 440, 21
pid = int(sys.argv[1]); min_bytes = int(sys.argv[2] if len(sys.argv) > 2 else 64) << 20
libc = ctypes.CDLL(None, use_errno=True)
libc.syscall.restype = ctypes.c_long


class iovec(ctypes.Structure):
    _fields_ = [("base", ctypes.c_void_p), ("len", ctypes.c_size_t)]


def status(p):
    d = {}
    for line in open(f"/proc/{p}/status"):
        k, _, v = line.partition(":")
        if k in ("VmRSS", "VmSwap", "RssAnon"):
            d[k] = v.strip()
    return d


regions = []
for line in open(f"/proc/{pid}/maps"):
    f = line.split()
    lo, hi = (int(x, 16) for x in f[0].split("-"))
    path = f[5] if len(f) > 5 else ""
    if f[1][3] == "p" and (path == "" or path == "[heap]") and hi - lo >= min_bytes:
        regions.append((lo, hi - lo))
print("before", status(pid), "regions", len(regions), "GiB", round(sum(r[1] for r in regions) / 2**30, 1))

pidfd = libc.syscall(SYS_pidfd_open, pid, 0)
if pidfd < 0:
    sys.exit(f"pidfd_open failed: errno {ctypes.get_errno()}")
done = 0
for i in range(0, len(regions), 512):
    chunk = regions[i:i + 512]
    arr = (iovec * len(chunk))(*[iovec(b, n) for b, n in chunk])
    r = libc.syscall(SYS_process_madvise, pidfd, arr, len(chunk), MADV_PAGEOUT, 0)
    if r < 0:
        print("process_madvise errno", ctypes.get_errno())
    else:
        done += r
os.close(pidfd)
print("advised GiB", round(done / 2**30, 1))
print("after", status(pid))
