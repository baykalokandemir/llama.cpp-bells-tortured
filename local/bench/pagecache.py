# pagecache.py resident|evict FILE : report page-cache residency (mincore) or evict (fadvise DONTNEED)
import ctypes, mmap, os, sys
mode, path = sys.argv[1], sys.argv[2]
fd = os.open(path, os.O_RDONLY)
size = os.fstat(fd).st_size
if mode == "evict":
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
libc = ctypes.CDLL("libc.so.6", use_errno=True)
libc.mmap.restype = ctypes.c_void_p
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
addr = libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, fd, 0)
ps = mmap.PAGESIZE
n = (size + ps - 1) // ps
vec = (ctypes.c_ubyte * n)()
if libc.mincore(ctypes.c_void_p(addr), ctypes.c_size_t(size), vec) != 0:
    raise OSError(ctypes.get_errno(), "mincore")
res = sum(1 for b in vec if b & 1)
print(f"{os.path.basename(path)}: {res*ps/2**30:.2f} GiB of {size/2**30:.2f} GiB resident ({100*res/n:.1f}%)")
