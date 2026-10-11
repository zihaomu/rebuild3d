"""macOS physical footprint, kept separate from RSS and GPU allocation."""
import ctypes
import sys

# sys/resource.h rusage_info_v2; libproc accounts physical footprint separately
# from resident pages and GPU driver allocation. Never add these metrics together.
class RUsageV2(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [
        (name, ctypes.c_uint64) for name in (
            "user_time", "system_time", "pkg_idle_wkups", "interrupt_wkups", "pageins",
            "wired_size", "resident_size", "phys_footprint", "proc_start_abstime",
            "proc_exit_abstime", "child_user_time", "child_system_time", "child_pkg_idle_wkups",
            "child_interrupt_wkups", "child_pageins", "child_elapsed_abstime",
            "diskio_bytesread", "diskio_byteswritten")]


_libproc = ctypes.CDLL("/usr/lib/libproc.dylib") if sys.platform == "darwin" else None
if _libproc:
    _libproc.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    _libproc.proc_pid_rusage.restype = ctypes.c_int


def footprint(pid):
    if not _libproc:
        return None
    info = RUsageV2()
    if _libproc.proc_pid_rusage(pid, 2, ctypes.byref(info)) != 0:
        return None
    return int(info.phys_footprint)


