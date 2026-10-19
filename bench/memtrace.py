"""Process memory timeline sampler (macOS). Model-agnostic.

Samples the kernel's phys_footprint (what /usr/bin/time -l reports as "peak memory footprint",
includes GPU-visible unified memory) plus MLX active/cache memory every `interval` seconds on a
background thread. Stage boundaries are recorded with mark(name).
"""

import ctypes
import os
import threading
import time

_libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
RUSAGE_INFO_V2 = 2


class _RusageV2(ctypes.Structure):
    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [(n, ctypes.c_uint64) for n in (
        "ri_user_time", "ri_system_time", "ri_pkg_idle_wkups", "ri_interrupt_wkups", "ri_pageins",
        "ri_wired_size", "ri_resident_size", "ri_phys_footprint", "ri_proc_start_abstime",
        "ri_proc_exit_abstime", "ri_child_user_time", "ri_child_system_time", "ri_child_pkg_idle_wkups",
        "ri_child_interrupt_wkups", "ri_child_pageins", "ri_child_elapsed_abstime", "ri_diskio_bytesread",
        "ri_diskio_byteswritten")]


def phys_footprint() -> tuple[int, int]:
    ri = _RusageV2()
    _libc.proc_pid_rusage(os.getpid(), RUSAGE_INFO_V2, ctypes.byref(ri))
    return ri.ri_phys_footprint, ri.ri_resident_size


class MemTrace:
    def __init__(self, interval=0.05):
        import mlx.core as mx
        self.mx, self.interval = mx, interval
        self.samples, self.marks = [], []
        self._stop = threading.Event()
        self.t0 = time.perf_counter()
        self._th = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            fp, rss = phys_footprint()
            self.samples.append((time.perf_counter() - self.t0, fp, rss, self.mx.get_active_memory(), self.mx.get_cache_memory()))
            time.sleep(self.interval)

    def start(self):
        self._th.start()
        return self

    def mark(self, name):
        self.marks.append((time.perf_counter() - self.t0, name))

    def stop(self):
        self._stop.set()
        self._th.join()

    def summary(self):
        g = 1024**3
        peak = max(self.samples, key=lambda s: s[1])
        # attribute peak footprint to the stage window it fell in
        stage = "?"
        for t, name in self.marks:
            if t <= peak[0]:
                stage = name
        per_stage = {}
        bounds = self.marks + [(float("inf"), "_end")]
        for (t0, name), (t1, _) in zip(bounds, bounds[1:]):
            win = [s for s in self.samples if t0 <= s[0] < t1]
            if win:
                per_stage.setdefault(name, {"footprint_peak_gb": 0, "mlx_active_peak_gb": 0})
                d = per_stage[name]
                d["footprint_peak_gb"] = round(max(d["footprint_peak_gb"], max(s[1] for s in win) / g), 3)
                d["mlx_active_peak_gb"] = round(max(d["mlx_active_peak_gb"], max(s[3] for s in win) / g), 3)
        return {"peak_footprint_gb": round(peak[1] / g, 3), "peak_at_s": round(peak[0], 2), "peak_stage": stage,
                "per_stage": per_stage}

    def dump(self, path):
        import json
        with open(path, "w") as f:
            json.dump({"cols": ["t_s", "phys_footprint", "rss", "mlx_active", "mlx_cache"], "samples": self.samples,
                       "marks": self.marks}, f)
