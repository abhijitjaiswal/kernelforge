"""powermetrics recorder (macOS, passwordless sudo for /usr/bin/powermetrics). Model-agnostic.

macOS 26 on Apple silicon has no `smc` sampler, so temperature is not available; GPU clock, GPU power
and the thermal pressure level are recorded instead (a falling GPU clock under load = throttling).
"""

import re
import subprocess
import time

SAMPLERS = "gpu_power,thermal,cpu_power"


class Power:
    def __init__(self, out_path, interval_ms=1000):
        self.out_path, self.interval_ms, self.p, self.error = str(out_path), interval_ms, None, None

    def start(self):
        try:
            self.p = subprocess.Popen(["sudo", "-n", "/usr/bin/powermetrics", "--samplers", SAMPLERS,
                                       "-i", str(self.interval_ms), "-o", self.out_path],
                                      stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            time.sleep(0.3)
            if self.p.poll() is not None:
                self.error = self.p.stderr.read().decode()[-300:]
        except OSError as e:
            self.error = str(e)
        return self

    def stop(self):
        if self.p and self.p.poll() is None:
            self.p.terminate()
            try:
                self.p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.p.kill()

    def summary(self, t0_wall=None):
        if self.error:
            return {"error": self.error}
        try:
            text = open(self.out_path).read()
        except OSError as e:
            return {"error": str(e)}
        samples = []
        for block in text.split("*** Sampled system activity")[1:]:
            f = re.search(r"GPU HW active frequency: (\d+) MHz", block)
            r = re.search(r"GPU HW active residency:\s+([\d.]+)%", block)
            w = re.search(r"GPU Power: (\d+) mW", block)
            c = re.search(r"CPU Power: (\d+) mW", block)
            th = re.search(r"Current pressure level: (\w+)", block)
            samples.append({"gpu_mhz": int(f.group(1)) if f else None, "gpu_active_pct": float(r.group(1)) if r else None,
                            "gpu_mw": int(w.group(1)) if w else None, "cpu_mw": int(c.group(1)) if c else None,
                            "thermal": th.group(1) if th else None})
        busy = [s for s in samples if (s["gpu_active_pct"] or 0) > 50]
        med = lambda xs: sorted(xs)[len(xs) // 2] if xs else None
        return {
            "n_samples": len(samples),
            "busy_samples": len(busy),
            "gpu_mhz_busy_median": med([s["gpu_mhz"] for s in busy if s["gpu_mhz"]]),
            "gpu_mhz_busy_min": min((s["gpu_mhz"] for s in busy if s["gpu_mhz"]), default=None),
            "gpu_mw_busy_median": med([s["gpu_mw"] for s in busy if s["gpu_mw"] is not None]),
            "thermal_levels": sorted({s["thermal"] for s in samples if s["thermal"]}),
            "series": samples,
        }


def thermal_now():
    """Current thermal pressure level (one quick sample), or None."""
    try:
        out = subprocess.run(["sudo", "-n", "/usr/bin/powermetrics", "--samplers", "thermal", "-i", "200", "-n", "1"],
                             capture_output=True, text=True, timeout=10).stdout
        m = re.search(r"Current pressure level: (\w+)", out)
        return m.group(1) if m else None
    except Exception:  # noqa: BLE001
        return None
