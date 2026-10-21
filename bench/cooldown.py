"""Cool-down gate: sleep at least --min seconds, then wait until thermal pressure is Nominal (max --max s)."""
import argparse, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench.power import thermal_now

ap = argparse.ArgumentParser()
ap.add_argument("--min", type=int, default=150)
ap.add_argument("--max", type=int, default=900)
a = ap.parse_args()
FLOOR = 120  # thermal-normalisation floor between runs (user, 2026-10-21 15:12; was 300 from ~14:16): >= 2 min, then wait for Nominal
t0 = time.time()
time.sleep(max(a.min, FLOOR))
lvl = thermal_now()
while lvl not in (None, "Nominal") and time.time() - t0 < a.max:
    time.sleep(15)
    lvl = thermal_now()
print(f"[cooldown] {time.time() - t0:.0f}s, thermal={lvl}", flush=True)
