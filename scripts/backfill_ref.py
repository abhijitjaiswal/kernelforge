"""One-off data fix: exp-002..004 ran before result.json recorded `ref`; their --ref came from scripts/phase0.sh."""
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
for exp in ("exp-002", "exp-003", "exp-004"):
    p = root / "runs" / exp / "result.json"
    r = json.loads(p.read_text())
    if r.get("ref") is None:
        r["ref"] = "baselines/q8_free_te"
        r["ref_backfilled_from"] = "scripts/phase0.sh"
        p.write_text(json.dumps(r, indent=2))
        print("backfilled", exp)
