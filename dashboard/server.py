"""Local experiment tracker for kernelforge. Stdlib only.

    python3 dashboard/server.py [--port 8790]     ->  http://localhost:8790

Reads runs/*/result.json, results.jsonl, observations.jsonl and the latest job log on m5-test
(read-only `ssh m5-test tail`). "Pull results" runs scripts/pull.sh (rsync m5-test -> local, additive).
"""

import argparse
import json
import mimetypes
import subprocess
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
OBS = ROOT / "observations.jsonl"
_job_cache = {"t": 0, "data": None}


def read_jsonl(p):
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def runs():
    out = []
    for rp in sorted((ROOT / "runs").glob("exp-*/result.json")):
        try:
            r = json.loads(rp.read_text())
        except json.JSONDecodeError:
            continue
        d = rp.parent
        r.pop("idle_check", None)
        r["_images"] = sorted(p.name for p in d.glob("*.png"))
        r["_has_memtrace"] = (d / "memtrace.json").exists()
        r["_per_run"] = read_jsonl(d / "per_run.jsonl")
        out.append(r)
    return out


def memtrace(exp, max_points=1500):
    p = ROOT / "runs" / exp / "memtrace.json"
    if not p.exists():
        return None
    m = json.loads(p.read_text())
    s = m["samples"]
    step = max(1, len(s) // max_points)
    # keep the max footprint within each bucket so peaks survive downsampling
    ds = [max(s[i:i + step], key=lambda r: r[1]) for i in range(0, len(s), step)]
    return {"cols": m["cols"], "samples": ds, "marks": m["marks"]}


def job_status():
    if time.time() - _job_cache["t"] < 25 and _job_cache["data"]:
        return _job_cache["data"]
    cmd = ("cd ~/kernelforge/runs/jobs 2>/dev/null && f=$(ls -t *.log 2>/dev/null | head -1) && "
           "echo \"$f\" && tail -25 \"$f\"; echo '---PS'; pgrep -fl 'bench/run.py' | grep -v pgrep | head -3; "
           "echo '---TOP'; top -l 1 -n 0 | grep -E 'CPU usage|PhysMem'; sysctl -n vm.swapusage")
    try:
        out = subprocess.run(["ssh", "-o", "ConnectTimeout=5", "m5-test", cmd], capture_output=True, text=True, timeout=20).stdout
        head, _, rest = out.partition("---PS")
        ps, _, top = rest.partition("---TOP")
        lines = head.strip().splitlines()
        data = {"ok": True, "job": lines[0] if lines else None, "log": lines[1:], "running": [l for l in ps.strip().splitlines() if l],
                "host": top.strip().splitlines(), "checked": datetime.now().strftime("%H:%M:%S")}
    except Exception as e:  # noqa: BLE001
        data = {"ok": False, "error": str(e), "checked": datetime.now().strftime("%H:%M:%S")}
    _job_cache.update(t=time.time(), data=data)
    return data


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body, ctype="application/json"):
        b = body if isinstance(body, bytes) else json.dumps(body, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        if path in ("/", "/index.html"):
            return self.send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
        if path == "/api/state":
            notes = ROOT / "NOTES.md"
            return self.send(200, {"runs": runs(), "results": read_jsonl(ROOT / "results.jsonl"),
                                   "observations": read_jsonl(OBS), "notes": notes.read_text() if notes.exists() else ""})
        if path.startswith("/api/memtrace/"):
            return self.send(200, memtrace(path.rsplit("/", 1)[1]))
        if path == "/api/job":
            return self.send(200, job_status())
        if path.startswith("/files/"):
            f = (ROOT / path[len("/files/"):]).resolve()
            if ROOT not in f.parents or not f.is_file():
                return self.send(404, {"error": "not found"})
            return self.send(200, f.read_bytes(), mimetypes.guess_type(f.name)[0] or "application/octet-stream")
        self.send(404, {"error": "not found"})

    def do_POST(self):
        path = urlparse(self.path).path
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        if path == "/api/observations":
            text = (body.get("text") or "").strip()
            if not text:
                return self.send(400, {"error": "Write something before saving."})
            rec = {"ts": datetime.now().isoformat(timespec="seconds"), "exp_id": body.get("exp_id") or None,
                   "kind": body.get("kind") or "observation", "text": text, "author": body.get("author") or "user"}
            with open(OBS, "a") as f:
                f.write(json.dumps(rec) + "\n")
            return self.send(200, rec)
        if path == "/api/pull":
            p = subprocess.run(["sh", str(ROOT / "scripts" / "pull.sh")], capture_output=True, text=True, timeout=300)
            _job_cache["t"] = 0
            return self.send(200 if p.returncode == 0 else 500, {"rc": p.returncode, "stderr": p.stderr[-2000:]})
        self.send(404, {"error": "not found"})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8790)
    a = ap.parse_args()
    print(f"kernelforge dashboard on http://localhost:{a.port}")
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()
