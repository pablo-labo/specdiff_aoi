#!/usr/bin/env python3
import argparse
import json
import os
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


ROOT_DIR = Path(__file__).resolve().parent.parent
DASHBOARD_DIR = ROOT_DIR / "dashboard"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def iter_timing_files(root: Path):
    for path in root.rglob("*.jsonl"):
        if "system_timing" in path.name:
            yield path
    for path in root.rglob("*.json"):
        if "system_timing" in path.name:
            yield path


def parse_timing_rows(path: Path):
    rows = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        return []
    return rows


def summarize_run(path: Path):
    rows = parse_timing_rows(path)
    stat = path.stat()
    latest = rows[-1] if rows else {}
    rounds = len(rows)
    done = int(latest.get("done", 0)) if latest else 0
    active_batch_size = int(latest.get("active_batch_size", 0)) if latest else 0
    acceptance = float(latest.get("acceptance_rate_round", 0.0)) if latest else 0.0
    tps = float(latest.get("accepted_tok_per_sec_round", 0.0)) if latest else 0.0
    wall_ms = float(latest.get("round_wall_ms", 0.0)) if latest else 0.0
    return {
        "id": str(path.relative_to(ROOT_DIR)),
        "path": str(path.relative_to(ROOT_DIR)),
        "rounds": rounds,
        "done": done,
        "active_batch_size": active_batch_size,
        "acceptance_rate_round": acceptance,
        "accepted_tok_per_sec_round": tps,
        "round_wall_ms": wall_ms,
        "updated_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "size_bytes": stat.st_size,
        "has_rows": bool(rows),
    }


def list_runs():
    roots = [ROOT_DIR / "outputs", ROOT_DIR]
    seen = set()
    runs = []
    for root in roots:
        if not root.exists():
            continue
        for path in iter_timing_files(root):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            runs.append(summarize_run(path))
    runs.sort(key=lambda item: item["updated_at"], reverse=True)
    return runs


def load_run(path_value: str):
    target = (ROOT_DIR / path_value).resolve()
    if ROOT_DIR.resolve() not in target.parents and target != ROOT_DIR.resolve():
        raise ValueError("path escapes project root")
    if not target.exists():
        raise FileNotFoundError(path_value)
    rows = parse_timing_rows(target)
    return {
        "run": summarize_run(target),
        "rows": rows,
        "server_time": utc_now_iso(),
    }


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DASHBOARD_DIR), **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/runs":
            self.send_json({
                "runs": list_runs(),
                "server_time": utc_now_iso(),
            })
            return
        if parsed.path == "/api/run":
            query = parse_qs(parsed.query)
            path_value = query.get("path", [None])[0]
            if not path_value:
                self.send_error(400, "missing path")
                return
            try:
                payload = load_run(path_value)
            except FileNotFoundError:
                self.send_error(404, "run not found")
                return
            except ValueError:
                self.send_error(400, "invalid path")
                return
            self.send_json(payload)
            return
        return super().do_GET()

    def log_message(self, fmt, *args):
        return

    def send_json(self, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser(description="Serve a local dashboard for FailFast runs.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"[dashboard] http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
