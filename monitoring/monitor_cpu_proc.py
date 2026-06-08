#!/usr/bin/env python3
"""Sample CPU, memory, and I/O stats for matching local processes."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from common import append_jsonl, event_base, load_manifest


SELF_MONITOR_MARKERS = (
    "workload_quantification/monitor_",
    "monitor_cpu_proc.py",
    "monitor_gpu_nvml.py",
    "monitor_pcie_nvml.py",
    "monitor_vllm_metrics.py",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--interval", type=float, default=1.0)
    p.add_argument(
        "--match",
        action="append",
        default=[],
        help="Substring to match in process cmdline/name. Repeatable. Default: vllm, minisweagent",
    )
    return p.parse_args()


def matches(proc_info: dict, patterns: list[str]) -> bool:
    text = " ".join([proc_info.get("name") or "", " ".join(proc_info.get("cmdline") or [])])
    return any(p in text for p in patterns)


def is_monitor_process(proc_info: dict) -> bool:
    text = " ".join([proc_info.get("name") or "", " ".join(proc_info.get("cmdline") or [])])
    return any(marker in text for marker in SELF_MONITOR_MARKERS)


def main() -> None:
    args = parse_args()
    try:
        import psutil
    except ImportError as exc:
        raise SystemExit("Install psutil to run CPU/process monitoring") from exc

    manifest = load_manifest(args.manifest)
    out = Path(args.out)
    patterns = args.match or ["vllm", "minisweagent"]
    print(f"Sampling processes matching {patterns} every {args.interval}s -> {out}")

    seen: dict[int, psutil.Process] = {}
    try:
        while True:
            current_pids = set()
            for proc in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
                try:
                    info = proc.info
                    if proc.pid == os.getpid() or is_monitor_process(info):
                        continue
                    if not matches(info, patterns):
                        continue
                    current_pids.add(proc.pid)
                    if proc.pid not in seen:
                        seen[proc.pid] = proc
                        proc.cpu_percent(interval=None)
                    mem = proc.memory_info()
                    io = None
                    try:
                        io = proc.io_counters()
                    except Exception:
                        pass
                    ctx = None
                    try:
                        ctx = proc.num_ctx_switches()
                    except Exception:
                        pass
                    event = event_base(manifest, source="cpu_proc", event_type="process_sample")
                    event.update({
                        "pid": proc.pid,
                        "name": info.get("name"),
                        "cmdline": info.get("cmdline"),
                        "create_time": info.get("create_time"),
                        "cpu_pct": proc.cpu_percent(interval=None),
                        "rss_mb": mem.rss / 1024 / 1024,
                        "vms_mb": mem.vms / 1024 / 1024,
                        "num_threads": proc.num_threads(),
                        "read_mb": io.read_bytes / 1024 / 1024 if io else None,
                        "write_mb": io.write_bytes / 1024 / 1024 if io else None,
                        "voluntary_ctx_switches": ctx.voluntary if ctx else None,
                        "involuntary_ctx_switches": ctx.involuntary if ctx else None,
                    })
                    append_jsonl(out, event)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            for pid in list(seen):
                if pid not in current_pids:
                    seen.pop(pid, None)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("Stopped CPU/process monitor")


if __name__ == "__main__":
    main()
