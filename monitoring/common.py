#!/usr/bin/env python3
"""Shared helpers for workload quantification scripts."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import time
import fcntl
from pathlib import Path
from typing import Any, Iterable


def now_ns() -> int:
    return time.time_ns()


def mono_ns() -> int:
    return time.monotonic_ns()


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def append_jsonl(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = (json.dumps(data, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        view = memoryview(line)
        while view:
            written = os.write(fd, view)
            if written == 0:
                raise OSError(f"zero-byte write while appending {path}")
            view = view[written:]
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(fd)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def git_commit(repo: str | Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return None


def create_manifest(
    out_dir: str | Path,
    *,
    run_id: str,
    mode: str,
    dataset_path: str | Path,
    model: str,
    vllm_url: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": run_id,
        "mode": mode,
        "dataset_path": str(dataset_path),
        "model": model,
        "vllm_url": vllm_url.rstrip("/"),
        "run_start_wall_ns": now_ns(),
        "run_start_mono_ns": mono_ns(),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "git_commit": git_commit(Path(__file__).resolve().parents[1]),
    }
    if extra:
        manifest.update(extra)
    write_json(out_dir / "run_manifest.json", manifest)
    return manifest


def load_manifest(path_or_dir: str | Path) -> dict[str, Any]:
    path = Path(path_or_dir)
    if path.is_dir():
        path = path / "run_manifest.json"
    return load_json(path)


def event_base(manifest: dict[str, Any], *, source: str, event_type: str) -> dict[str, Any]:
    t_wall = now_ns()
    t_mono = mono_ns()
    return {
        "run_id": manifest["run_id"],
        "source": source,
        "event_type": event_type,
        "ts_ns": t_wall,
        "t_rel_ms": (t_mono - int(manifest["run_start_mono_ns"])) / 1e6,
    }


def load_dataset_jsonl(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if limit is not None:
        rows = rows[:limit]
    return rows


def quantiles(values: Iterable[float], qs: Iterable[float]) -> dict[str, float | None]:
    data = sorted(float(v) for v in values)
    if not data:
        return {str(q): None for q in qs}
    out: dict[str, float | None] = {}
    n = len(data)
    for q in qs:
        if n == 1:
            out[str(q)] = data[0]
            continue
        pos = (n - 1) * q
        lo = int(pos)
        hi = min(lo + 1, n - 1)
        frac = pos - lo
        out[str(q)] = data[lo] * (1.0 - frac) + data[hi] * frac
    return out


def seconds_to_sleep_until_next_sample(last_mono_ns: int, interval_s: float) -> float:
    target = last_mono_ns + int(interval_s * 1e9)
    return max(0.0, (target - mono_ns()) / 1e9)
