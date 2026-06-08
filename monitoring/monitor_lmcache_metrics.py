#!/usr/bin/env python3
"""Scrape LMCache metrics endpoints and record Prometheus samples as JSONL."""

from __future__ import annotations

import argparse
import re
import time
import urllib.request
from pathlib import Path
from typing import Any

from common import append_jsonl, event_base, load_manifest


METRIC_RE = re.compile(r"^([A-Za-z_:][A-Za-z0-9_:]*)(\{[^}]*\})?\s+([-+eE0-9.]+)$")
LABEL_RE = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)="([^"]*)"')

DEFAULT_KEEP_PREFIXES = (
    "lmcache:",
    "lmcache_",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--url",
        action="append",
        default=[],
        help="LMCache metrics URL. Repeat for scheduler/worker endpoints.",
    )
    p.add_argument("--interval", type=float, default=1.0)
    p.add_argument("--keep-prefix", action="append", default=[], help="Metric name prefix to retain")
    return p.parse_args()


def parse_metrics(text: str, keep_prefixes: tuple[str, ...]) -> list[dict[str, Any]]:
    rows = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = METRIC_RE.match(line)
        if not m:
            continue
        name, label_text, value_text = m.groups()
        if keep_prefixes and not any(name.startswith(prefix) for prefix in keep_prefixes):
            continue
        try:
            value = float(value_text)
        except ValueError:
            continue
        rows.append(
            {
                "metric": name,
                "labels": dict(LABEL_RE.findall(label_text or "")),
                "value": value,
            }
        )
    return rows


def scrape_one(url: str, timeout_s: float, keep: tuple[str, ...]) -> tuple[str, list[dict[str, Any]], str | None]:
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            text = resp.read().decode("utf-8", errors="replace")
        return "ok", parse_metrics(text, keep), None
    except Exception as exc:
        return "error", [], str(exc)


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    out = Path(args.out)
    urls = args.url or ["http://127.0.0.1:6999/metrics"]
    keep = tuple(args.keep_prefix or DEFAULT_KEEP_PREFIXES)
    print(f"Scraping LMCache metrics {urls} every {args.interval}s -> {out}")
    try:
        while True:
            for url in urls:
                status, samples, error = scrape_one(url, timeout_s=10, keep=keep)
                row = {
                    **event_base(manifest, source="lmcache_metrics", event_type="metrics_sample"),
                    "url": url,
                    "status": status,
                    "sample_count": len(samples),
                    "samples": samples,
                }
                if error is not None:
                    row["error"] = error
                append_jsonl(out, row)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("Stopped LMCache metrics monitor")


if __name__ == "__main__":
    main()
