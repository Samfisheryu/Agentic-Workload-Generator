#!/usr/bin/env python3
"""Scrape vLLM /metrics and record selected Prometheus samples as JSONL."""

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
    "vllm:",
    "vllm_",
    "lmcache:",
    "lmcache_",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--url", default="http://127.0.0.1:8000/metrics")
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
        labels = dict(LABEL_RE.findall(label_text or ""))
        rows.append({"metric": name, "labels": labels, "value": value})
    return rows


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    out = Path(args.out)
    keep = tuple(args.keep_prefix or DEFAULT_KEEP_PREFIXES)
    print(f"Scraping {args.url} every {args.interval}s -> {out}")
    try:
        while True:
            try:
                with urllib.request.urlopen(args.url, timeout=10) as resp:
                    text = resp.read().decode("utf-8", errors="replace")
                base = event_base(manifest, source="vllm_metrics", event_type="metrics_sample")
                samples = parse_metrics(text, keep)
                append_jsonl(out, {**base, "status": "ok", "sample_count": len(samples), "samples": samples})
            except Exception as exc:
                append_jsonl(
                    out,
                    {
                        **event_base(manifest, source="vllm_metrics", event_type="metrics_sample"),
                        "status": "error",
                        "error": str(exc),
                        "sample_count": 0,
                        "samples": [],
                    },
                )
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("Stopped vLLM metrics monitor")


if __name__ == "__main__":
    main()
