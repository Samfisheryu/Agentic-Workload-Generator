from __future__ import annotations

import argparse
import csv
from collections import Counter, deque
from pathlib import Path
from typing import Any

try:
    from .common import dump_json, read_jsonl, stats
except ImportError:
    from common import dump_json, read_jsonl, stats  # type: ignore


def summarize(events: list[dict[str, Any]], working_set_window_s: float) -> dict[str, Any]:
    prefix_counts = Counter(e["prefix_id"] for e in events)
    workflows = {e["workflow_id"] for e in events}
    prefixes = {e["prefix_id"] for e in events}
    prompt_prefixes = {
        e.get("prompt_prefix_id", e["prefix_id"]) for e in events
    }
    gaps = [
        float(e["reactivation_gap_s"])
        for e in events
        if e.get("reactivation_gap_s") is not None
    ]
    suffix_lens = [float(e["suffix_len_tokens"]) for e in events]
    prefix_lens = [float(e["prefix_len_tokens"]) for e in events]
    history_lens = [
        float(e.get("history_len_tokens", 0)) for e in events
    ]
    max_tokens = [float(e["max_tokens"]) for e in events]
    prompt_tokens = [float(e["prefix_len_tokens"] + e["suffix_len_tokens"]) for e in events]
    reuse_events = sum(1 for e in events if e.get("expected_prefix_reuse"))

    active: deque[tuple[float, str, int]] = deque()
    latest: dict[str, tuple[float, int]] = {}
    working_set_values: list[float] = []
    for e in sorted(events, key=lambda x: (x["timestamp_s"], x["event_id"])):
        ts = float(e["timestamp_s"])
        prefix_id = e["prefix_id"]
        prefix_len = int(e["prefix_len_tokens"])
        latest[prefix_id] = (ts, prefix_len)
        active.append((ts, prefix_id, prefix_len))
        cutoff = ts - working_set_window_s
        while active and active[0][0] < cutoff:
            active.popleft()
        live_tokens = 0
        live_prefixes = set()
        for item_ts, item_prefix, item_len in active:
            if item_prefix in live_prefixes:
                continue
            latest_ts, _ = latest[item_prefix]
            if latest_ts == item_ts:
                live_prefixes.add(item_prefix)
                live_tokens += item_len
        working_set_values.append(float(live_tokens))

    return {
        "num_events": len(events),
        "num_workflows": len(workflows),
        "num_prefixes": len(prefixes),
        "num_prompt_prefixes": len(prompt_prefixes),
        "prompt_mode": events[0].get("prompt_mode", "static_prefix") if events else None,
        "total_prompt_tokens": int(sum(prompt_tokens)),
        "total_expected_completion_tokens": int(sum(max_tokens)),
        "prefix_reuse_events": reuse_events,
        "prefix_reuse_rate": reuse_events / len(events) if events else 0.0,
        "branch_fanout_events": sum(1 for e in events if e.get("is_branch_fanout")),
        "reactivation_gap_s": stats(gaps),
        "prefix_len_tokens": stats(prefix_lens),
        "history_len_tokens": stats(history_lens),
        "suffix_len_tokens": stats(suffix_lens),
        "prompt_len_tokens": stats(prompt_tokens),
        "max_tokens": stats(max_tokens),
        "live_prefix_working_set_tokens": stats(working_set_values),
        "working_set_window_s": working_set_window_s,
        "top_prefixes": [
            {"prefix_id": prefix_id, "count": count}
            for prefix_id, count in prefix_counts.most_common(20)
        ],
    }


def write_prefix_counts(path: Path, events: list[dict[str, Any]]) -> None:
    counts = Counter(e["prefix_id"] for e in events)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["prefix_id", "count"])
        for prefix_id, count in counts.most_common():
            writer.writerow([prefix_id, count])


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a generated HiddenCache trace.")
    parser.add_argument("--trace", required=True, help="Path to trace.jsonl.")
    parser.add_argument("--out-dir", default=None, help="Output directory. Defaults to trace dir.")
    parser.add_argument(
        "--working-set-window-s",
        type=float,
        default=10.0,
        help="Window for approximate live prefix working set.",
    )
    args = parser.parse_args()

    trace_path = Path(args.trace)
    out_dir = Path(args.out_dir) if args.out_dir else trace_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    events = read_jsonl(trace_path)
    summary = summarize(events, args.working_set_window_s)
    dump_json(out_dir / "trace_summary.json", summary)
    write_prefix_counts(out_dir / "prefix_reuse_counts.csv", events)
    print(f"wrote {out_dir / 'trace_summary.json'}")


if __name__ == "__main__":
    main()
