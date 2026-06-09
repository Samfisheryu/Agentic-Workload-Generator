#!/usr/bin/env python3
"""Summarize per-request LMCache KV load wait trace events."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    vals = sorted(values)
    pos = (len(vals) - 1) * pct / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(vals) - 1)
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def summarize_values(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "count": len(values),
        "mean": mean(values),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": max(values),
    }


def response_id_from_engine_req_id(req_id: str) -> str:
    parts = req_id.rsplit("-", 2)
    if len(parts) == 3 and parts[1].isdigit():
        return parts[0]
    parts = req_id.rsplit("-", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return req_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out-csv", default="")
    parser.add_argument("--out-json", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    trace_rows = read_jsonl(run_dir / "kv_wait_trace.jsonl")
    client_rows = read_jsonl(run_dir / "client_results.jsonl")
    out_csv = Path(args.out_csv) if args.out_csv else run_dir / "kv_wait_per_request.csv"
    out_json = Path(args.out_json) if args.out_json else run_dir / "kv_wait_summary.json"

    client_by_response_id = {
        row.get("vllm_response_id"): row
        for row in client_rows
        if row.get("vllm_response_id")
    }

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trace_rows:
        req_id = row.get("req_id")
        if req_id:
            grouped[str(req_id)].append(row)

    request_rows: list[dict[str, Any]] = []
    for req_id, rows in grouped.items():
        allowed = [r for r in rows if r.get("event_type") == "kv_load_allowed"]
        adapter_done = [r for r in rows if r.get("event_type") == "kv_retrieve_done"]
        engine_done = [
            r for r in rows if r.get("event_type") == "kv_retrieve_engine_done"
        ]
        if not adapter_done and not engine_done:
            continue

        allowed_ns = [int(r["ts_ns"]) for r in allowed if isinstance(r.get("ts_ns"), int)]
        start_ns = [
            int(r["retrieve_start_ns"])
            for r in adapter_done
            if isinstance(r.get("retrieve_start_ns"), int)
        ]
        end_ns = [
            int(r["retrieve_end_ns"])
            for r in adapter_done
            if isinstance(r.get("retrieve_end_ns"), int)
        ]
        adapter_ms = [
            float(r["adapter_retrieve_ms"])
            for r in adapter_done
            if isinstance(r.get("adapter_retrieve_ms"), (int, float))
        ]
        engine_ms = [
            float(r["engine_retrieve_ms"])
            for r in engine_done
            if isinstance(r.get("engine_retrieve_ms"), (int, float))
        ]
        process_ms = [
            float(r["process_tokens_ms"])
            for r in engine_done
            if isinstance(r.get("process_tokens_ms"), (int, float))
        ]
        to_gpu_ms = [
            float(r["to_gpu_ms"])
            for r in engine_done
            if isinstance(r.get("to_gpu_ms"), (int, float))
        ]
        need_to_load_tokens = [
            int(r["need_to_load_tokens"])
            for r in rows
            if isinstance(r.get("need_to_load_tokens"), int)
        ]
        retrieved_tokens = [
            int(r["retrieved_tokens"])
            for r in adapter_done
            if isinstance(r.get("retrieved_tokens"), int)
        ]
        worker_ids = {
            str(r.get("worker_id"))
            for r in adapter_done + engine_done
            if r.get("worker_id") is not None
        }

        exposed_wait_ms = None
        if allowed_ns and end_ns:
            exposed_wait_ms = (max(end_ns) - min(allowed_ns)) / 1e6

        scheduler_to_worker_start_ms = None
        if allowed_ns and start_ns:
            scheduler_to_worker_start_ms = (min(start_ns) - min(allowed_ns)) / 1e6

        worker_window_ms = None
        if start_ns and end_ns:
            worker_window_ms = (max(end_ns) - min(start_ns)) / 1e6

        client = client_by_response_id.get(req_id)
        if client is None:
            client = client_by_response_id.get(response_id_from_engine_req_id(req_id))
        if client is None:
            client = {}
        request_rows.append(
            {
                "req_id": req_id,
                "event_id": client.get("event_id"),
                "workflow_id": client.get("workflow_id"),
                "agent_id": client.get("agent_id"),
                "step_id": client.get("step_id"),
                "prompt_tokens": client.get("prompt_tokens"),
                "completion_tokens": client.get("completion_tokens"),
                "ttft_s": client.get("ttft_s"),
                "latency_s": client.get("latency_s"),
                "worker_count": len(worker_ids),
                "need_to_load_tokens": max(need_to_load_tokens)
                if need_to_load_tokens
                else None,
                "retrieved_tokens_max_rank": max(retrieved_tokens)
                if retrieved_tokens
                else None,
                "kv_exposed_wait_ms": exposed_wait_ms,
                "scheduler_to_worker_start_ms": scheduler_to_worker_start_ms,
                "worker_retrieve_window_ms": worker_window_ms,
                "adapter_retrieve_ms_max_rank": max(adapter_ms)
                if adapter_ms
                else None,
                "engine_retrieve_ms_max_rank": max(engine_ms) if engine_ms else None,
                "engine_process_tokens_ms_max_rank": max(process_ms)
                if process_ms
                else None,
                "engine_to_gpu_ms_max_rank": max(to_gpu_ms) if to_gpu_ms else None,
            }
        )

    request_rows.sort(key=lambda row: row["req_id"])
    fieldnames = [
        "req_id",
        "event_id",
        "workflow_id",
        "agent_id",
        "step_id",
        "prompt_tokens",
        "completion_tokens",
        "ttft_s",
        "latency_s",
        "worker_count",
        "need_to_load_tokens",
        "retrieved_tokens_max_rank",
        "kv_exposed_wait_ms",
        "scheduler_to_worker_start_ms",
        "worker_retrieve_window_ms",
        "adapter_retrieve_ms_max_rank",
        "engine_retrieve_ms_max_rank",
        "engine_process_tokens_ms_max_rank",
        "engine_to_gpu_ms_max_rank",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(request_rows)

    summary = {
        "trace_event_count": len(trace_rows),
        "request_count": len(request_rows),
        "joined_client_count": sum(1 for row in request_rows if row.get("event_id")),
        "kv_exposed_wait_ms": summarize_values(
            [
                float(row["kv_exposed_wait_ms"])
                for row in request_rows
                if isinstance(row.get("kv_exposed_wait_ms"), (int, float))
            ]
        ),
        "adapter_retrieve_ms_max_rank": summarize_values(
            [
                float(row["adapter_retrieve_ms_max_rank"])
                for row in request_rows
                if isinstance(row.get("adapter_retrieve_ms_max_rank"), (int, float))
            ]
        ),
        "engine_retrieve_ms_max_rank": summarize_values(
            [
                float(row["engine_retrieve_ms_max_rank"])
                for row in request_rows
                if isinstance(row.get("engine_retrieve_ms_max_rank"), (int, float))
            ]
        ),
    }
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")

    print(out_csv)
    print(out_json)


if __name__ == "__main__":
    main()
