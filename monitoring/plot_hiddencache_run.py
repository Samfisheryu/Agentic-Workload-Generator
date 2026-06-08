#!/usr/bin/env python3
"""Generate plots for one HiddenCache workload-generator run."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import load_manifest, read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out-dir", default="")
    return parser.parse_args()


def numeric(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    vals = [float(r[key]) for r in rows if isinstance(r.get(key), (int, float))]
    return np.asarray(vals, dtype=float)


def monitor_rel_s(rows: list[dict[str, Any]]) -> np.ndarray:
    return numeric(rows, "t_rel_ms") / 1000.0


def client_rel_s(rows: list[dict[str, Any]], manifest: dict[str, Any], key: str) -> np.ndarray:
    start = float(manifest["run_start_wall_ns"]) / 1e9
    return np.asarray(
        [float(r[key]) - start for r in rows if isinstance(r.get(key), (int, float))],
        dtype=float,
    )


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(path)


def quantiles(values: np.ndarray) -> dict[str, float | None]:
    if values.size == 0:
        return {"p50": None, "p90": None, "p95": None, "p99": None}
    return {
        "p50": float(np.nanpercentile(values, 50)),
        "p90": float(np.nanpercentile(values, 90)),
        "p95": float(np.nanpercentile(values, 95)),
        "p99": float(np.nanpercentile(values, 99)),
    }


def metric_value(
    row: dict[str, Any],
    metric_names: str | list[str],
    labels: dict[str, str] | None = None,
) -> float | None:
    names = [metric_names] if isinstance(metric_names, str) else metric_names
    vals: list[float] = []
    for sample in row.get("samples", []):
        if sample.get("metric") not in names:
            continue
        sample_labels = sample.get("labels") or {}
        if labels and any(sample_labels.get(k) != v for k, v in labels.items()):
            continue
        value = sample.get("value")
        if isinstance(value, (int, float)):
            vals.append(float(value))
    if not vals:
        return None
    return float(sum(vals))


def metric_series(
    rows: list[dict[str, Any]],
    names: str | list[str],
    labels: dict[str, str] | None = None,
) -> np.ndarray:
    out = []
    for row in rows:
        value = metric_value(row, names, labels=labels)
        out.append(np.nan if value is None else value)
    return np.asarray(out, dtype=float)


def first_valid(values: np.ndarray) -> float:
    valid = values[~np.isnan(values)]
    return float(valid[0]) if valid.size else 0.0


def group_by_gpu(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        idx = row.get("gpu_index")
        if isinstance(idx, int):
            grouped[idx].append(row)
    return dict(grouped)


def plot_client(run_dir: Path, out_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    rows = [r for r in read_jsonl(run_dir / "client_results.jsonl") if r.get("status") == "ok"]
    summary: dict[str, Any] = {"request_count": len(rows)}
    if not rows:
        return summary

    latency = numeric(rows, "latency_s")
    ttft = numeric(rows, "ttft_s")
    prompt = numeric(rows, "prompt_tokens")
    completion = numeric(rows, "completion_tokens")
    max_tokens = numeric(rows, "max_tokens")
    send_t = client_rel_s(rows, manifest, "send_ts")
    done_t = client_rel_s(rows, manifest, "done_ts")

    summary.update(
        {
            "latency_s": quantiles(latency),
            "ttft_s": quantiles(ttft),
            "prompt_tokens": quantiles(prompt),
            "completion_tokens": quantiles(completion),
            "total_prompt_tokens": int(np.nansum(prompt)),
            "total_completion_tokens": int(np.nansum(completion)),
            "max_completion_budget_tokens": int(np.nansum(max_tokens)),
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes[0, 0].hist(latency, bins=40, color="#4C78A8", edgecolor="white")
    axes[0, 0].set_title("Request Latency")
    axes[0, 0].set_xlabel("seconds")
    axes[0, 0].set_ylabel("requests")

    if ttft.size:
        axes[0, 1].hist(ttft, bins=40, color="#72B7B2", edgecolor="white")
    axes[0, 1].set_title("TTFT")
    axes[0, 1].set_xlabel("seconds")
    axes[0, 1].set_ylabel("requests")

    axes[1, 0].hist(prompt, bins=40, color="#F58518", edgecolor="white")
    axes[1, 0].set_title("Prompt Tokens")
    axes[1, 0].set_xlabel("tokens")
    axes[1, 0].set_ylabel("requests")

    axes[1, 1].scatter(prompt, latency, s=14, alpha=0.75, color="#B279A2")
    axes[1, 1].set_title("Prompt Tokens vs Latency")
    axes[1, 1].set_xlabel("prompt tokens")
    axes[1, 1].set_ylabel("latency seconds")
    axes[1, 1].grid(alpha=0.25)
    save(fig, out_dir / "client_request_overview.png")

    order = np.argsort(done_t)
    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
    axes[0].scatter(send_t, prompt, s=13, alpha=0.75, color="#F58518")
    axes[0].set_title("Prompt Size Over Run Time")
    axes[0].set_ylabel("tokens")
    axes[0].grid(alpha=0.25)

    axes[1].scatter(done_t, latency, s=13, alpha=0.75, color="#E45756")
    axes[1].set_title("Latency Over Run Time")
    axes[1].set_ylabel("seconds")
    axes[1].grid(alpha=0.25)

    if ttft.size:
        axes[2].scatter(done_t[: ttft.size], ttft, s=13, alpha=0.75, color="#72B7B2")
    axes[2].set_title("TTFT Over Run Time")
    axes[2].set_ylabel("seconds")
    axes[2].grid(alpha=0.25)

    axes[3].plot(done_t[order], np.arange(1, len(rows) + 1), color="#4C78A8")
    axes[3].set_title("Cumulative Completed Requests")
    axes[3].set_xlabel("seconds since run start")
    axes[3].set_ylabel("requests")
    axes[3].grid(alpha=0.25)
    save(fig, out_dir / "client_timeline.png")
    return summary


def plot_vllm(run_dir: Path, out_dir: Path) -> dict[str, Any]:
    rows = [r for r in read_jsonl(run_dir / "vllm_metrics.jsonl") if r.get("status") == "ok"]
    summary: dict[str, Any] = {"vllm_scrapes": len(rows)}
    if not rows:
        return summary
    t = monitor_rel_s(rows)

    running = metric_series(rows, ["vllm:num_requests_running", "vllm_num_requests_running"])
    waiting = metric_series(rows, ["vllm:num_requests_waiting", "vllm_num_requests_waiting"])
    kv = metric_series(
        rows,
        [
            "vllm:kv_cache_usage_perc",
            "vllm:gpu_cache_usage_perc",
            "vllm_kv_cache_usage_perc",
            "vllm_gpu_cache_usage_perc",
        ],
    )
    if np.nanmax(kv) <= 1.5:
        kv = kv * 100.0

    prompt = metric_series(rows, ["vllm:prompt_tokens_total", "vllm_prompt_tokens_total"])
    generation = metric_series(rows, ["vllm:generation_tokens_total", "vllm_generation_tokens_total"])
    prefix_queries = metric_series(
        rows,
        [
            "vllm:prefix_cache_queries_total",
            "vllm_prefix_cache_queries_total",
            "vllm:prefix_cache_query_total",
            "vllm_prefix_cache_query_total",
        ],
    )
    prefix_hits = metric_series(
        rows,
        [
            "vllm:prefix_cache_hits_total",
            "vllm_prefix_cache_hits_total",
            "vllm:prefix_cache_hit_total",
            "vllm_prefix_cache_hit_total",
        ],
    )

    prompt_delta = prompt - first_valid(prompt)
    generation_delta = generation - first_valid(generation)
    query_delta = prefix_queries - first_valid(prefix_queries)
    hit_delta = prefix_hits - first_valid(prefix_hits)
    with np.errstate(divide="ignore", invalid="ignore"):
        hit_rate = np.where(query_delta > 0, hit_delta / query_delta * 100.0, np.nan)

    summary.update(
        {
            "max_running_requests": None if np.all(np.isnan(running)) else float(np.nanmax(running)),
            "max_waiting_requests": None if np.all(np.isnan(waiting)) else float(np.nanmax(waiting)),
            "max_kv_cache_usage_pct": None if np.all(np.isnan(kv)) else float(np.nanmax(kv)),
            "prompt_tokens_delta": None if np.all(np.isnan(prompt_delta)) else int(np.nanmax(prompt_delta)),
            "generation_tokens_delta": None
            if np.all(np.isnan(generation_delta))
            else int(np.nanmax(generation_delta)),
            "prefix_cache_hit_rate_pct_final": None
            if np.all(np.isnan(hit_rate))
            else float(hit_rate[~np.isnan(hit_rate)][-1]),
        }
    )

    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
    axes[0].plot(t, running, label="running", color="#4C78A8")
    axes[0].plot(t, waiting, label="waiting", color="#E45756")
    axes[0].set_title("vLLM Request Queue")
    axes[0].set_ylabel("requests")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(t, kv, color="#B279A2")
    axes[1].set_title("vLLM KV Cache Usage")
    axes[1].set_ylabel("percent")
    axes[1].grid(alpha=0.25)

    axes[2].plot(t, prompt_delta, label="prompt", color="#F58518")
    axes[2].plot(t, generation_delta, label="generation", color="#54A24B")
    axes[2].set_title("Cumulative Tokens Since Run Start")
    axes[2].set_ylabel("tokens")
    axes[2].legend()
    axes[2].grid(alpha=0.25)

    axes[3].plot(t, hit_rate, color="#72B7B2")
    axes[3].set_title("Prefix Cache Hit Rate Since Run Start")
    axes[3].set_xlabel("seconds since run start")
    axes[3].set_ylabel("percent")
    axes[3].grid(alpha=0.25)
    save(fig, out_dir / "vllm_kv_prefix_timeline.png")
    return summary


def metric_series_any(rows: list[dict[str, Any]], candidates: list[str]) -> np.ndarray:
    values = []
    for row in rows:
        total = 0.0
        found = False
        for sample in row.get("samples", []):
            name = str(sample.get("metric", ""))
            if name in candidates or any(name.endswith(candidate) for candidate in candidates):
                value = sample.get("value")
                if isinstance(value, (int, float)):
                    total += float(value)
                    found = True
        values.append(total if found else np.nan)
    return np.asarray(values, dtype=float)


def plot_lmcache(run_dir: Path, out_dir: Path) -> dict[str, Any]:
    rows_all = read_jsonl(run_dir / "lmcache_metrics.jsonl")
    rows = [r for r in rows_all if r.get("status") == "ok"]
    summary: dict[str, Any] = {
        "lmcache_scrapes": len(rows_all),
        "lmcache_successful_scrapes": len(rows),
    }
    if not rows_all:
        return summary
    if not rows:
        summary["lmcache_last_error"] = rows_all[-1].get("error")
        return summary

    t = monitor_rel_s(rows)
    lookup_req = metric_series_any(
        rows,
        [
            "lmcache_mp_lookup_requested_tokens_total",
            "lmcache_mp_lookup_requested_total",
            "lmcache:num_lookup_requests_total",
            "lmcache:num_lookup_requests",
        ],
    )
    lookup_hit = metric_series_any(
        rows,
        [
            "lmcache_mp_lookup_hit_tokens_total",
            "lmcache_mp_lookup_hit_total",
            "lmcache:num_lookup_hits_total",
            "lmcache:num_lookup_hits",
        ],
    )
    store_req = metric_series_any(rows, ["lmcache:num_store_requests_total", "lmcache:num_store_requests"])
    retrieve_req = metric_series_any(
        rows,
        ["lmcache:num_retrieve_requests_total", "lmcache:num_retrieve_requests"],
    )
    l1_ticks = metric_series_any(
        rows,
        ["lmcache_mp_l1_eviction_loop_ticks_total", "lmcache_mp_l1_eviction_loop_ticks"],
    )
    l1_triggered = metric_series_any(
        rows,
        ["lmcache_mp_l1_eviction_loop_triggered_total", "lmcache_mp_l1_eviction_loop_triggered"],
    )
    l1_usage = metric_series_any(rows, ["lmcache_mp_l1_usage_ratio", "lmcache_mp_l1_memory_usage_ratio"])
    l1_bytes = metric_series_any(rows, ["lmcache_mp_l1_memory_usage_bytes"])

    lookup_req_delta = lookup_req - first_valid(lookup_req)
    lookup_hit_delta = lookup_hit - first_valid(lookup_hit)
    store_delta = store_req - first_valid(store_req)
    retrieve_delta = retrieve_req - first_valid(retrieve_req)
    ticks_delta = l1_ticks - first_valid(l1_ticks)
    triggered_delta = l1_triggered - first_valid(l1_triggered)

    with np.errstate(divide="ignore", invalid="ignore"):
        hit_rate = np.where(lookup_req_delta > 0, lookup_hit_delta / lookup_req_delta * 100.0, np.nan)
        eviction_trigger_rate = np.where(ticks_delta > 0, triggered_delta / ticks_delta * 100.0, np.nan)

    def last_valid(values: np.ndarray) -> float | None:
        valid = values[~np.isnan(values)]
        return None if valid.size == 0 else float(valid[-1])

    summary.update(
        {
            "lmcache_lookup_requested_delta": last_valid(lookup_req_delta),
            "lmcache_lookup_hit_delta": last_valid(lookup_hit_delta),
            "lmcache_lookup_hit_rate_pct_final": last_valid(hit_rate),
            "lmcache_store_requests_delta": last_valid(store_delta),
            "lmcache_retrieve_requests_delta": last_valid(retrieve_delta),
            "lmcache_l1_eviction_ticks_delta": last_valid(ticks_delta),
            "lmcache_l1_eviction_triggered_delta": last_valid(triggered_delta),
            "lmcache_l1_eviction_trigger_rate_pct_final": last_valid(eviction_trigger_rate),
        }
    )

    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
    axes[0].plot(t, lookup_req_delta, label="lookup requested", color="#4C78A8")
    axes[0].plot(t, lookup_hit_delta, label="lookup hit", color="#54A24B")
    axes[0].set_title("LMCache Lookup Tokens")
    axes[0].set_ylabel("tokens")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(t, hit_rate, color="#72B7B2")
    axes[1].set_title("LMCache Lookup Hit Rate")
    axes[1].set_ylabel("percent")
    axes[1].grid(alpha=0.25)

    axes[2].plot(t, store_delta, label="store", color="#F58518")
    axes[2].plot(t, retrieve_delta, label="retrieve", color="#E45756")
    axes[2].set_title("LMCache Store/Retrieve Requests")
    axes[2].set_ylabel("requests")
    axes[2].legend()
    axes[2].grid(alpha=0.25)

    plotted_usage = False
    if not np.all(np.isnan(l1_usage)):
        usage = l1_usage * 100.0 if np.nanmax(l1_usage) <= 1.5 else l1_usage
        axes[3].plot(t, usage, label="L1 usage", color="#B279A2")
        axes[3].set_ylabel("percent")
        plotted_usage = True
    if not np.all(np.isnan(l1_bytes)):
        axes[3].plot(t, l1_bytes / (1024.0**3), label="L1 memory", color="#9D755D")
        axes[3].set_ylabel("GiB / percent")
        plotted_usage = True
    if not plotted_usage:
        axes[3].plot(t, ticks_delta, label="eviction ticks", color="#B279A2")
        axes[3].plot(t, triggered_delta, label="eviction triggered", color="#9D755D")
        axes[3].set_ylabel("count")
    axes[3].set_title("LMCache L1 Usage / Eviction")
    axes[3].set_xlabel("seconds since run start")
    axes[3].legend()
    axes[3].grid(alpha=0.25)
    save(fig, out_dir / "lmcache_metrics_timeline.png")
    return summary


def plot_hardware(run_dir: Path, out_dir: Path) -> dict[str, Any]:
    gpu_rows = read_jsonl(run_dir / "gpu.jsonl")
    pcie_rows = read_jsonl(run_dir / "pcie.jsonl")
    cpu_rows = read_jsonl(run_dir / "cpu.jsonl")
    summary: dict[str, Any] = {
        "gpu_samples": len(gpu_rows),
        "pcie_samples": len(pcie_rows),
        "cpu_samples": len(cpu_rows),
    }

    if gpu_rows or pcie_rows:
        fig, axes = plt.subplots(4, 1, figsize=(13, 12), sharex=True)
        for gpu, rows in sorted(group_by_gpu(gpu_rows).items()):
            axes[0].plot(monitor_rel_s(rows), numeric(rows, "util_gpu_pct"), label=f"GPU {gpu}")
        axes[0].set_title("GPU Utilization")
        axes[0].set_ylabel("percent")
        axes[0].legend()
        axes[0].grid(alpha=0.25)

        for gpu, rows in sorted(group_by_gpu(gpu_rows).items()):
            axes[1].plot(monitor_rel_s(rows), numeric(rows, "mem_used_mb") / 1024.0, label=f"GPU {gpu}")
        axes[1].set_title("GPU Memory")
        axes[1].set_ylabel("GiB")
        axes[1].legend()
        axes[1].grid(alpha=0.25)

        for gpu, rows in sorted(group_by_gpu(pcie_rows).items()):
            axes[2].plot(monitor_rel_s(rows), numeric(rows, "pcie_rx_mb_s"), label=f"GPU {gpu} RX")
        axes[2].set_title("PCIe RX")
        axes[2].set_ylabel("MB/s")
        axes[2].legend()
        axes[2].grid(alpha=0.25)

        for gpu, rows in sorted(group_by_gpu(pcie_rows).items()):
            axes[3].plot(monitor_rel_s(rows), numeric(rows, "pcie_tx_mb_s"), label=f"GPU {gpu} TX")
        axes[3].set_title("PCIe TX")
        axes[3].set_xlabel("seconds since run start")
        axes[3].set_ylabel("MB/s")
        axes[3].legend()
        axes[3].grid(alpha=0.25)
        save(fig, out_dir / "hardware_gpu_pcie_timeline.png")

    if cpu_rows:
        fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
        by_pid: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in cpu_rows:
            if isinstance(row.get("pid"), int):
                by_pid[row["pid"]].append(row)
        for pid, rows in sorted(by_pid.items()):
            label = f"{rows[0].get('name')}:{pid}"
            axes[0].plot(monitor_rel_s(rows), numeric(rows, "cpu_pct"), label=label)
            axes[1].plot(monitor_rel_s(rows), numeric(rows, "rss_mb"), label=label)
        axes[0].set_title("Process CPU")
        axes[0].set_ylabel("percent")
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=0.25)
        axes[1].set_title("Process RSS")
        axes[1].set_xlabel("seconds since run start")
        axes[1].set_ylabel("MiB")
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.25)
        save(fig, out_dir / "hardware_cpu_timeline.png")

    return summary


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(run_dir)

    summary: dict[str, Any] = {
        "run_id": manifest.get("run_id"),
        "mode": manifest.get("mode"),
        "model": manifest.get("model"),
        "config_path": manifest.get("config_path"),
        "backend": manifest.get("backend"),
    }
    summary.update(plot_client(run_dir, out_dir, manifest))
    summary.update(plot_vllm(run_dir, out_dir))
    summary.update(plot_lmcache(run_dir, out_dir))
    summary.update(plot_hardware(run_dir, out_dir))
    write_json(out_dir / "plot_summary.json", summary)
    print(out_dir / "plot_summary.json")


if __name__ == "__main__":
    main()
