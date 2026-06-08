#!/usr/bin/env python3
"""Sample GPU utilization, memory, power, temperature, and PCIe throughput."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from common import append_jsonl, event_base, load_manifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--interval", type=float, default=1.0)
    p.add_argument("--gpus", default="", help="Comma-separated GPU indices; default all")
    return p.parse_args()


def call_or_none(fn, *args) -> Any:
    try:
        return fn(*args)
    except Exception:
        return None


def main() -> None:
    args = parse_args()
    try:
        import pynvml
    except ImportError as exc:
        raise SystemExit("Install pynvml to run GPU monitoring") from exc

    manifest = load_manifest(args.manifest)
    out = Path(args.out)
    pynvml.nvmlInit()
    try:
        count = pynvml.nvmlDeviceGetCount()
        indices = [int(x) for x in args.gpus.split(",") if x.strip()] if args.gpus else list(range(count))
        print(f"Sampling GPUs {indices} every {args.interval}s -> {out}")
        while True:
            for idx in indices:
                handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                util = call_or_none(pynvml.nvmlDeviceGetUtilizationRates, handle)
                mem = call_or_none(pynvml.nvmlDeviceGetMemoryInfo, handle)
                name = call_or_none(pynvml.nvmlDeviceGetName, handle)
                uuid = call_or_none(pynvml.nvmlDeviceGetUUID, handle)
                power_mw = call_or_none(pynvml.nvmlDeviceGetPowerUsage, handle)
                temp_c = call_or_none(pynvml.nvmlDeviceGetTemperature, handle, pynvml.NVML_TEMPERATURE_GPU)
                rx_kb_s = call_or_none(pynvml.nvmlDeviceGetPcieThroughput, handle, pynvml.NVML_PCIE_UTIL_RX_BYTES)
                tx_kb_s = call_or_none(pynvml.nvmlDeviceGetPcieThroughput, handle, pynvml.NVML_PCIE_UTIL_TX_BYTES)
                event = event_base(manifest, source="gpu_nvml", event_type="gpu_sample")
                event.update({
                    "gpu_index": idx,
                    "gpu_name": name.decode("utf-8") if isinstance(name, bytes) else name,
                    "gpu_uuid": uuid.decode("utf-8") if isinstance(uuid, bytes) else uuid,
                    "util_gpu_pct": getattr(util, "gpu", None),
                    "util_mem_pct": getattr(util, "memory", None),
                    "mem_used_mb": mem.used / 1024 / 1024 if mem else None,
                    "mem_total_mb": mem.total / 1024 / 1024 if mem else None,
                    "power_w": power_mw / 1000 if power_mw is not None else None,
                    "temperature_c": temp_c,
                    "pcie_rx_mb_s": rx_kb_s / 1024 if rx_kb_s is not None else None,
                    "pcie_tx_mb_s": tx_kb_s / 1024 if tx_kb_s is not None else None,
                })
                append_jsonl(out, event)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("Stopped GPU monitor")
    finally:
        pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
