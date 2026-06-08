#!/usr/bin/env python3
"""Create a run manifest with one shared time origin for all workload logs."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from common import create_manifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", required=True, help="Workload mode, for example direct-batch, agent-c1, agent-c4")
    p.add_argument("--dataset", required=True, help="Prepared dataset JSONL")
    p.add_argument("--model", required=True, help="Model name served by vLLM")
    p.add_argument("--vllm-url", default="http://127.0.0.1:8000/v1", help="Real vLLM OpenAI API base URL")
    p.add_argument("--out", required=True, help="Run output directory")
    p.add_argument("--run-id", default="", help="Optional stable run id")
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--notes", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_id = args.run_id or f"{args.mode}-{time.strftime('%Y%m%d-%H%M%S')}"
    manifest = create_manifest(
        Path(args.out),
        run_id=run_id,
        mode=args.mode,
        dataset_path=args.dataset,
        model=args.model,
        vllm_url=args.vllm_url,
        extra={
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "notes": args.notes,
        },
    )
    print(Path(args.out) / "run_manifest.json")
    print(f"run_id={manifest['run_id']}")


if __name__ == "__main__":
    main()
