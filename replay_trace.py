from __future__ import annotations

import argparse
import concurrent.futures
import json
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import requests

try:
    from .common import read_jsonl
except ImportError:
    from common import read_jsonl  # type: ignore


def load_prefix_bank(path: str | Path) -> dict[str, str]:
    rows = read_jsonl(path)
    return {row["prefix_id"]: row["prefix_text"] for row in rows}


def build_payload(
    event: dict[str, Any],
    prompt: str,
    model: str,
    endpoint: str,
    stream: bool,
    ignore_eos: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": int(event["max_tokens"]),
        "temperature": float(event.get("temperature", 0.0)),
        "top_p": float(event.get("top_p", 1.0)),
        "stream": stream,
    }
    if endpoint == "chat":
        payload["messages"] = [{"role": "user", "content": prompt}]
    else:
        payload["prompt"] = prompt
    if stream:
        payload["stream_options"] = {"include_usage": True}
    if ignore_eos:
        payload["ignore_eos"] = True
    return payload


def parse_stream_response(resp: requests.Response) -> tuple[float | None, dict[str, Any] | None]:
    first_token_ts: float | None = None
    usage: dict[str, Any] | None = None
    for raw_line in resp.iter_lines(decode_unicode=True):
        if not raw_line:
            continue
        line = raw_line.strip()
        if line.startswith("data:"):
            line = line[len("data:") :].strip()
        if line == "[DONE]":
            break
        try:
            chunk = json.loads(line)
        except json.JSONDecodeError:
            continue
        if chunk.get("usage"):
            usage = chunk["usage"]
        choices = chunk.get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        text_piece = choice.get("text")
        delta = choice.get("delta") or {}
        if text_piece or delta.get("content"):
            if first_token_ts is None:
                first_token_ts = time.time()
    return first_token_ts, usage


def send_one(
    event: dict[str, Any],
    prefix_text: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    prompt = prefix_text + "\n" + event.get("suffix_text", "")
    endpoint_path = "/v1/chat/completions" if args.endpoint == "chat" else "/v1/completions"
    url = args.base_url.rstrip("/")
    if url.endswith("/v1"):
        url = url[: -len("/v1")]
    url = url + endpoint_path
    payload = build_payload(
        event,
        prompt,
        args.model,
        args.endpoint,
        stream=not args.no_stream,
        ignore_eos=args.ignore_eos,
    )

    send_ts = time.time()
    first_token_ts: float | None = None
    done_ts: float | None = None
    status = "ok"
    http_status: int | None = None
    error: str | None = None
    usage: dict[str, Any] | None = None
    response_id: str | None = None
    try:
        resp = requests.post(
            url,
            json=payload,
            stream=not args.no_stream,
            timeout=args.timeout_s,
        )
        http_status = resp.status_code
        if resp.status_code >= 400:
            status = "error"
            error = resp.text[:1000]
        elif args.no_stream:
            body = resp.json()
            done_ts = time.time()
            usage = body.get("usage")
            response_id = body.get("id")
        else:
            first_token_ts, usage = parse_stream_response(resp)
            done_ts = time.time()
    except Exception as exc:
        status = "error"
        error = f"{type(exc).__name__}: {exc}"
        done_ts = time.time()

    if done_ts is None:
        done_ts = time.time()
    prompt_tokens = None
    completion_tokens = None
    if usage:
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
    if prompt_tokens is None:
        prompt_tokens = int(event.get("prefix_len_tokens", 0)) + int(
            event.get("suffix_len_tokens", 0)
        )
    if completion_tokens is None:
        completion_tokens = None

    return {
        "event_id": event["event_id"],
        "workflow_id": event["workflow_id"],
        "agent_id": event["agent_id"],
        "prefix_id": event["prefix_id"],
        "prompt_prefix_id": event.get("prompt_prefix_id"),
        "send_ts": send_ts,
        "first_token_ts": first_token_ts,
        "done_ts": done_ts,
        "status": status,
        "http_status": http_status,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "latency_s": done_ts - send_ts,
        "ttft_s": None if first_token_ts is None else first_token_ts - send_ts,
        "vllm_response_id": response_id,
        "planned_timestamp_s": event.get("timestamp_s"),
        "step_id": event.get("step_id"),
        "branch_id": event.get("branch_id"),
        "max_tokens": event.get("max_tokens"),
        "error": error,
    }


class JsonlWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.lock = threading.Lock()
        self.f = self.path.open("w", encoding="utf-8")

    def write(self, row: dict[str, Any]) -> None:
        with self.lock:
            self.f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            self.f.flush()

    def close(self) -> None:
        self.f.close()


def run_open_loop(
    events: list[dict[str, Any]],
    prefixes: dict[str, str],
    args: argparse.Namespace,
    writer: JsonlWriter,
) -> None:
    start_wall = time.time() + args.start_delay_s
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures: list[concurrent.futures.Future[dict[str, Any]]] = []
        for event in sorted(events, key=lambda x: (x["timestamp_s"], x["event_id"])):
            target = start_wall + float(event["timestamp_s"])
            sleep_s = target - time.time()
            if sleep_s > 0:
                time.sleep(sleep_s)
            prefix_key = event.get("prompt_prefix_id", event["prefix_id"])
            future = pool.submit(send_one, event, prefixes[prefix_key], args)
            future.add_done_callback(lambda fut: writer.write(fut.result()))
            futures.append(future)
        for future in concurrent.futures.as_completed(futures):
            future.result()


def run_closed_loop(
    events: list[dict[str, Any]],
    prefixes: dict[str, str],
    args: argparse.Namespace,
    writer: JsonlWriter,
) -> None:
    by_workflow: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        by_workflow[event["workflow_id"]].append(event)

    def run_workflow(workflow_events: list[dict[str, Any]]) -> None:
        workflow_events = sorted(workflow_events, key=lambda x: (x["timestamp_s"], x["event_id"]))
        prev_planned = workflow_events[0]["timestamp_s"]
        for event in workflow_events:
            planned_delta = float(event["timestamp_s"]) - float(prev_planned)
            if planned_delta > 0:
                time.sleep(planned_delta)
            prefix_key = event.get("prompt_prefix_id", event["prefix_id"])
            row = send_one(event, prefixes[prefix_key], args)
            writer.write(row)
            prev_planned = event["timestamp_s"]

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = [pool.submit(run_workflow, evs) for evs in by_workflow.values()]
        for future in concurrent.futures.as_completed(futures):
            future.result()


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a HiddenCache trace against vLLM.")
    parser.add_argument("--trace", required=True, help="Path to trace.jsonl.")
    parser.add_argument("--prefix-bank", required=True, help="Path to prefix_bank.jsonl.")
    parser.add_argument("--results", required=True, help="Output client_results.jsonl.")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=["open-loop", "closed-loop"], default="open-loop")
    parser.add_argument("--endpoint", choices=["completions", "chat"], default="completions")
    parser.add_argument("--max-workers", type=int, default=256)
    parser.add_argument("--timeout-s", type=float, default=600.0)
    parser.add_argument("--start-delay-s", type=float, default=2.0)
    parser.add_argument("--no-stream", action="store_true")
    parser.add_argument("--ignore-eos", action="store_true")
    args = parser.parse_args()

    events = read_jsonl(args.trace)
    prefixes = load_prefix_bank(args.prefix_bank)
    writer = JsonlWriter(args.results)
    try:
        if args.mode == "open-loop":
            run_open_loop(events, prefixes, args, writer)
        else:
            run_closed_loop(events, prefixes, args, writer)
    finally:
        writer.close()


if __name__ == "__main__":
    main()
