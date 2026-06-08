from __future__ import annotations

import argparse
import heapq
import json
import random
from pathlib import Path
from typing import Any

try:
    from .common import (
        TokenizerAdapter,
        choose_delay,
        dump_json,
        load_config,
        make_text_to_length,
        role_param,
        sample_from_param,
        stable_int,
        write_jsonl,
    )
except ImportError:
    from common import (  # type: ignore
        TokenizerAdapter,
        choose_delay,
        dump_json,
        load_config,
        make_text_to_length,
        role_param,
        sample_from_param,
        stable_int,
        write_jsonl,
    )


def unique_roles(config: dict[str, Any]) -> list[str]:
    roles: list[str] = []
    for role in config.get("agents", []):
        if role not in roles:
            roles.append(role)
    for role in config.get("workflow_graph", []):
        if role not in roles:
            roles.append(role)
    return roles


def prompt_mode(config: dict[str, Any]) -> str:
    return str(config.get("prompt_mode", "static_prefix"))


def build_prefix_bank(
    config: dict[str, Any], tokenizer: TokenizerAdapter
) -> list[dict[str, Any]]:
    seed = int(config["seed"])
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    mode = prompt_mode(config)
    roles = ["base"] if mode == "growing_history" else unique_roles(config)
    for wf_idx in range(int(config["num_workflows"])):
        workflow_id = f"wf_{wf_idx:05d}"
        for role in roles:
            target_len = int(role_param(config["prefix_len_tokens"], role, rng))
            prefix_id = f"{workflow_id}/{role}"
            label = f"prefix workflow={workflow_id} agent={role}"
            text, actual_len = make_text_to_length(
                tokenizer,
                target_len,
                label,
                seed + stable_int(prefix_id) % 1_000_000_000,
            )
            rows.append(
                {
                    "prefix_id": prefix_id,
                    "workflow_id": workflow_id,
                    "agent_id": role,
                    "target_prefix_len_tokens": target_len,
                    "actual_prefix_len_tokens": actual_len,
                    "prefix_text": text,
                }
            )
    return rows


def generate_static_prefix_events(
    config: dict[str, Any], tokenizer: TokenizerAdapter, prefix_by_id: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seed = int(config["seed"])
    rng = random.Random(seed)
    graph = list(config["workflow_graph"])
    if not graph:
        raise ValueError("workflow_graph must not be empty")

    heap: list[tuple[float, int]] = []
    states: dict[int, dict[str, Any]] = {}
    mean_think = float(config.get("mean_think_time_s", 0.5))
    for wf_idx in range(int(config["num_workflows"])):
        first_ts = rng.random() * max(mean_think, 0.001)
        heapq.heappush(heap, (first_ts, wf_idx))
        states[wf_idx] = {
            "step_id": 0,
            "graph_index": 0,
            "last_event_id": None,
        }

    events: list[dict[str, Any]] = []
    last_prefix_seen: dict[str, float] = {}
    duration_s = float(config["duration_s"])
    max_events = config.get("max_events")
    max_events = int(max_events) if max_events is not None else None

    while heap:
        ts, wf_idx = heapq.heappop(heap)
        if ts > duration_s:
            break
        if max_events is not None and len(events) >= max_events:
            break

        state = states[wf_idx]
        workflow_id = f"wf_{wf_idx:05d}"
        agent_id = graph[state["graph_index"] % len(graph)]
        prefix_id = f"{workflow_id}/{agent_id}"
        prefix_row = prefix_by_id[prefix_id]
        prior_last_ts = last_prefix_seen.get(prefix_id)
        reactivation_gap = None if prior_last_ts is None else ts - prior_last_ts
        expected_reuse = prior_last_ts is not None

        fanout = int(sample_from_param(config.get("branch_fanout", 1), rng))
        fanout = max(1, fanout)
        parent_event_id = state["last_event_id"]
        branch_event_ids: list[str] = []
        for branch_id in range(fanout):
            if max_events is not None and len(events) >= max_events:
                break
            event_index = len(events)
            event_id = f"evt_{event_index:08d}"
            suffix_target = int(role_param(config["suffix_len_tokens"], agent_id, rng))
            suffix_label = (
                f"suffix event={event_id} workflow={workflow_id} "
                f"agent={agent_id} step={state['step_id']} branch={branch_id}"
            )
            suffix_text, suffix_actual = make_text_to_length(
                tokenizer,
                suffix_target,
                suffix_label,
                seed + stable_int(event_id) % 1_000_000_000,
            )
            max_tokens = int(role_param(config["max_tokens_dist"], agent_id, rng))
            event = {
                "event_id": event_id,
                "timestamp_s": round(ts, 6),
                "workflow_id": workflow_id,
                "agent_id": agent_id,
                "prefix_id": prefix_id,
                "step_id": int(state["step_id"]),
                "branch_id": branch_id,
                "parent_event_id": parent_event_id,
                "prefix_len_tokens": int(prefix_row["actual_prefix_len_tokens"]),
                "suffix_len_tokens": suffix_actual,
                "max_tokens": max_tokens,
                "temperature": float(config.get("temperature", 0.0)),
                "top_p": float(config.get("top_p", 1.0)),
                "reactivation_gap_s": None
                if reactivation_gap is None
                else round(float(reactivation_gap), 6),
                "expected_prefix_reuse": expected_reuse,
                "is_branch_fanout": fanout > 1,
                "suffix_text": suffix_text,
            }
            events.append(event)
            branch_event_ids.append(event_id)

        last_prefix_seen[prefix_id] = ts
        if branch_event_ids:
            state["last_event_id"] = branch_event_ids[-1]
        state["step_id"] += 1
        state["graph_index"] += 1
        heapq.heappush(heap, (ts + choose_delay(config, rng), wf_idx))

    return events, []


def history_assistant_target(
    config: dict[str, Any], agent_id: str, max_tokens: int, rng: random.Random
) -> int:
    value = config.get("history_assistant_len_tokens", "max_tokens")
    if value == "max_tokens":
        return max_tokens
    sampled = role_param(value, agent_id, rng)
    if sampled == "max_tokens":
        return max_tokens
    return int(sampled)


def build_history_segment(
    config: dict[str, Any],
    tokenizer: TokenizerAdapter,
    *,
    event: dict[str, Any],
    seed: int,
    rng: random.Random,
) -> tuple[str, int]:
    agent_id = event["agent_id"]
    assistant_target = history_assistant_target(
        config, agent_id, int(event["max_tokens"]), rng
    )
    observation_target = int(
        role_param(config.get("history_observation_len_tokens", 256), agent_id, rng)
    )
    assistant_text, _ = make_text_to_length(
        tokenizer,
        assistant_target,
        (
            f"assistant event={event['event_id']} workflow={event['workflow_id']} "
            f"agent={agent_id} step={event['step_id']} branch={event['branch_id']}"
        ),
        seed + stable_int(f"{event['event_id']}/assistant") % 1_000_000_000,
    )
    observation_text, _ = make_text_to_length(
        tokenizer,
        observation_target,
        (
            f"tool observation event={event['event_id']} workflow={event['workflow_id']} "
            f"agent={agent_id} step={event['step_id']} branch={event['branch_id']}"
        ),
        seed + stable_int(f"{event['event_id']}/observation") % 1_000_000_000,
    )
    segment = "\n".join(
        [
            f"<assistant_output event_id={event['event_id']} agent={agent_id}>",
            assistant_text,
            "</assistant_output>",
            f"<tool_observation event_id={event['event_id']} agent={agent_id}>",
            observation_text,
            "</tool_observation>",
        ]
    )
    return segment, tokenizer.count(segment)


def cap_history_text(
    config: dict[str, Any],
    tokenizer: TokenizerAdapter,
    history_text: str,
    base_len_tokens: int,
) -> str:
    cap = config.get("history_max_prefix_len_tokens")
    if cap is None:
        return history_text
    allowed_history_tokens = max(0, int(cap) - base_len_tokens - 16)
    return tokenizer.tail_to_tokens(history_text, allowed_history_tokens)


def build_growing_prefix_text(base_text: str, history_text: str) -> str:
    if not history_text:
        return base_text
    return "\n".join(
        [
            base_text,
            "<conversation_history>",
            history_text,
            "</conversation_history>",
        ]
    )


def generate_growing_history_events(
    config: dict[str, Any], tokenizer: TokenizerAdapter, prefix_by_id: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seed = int(config["seed"])
    rng = random.Random(seed)
    graph = list(config["workflow_graph"])
    if not graph:
        raise ValueError("workflow_graph must not be empty")

    heap: list[tuple[float, int]] = []
    states: dict[int, dict[str, Any]] = {}
    mean_think = float(config.get("mean_think_time_s", 0.5))
    for wf_idx in range(int(config["num_workflows"])):
        first_ts = rng.random() * max(mean_think, 0.001)
        heapq.heappush(heap, (first_ts, wf_idx))
        states[wf_idx] = {
            "step_id": 0,
            "graph_index": 0,
            "last_event_id": None,
            "last_ts": None,
            "history_text": "",
        }

    events: list[dict[str, Any]] = []
    prompt_prefix_rows: list[dict[str, Any]] = []
    duration_s = float(config["duration_s"])
    max_events = config.get("max_events")
    max_events = int(max_events) if max_events is not None else None

    while heap:
        ts, wf_idx = heapq.heappop(heap)
        if ts > duration_s:
            break
        if max_events is not None and len(events) >= max_events:
            break

        state = states[wf_idx]
        workflow_id = f"wf_{wf_idx:05d}"
        agent_id = graph[state["graph_index"] % len(graph)]
        base_prefix_id = f"{workflow_id}/base"
        base_row = prefix_by_id[base_prefix_id]
        base_text = base_row["prefix_text"]
        base_len = int(base_row["actual_prefix_len_tokens"])
        history_text = cap_history_text(
            config, tokenizer, str(state["history_text"]), base_len
        )
        state["history_text"] = history_text
        prefix_text = build_growing_prefix_text(base_text, history_text)
        prefix_actual = tokenizer.count(prefix_text)
        prompt_prefix_id = f"{workflow_id}/history_step_{int(state['step_id']):05d}"
        prompt_prefix_rows.append(
            {
                "prefix_id": prompt_prefix_id,
                "workflow_id": workflow_id,
                "agent_id": agent_id,
                "base_prefix_id": base_prefix_id,
                "target_prefix_len_tokens": prefix_actual,
                "actual_prefix_len_tokens": prefix_actual,
                "history_len_tokens": max(0, prefix_actual - base_len),
                "prompt_mode": "growing_history",
                "prefix_text": prefix_text,
            }
        )

        prior_last_ts = state.get("last_ts")
        reactivation_gap = None if prior_last_ts is None else ts - float(prior_last_ts)
        expected_reuse = prior_last_ts is not None

        fanout = int(sample_from_param(config.get("branch_fanout", 1), rng))
        fanout = max(1, fanout)
        parent_event_id = state["last_event_id"]
        branch_event_ids: list[str] = []
        history_segments: list[str] = []
        for branch_id in range(fanout):
            if max_events is not None and len(events) >= max_events:
                break
            event_index = len(events)
            event_id = f"evt_{event_index:08d}"
            suffix_target = int(role_param(config["suffix_len_tokens"], agent_id, rng))
            suffix_label = (
                f"suffix event={event_id} workflow={workflow_id} "
                f"agent={agent_id} step={state['step_id']} branch={branch_id}"
            )
            suffix_text, suffix_actual = make_text_to_length(
                tokenizer,
                suffix_target,
                suffix_label,
                seed + stable_int(event_id) % 1_000_000_000,
            )
            max_tokens = int(role_param(config["max_tokens_dist"], agent_id, rng))
            event = {
                "event_id": event_id,
                "timestamp_s": round(ts, 6),
                "workflow_id": workflow_id,
                "agent_id": agent_id,
                "prefix_id": base_prefix_id,
                "prompt_prefix_id": prompt_prefix_id,
                "step_id": int(state["step_id"]),
                "branch_id": branch_id,
                "parent_event_id": parent_event_id,
                "base_prefix_len_tokens": base_len,
                "history_len_tokens": max(0, prefix_actual - base_len),
                "prefix_len_tokens": prefix_actual,
                "suffix_len_tokens": suffix_actual,
                "max_tokens": max_tokens,
                "temperature": float(config.get("temperature", 0.0)),
                "top_p": float(config.get("top_p", 1.0)),
                "reactivation_gap_s": None
                if reactivation_gap is None
                else round(float(reactivation_gap), 6),
                "expected_prefix_reuse": expected_reuse,
                "is_branch_fanout": fanout > 1,
                "prompt_mode": "growing_history",
                "suffix_text": suffix_text,
            }
            events.append(event)
            branch_event_ids.append(event_id)
            segment, _ = build_history_segment(
                config, tokenizer, event=event, seed=seed, rng=rng
            )
            history_segments.append(segment)

        if history_segments:
            state["history_text"] = "\n".join(
                [part for part in [state["history_text"], *history_segments] if part]
            )
            state["last_event_id"] = branch_event_ids[-1]
        state["last_ts"] = ts
        state["step_id"] += 1
        state["graph_index"] += 1
        heapq.heappush(heap, (ts + choose_delay(config, rng), wf_idx))

    return events, prompt_prefix_rows


def generate_events(
    config: dict[str, Any], tokenizer: TokenizerAdapter, prefix_by_id: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    mode = prompt_mode(config)
    if mode == "static_prefix":
        return generate_static_prefix_events(config, tokenizer, prefix_by_id)
    if mode == "growing_history":
        return generate_growing_history_events(config, tokenizer, prefix_by_id)
    raise ValueError(f"Unknown prompt_mode {mode!r}. Expected 'static_prefix' or 'growing_history'.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate configurable agentic workload traces.")
    parser.add_argument("--config", type=str, default=None, help="JSON/YAML config path.")
    parser.add_argument("--out-dir", type=str, required=True, help="Output run directory.")
    args = parser.parse_args()

    config = load_config(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = TokenizerAdapter(
        config.get("tokenizer_name"),
        trust_remote_code=bool(config.get("tokenizer_trust_remote_code", False)),
        local_files_only=bool(config.get("tokenizer_local_files_only", False)),
    )
    config["tokenizer_kind"] = tokenizer.kind

    prefix_bank = build_prefix_bank(config, tokenizer)
    prefix_by_id = {row["prefix_id"]: row for row in prefix_bank}
    events, generated_prefix_rows = generate_events(config, tokenizer, prefix_by_id)
    prefix_bank.extend(generated_prefix_rows)

    write_jsonl(out_dir / "prefix_bank.jsonl", prefix_bank)
    write_jsonl(out_dir / "trace.jsonl", events)
    dump_json(out_dir / "config_resolved.json", config)
    dump_json(
        out_dir / "manifest.json",
        {
            "generator": "agentic_workload_generator.generate_trace",
            "num_prefixes": len(prefix_bank),
            "num_events": len(events),
            "tokenizer_kind": tokenizer.kind,
            "config_path": args.config,
        },
    )
    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "num_prefixes": len(prefix_bank),
                "num_events": len(events),
                "tokenizer_kind": tokenizer.kind,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
