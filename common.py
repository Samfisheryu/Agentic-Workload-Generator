from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable


DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 7,
    "tokenizer_name": None,
    "tokenizer_trust_remote_code": False,
    "tokenizer_local_files_only": False,
    "num_workflows": 4,
    "duration_s": 30.0,
    "max_events": 200,
    "agents": ["planner", "toolcaller", "executor", "verifier"],
    "workflow_graph": [
        "planner",
        "toolcaller",
        "planner",
        "executor",
        "verifier",
        "planner",
    ],
    "prefix_len_tokens": 1024,
    "suffix_len_tokens": [128],
    "mean_think_time_s": 0.5,
    "think_time_jitter_s": 0.1,
    "burst_probability": 0.0,
    "burst_delay_multiplier": 0.1,
    "branch_fanout": 1,
    "max_tokens_dist": {"16": 0.5, "32": 0.5},
    "prompt_mode": "static_prefix",
    "history_assistant_len_tokens": "max_tokens",
    "history_observation_len_tokens": 256,
    "history_max_prefix_len_tokens": None,
    "temperature": 0.0,
    "top_p": 1.0,
}


class TokenizerAdapter:
    def __init__(
        self,
        tokenizer_name: str | None,
        trust_remote_code: bool = False,
        local_files_only: bool = False,
    ) -> None:
        self.tokenizer_name = tokenizer_name
        self.kind = "simple"
        self.tokenizer = None
        if tokenizer_name:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                trust_remote_code=trust_remote_code,
                local_files_only=local_files_only,
            )
            self.kind = "transformers"

    def encode(self, text: str) -> list[int] | list[str]:
        if self.tokenizer is None:
            return text.split()
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, ids: list[int] | list[str]) -> str:
        if self.tokenizer is None:
            return " ".join(str(x) for x in ids)
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def count(self, text: str) -> int:
        return len(self.encode(text))

    def trim_to_tokens(self, text: str, target_tokens: int) -> str:
        if target_tokens <= 0:
            return ""
        ids = self.encode(text)
        if len(ids) <= target_tokens:
            return text
        return self.decode(ids[:target_tokens])

    def tail_to_tokens(self, text: str, target_tokens: int) -> str:
        if target_tokens <= 0:
            return ""
        ids = self.encode(text)
        if len(ids) <= target_tokens:
            return text
        return self.decode(ids[-target_tokens:])


def stable_int(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def load_config(path: str | Path | None) -> dict[str, Any]:
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    if path is None:
        return cfg
    p = Path(path)
    raw = p.read_text()
    if p.suffix in {".yaml", ".yml"}:
        import yaml

        user_cfg = yaml.safe_load(raw) or {}
    else:
        user_cfg = json.loads(raw)
    deep_update(cfg, user_cfg)
    return cfg


def deep_update(base: dict[str, Any], update: dict[str, Any]) -> None:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def dump_json(path: str | Path, obj: Any) -> None:
    Path(path).write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sample_from_param(value: Any, rng: random.Random) -> Any:
    if isinstance(value, list):
        return rng.choice(value)
    if isinstance(value, dict):
        items = list(value.items())
        total = sum(float(v) for _, v in items)
        pick = rng.random() * total
        acc = 0.0
        for k, weight in items:
            acc += float(weight)
            if pick <= acc:
                return parse_scalar(k)
        return parse_scalar(items[-1][0])
    return value


def parse_scalar(value: Any) -> Any:
    if isinstance(value, str):
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            return value
    return value


def role_param(config_value: Any, role: str, rng: random.Random) -> Any:
    if isinstance(config_value, dict):
        if role in config_value or "default" in config_value:
            value = config_value.get(role, config_value.get("default"))
            if value is None:
                raise KeyError(f"No config value for role {role!r} and no default.")
            return sample_from_param(value, rng)
        return sample_from_param(config_value, rng)
    return sample_from_param(config_value, rng)


def choose_delay(config: dict[str, Any], rng: random.Random) -> float:
    mean = float(config.get("mean_think_time_s", 0.5))
    jitter = float(config.get("think_time_jitter_s", 0.0))
    delay = mean + rng.uniform(-jitter, jitter)
    delay = max(0.0, delay)
    if rng.random() < float(config.get("burst_probability", 0.0)):
        delay *= float(config.get("burst_delay_multiplier", 0.1))
    return delay


def make_text_to_length(
    tokenizer: TokenizerAdapter,
    target_tokens: int,
    label: str,
    seed: int,
) -> tuple[str, int]:
    rng = random.Random(seed)
    chunks: list[str] = []
    i = 0
    approx_tokens = 0
    target_with_margin = max(target_tokens + 16, 1)
    while approx_tokens < target_with_margin:
        topic = rng.choice(
            [
                "planning",
                "tool output",
                "verification",
                "state summary",
                "memory",
                "execution",
            ]
        )
        chunks.append(
            " ".join(
                [
                    f"{label}",
                    f"chunk_{i:05d}.",
                    f"topic={topic}.",
                    "This deterministic synthetic block is repeated to create stable token length.",
                    "It is intended for serving-system profiling rather than semantic task quality.",
                    f"marker_{seed % 100000}_{i:05d}.",
                ]
            )
        )
        approx_tokens += tokenizer.count(chunks[-1])
        i += 1
    text = tokenizer.trim_to_tokens("\n".join(chunks), target_tokens)
    actual = tokenizer.count(text)
    if actual < target_tokens:
        filler = " ".join(f"fill_{j}" for j in range(target_tokens - actual + 8))
        text = tokenizer.trim_to_tokens(text + " " + filler, target_tokens)
        actual = tokenizer.count(text)
    return text, actual


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def stats(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values) if values else None,
    }
