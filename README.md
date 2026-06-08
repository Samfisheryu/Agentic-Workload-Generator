# Agentic Workload Generator

This repository contains a trace-driven workload generator for simulating configurable agentic workloads. It is meant for serving-system experiments and profiling, where you want controlled agent behavior instead of noise from a full real agent framework.

The generator is CPU-side. It creates controlled request traces with stable long prefixes, dynamic suffixes, workflow re-entry, branch fanout, growing agent history, and short output steps. A real serving backend is responsible for prefill, decode, KV cache, offload, scheduling, and hardware behavior.

The replay client targets OpenAI-compatible completion/chat APIs, so the core generator is not tied to one backend. The bundled LMCache script is an example profiling harness for `vLLM + LMCache`.

## Prompt Modes

`prompt_mode: "static_prefix"` keeps the original shape:

```text
request N = stable workflow/agent prefix + per-step suffix
```

`prompt_mode: "growing_history"` simulates a linear agent transcript:

```text
step 1 = base prefix + suffix1
step 2 = base prefix + synthetic output1 + synthetic observation1 + suffix2
step 3 = base prefix + synthetic output1 + observation1 + output2 + observation2 + suffix3
```

Use `history_assistant_len_tokens`, `history_observation_len_tokens`, and `history_max_prefix_len_tokens` to control how quickly the prompt grows and whether old history is truncated to fit the model context.

## Files

```text
generate_trace.py     Generate prefix_bank.jsonl and trace.jsonl.
analyze_trace.py      Analyze trace-only properties without running a model.
replay_trace.py       Replay a fixed trace against vLLM OpenAI-compatible API.
configs/              Example workload configs.
common.py             Shared config, tokenizer, JSONL, and sampling helpers.
```

## Generate A Sanity Trace

```bash
python generate_trace.py \
  --config configs/sanity_small.json \
  --out-dir /tmp/agentic_sanity
```

Outputs:

```text
/tmp/agentic_sanity/
  config_resolved.json
  manifest.json
  prefix_bank.jsonl
  trace.jsonl
```

## Analyze The Trace

```bash
python analyze_trace.py \
  --trace /tmp/agentic_sanity/trace.jsonl
```

Outputs:

```text
/tmp/agentic_sanity/
  trace_summary.json
  prefix_reuse_counts.csv
```

## Replay Against vLLM

Start vLLM separately, then run:

```bash
python replay_trace.py \
  --trace /tmp/agentic_sanity/trace.jsonl \
  --prefix-bank /tmp/agentic_sanity/prefix_bank.jsonl \
  --results /tmp/agentic_sanity/client_results.jsonl \
  --base-url http://localhost:8000/v1 \
  --model Qwen/Qwen3-8B \
  --mode open-loop \
  --endpoint completions
```

By default, replay uses streaming responses so it can record `ttft_s`. Use `--no-stream` if the backend does not support streaming usage.

The bundled LMCache run script defaults to `REPLAY_MODE=closed-loop`, which is closer to a real agent: each workflow waits for the previous step to finish before issuing the next step. Override with `REPLAY_MODE=open-loop` when you want a fixed-arrival-rate backend pressure test.

## LMCache CPU/Disk Harness

`run_lmcache_offload_pressure.sh` starts `vLLM + LMCache`, records metrics, and plots the run. Its default LMCache storage hierarchy is:

```text
CPU tier:  100 GB
Disk tier: 500 GB
Disk path: /data1/lmcache_kv/${RUN_ID}/gpu0,/data1/lmcache_kv/${RUN_ID}/gpu1
```

Override with environment variables when needed:

```bash
LMCACHE_CPU_SIZE_GB=64 \
LMCACHE_DISK_SIZE_GB=250 \
LMCACHE_DISK_PATH=/data1/lmcache_kv/shared \
./run_lmcache_offload_pressure.sh
```

The script fails early if the disk path is not writable.

## Tokenizer Control

Set `tokenizer_name` in the config to the target model tokenizer when real token length matters:

```json
{
  "tokenizer_name": "Qwen/Qwen3-8B",
  "tokenizer_local_files_only": true
}
```

If `tokenizer_name` is null, the generator uses a whitespace fallback tokenizer. That is useful for fast local structure checks but should not be used for final profiling.

## Current Scope

Implemented:

- deterministic prefix bank
- deterministic trace generation
- workflow re-entry
- growing-history prompt mode
- branch fanout support
- trace summary
- open-loop replay
- simple closed-loop replay
- streaming TTFT capture
- vLLM metrics scraper
- LMCache metrics scraper
- hardware monitor
- plotting

Not implemented yet:

- calibration against real agent traces and frameworks
