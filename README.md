# HiddenCache Workload Generator

This directory contains a trace-driven workload generator for profiling agentic workloads on `vLLM + LMCache`.

The generator is CPU-side. It creates controlled request traces with stable long prefixes, dynamic suffixes, workflow re-entry, branch fanout, growing agent history, and short output steps. The real serving backend is responsible for prefill, decode, KV cache, LMCache load/offload, and scheduling behavior.

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
  --out-dir /tmp/hc_sanity
```

Outputs:

```text
/tmp/hc_sanity/
  config_resolved.json
  manifest.json
  prefix_bank.jsonl
  trace.jsonl
```

## Analyze The Trace

```bash
python analyze_trace.py \
  --trace /tmp/hc_sanity/trace.jsonl
```

Outputs:

```text
/tmp/hc_sanity/
  trace_summary.json
  prefix_reuse_counts.csv
```

## Replay Against vLLM

Start vLLM separately, then run:

```bash
python replay_trace.py \
  --trace /tmp/hc_sanity/trace.jsonl \
  --prefix-bank /tmp/hc_sanity/prefix_bank.jsonl \
  --results /tmp/hc_sanity/client_results.jsonl \
  --base-url http://localhost:8000/v1 \
  --model Qwen/Qwen3-8B \
  --mode open-loop \
  --endpoint completions
```

By default, replay uses streaming responses so it can record `ttft_s`. Use `--no-stream` if the backend does not support streaming usage.

The bundled LMCache run script defaults to `REPLAY_MODE=closed-loop`, which is closer to a real agent: each workflow waits for the previous step to finish before issuing the next step. Override with `REPLAY_MODE=open-loop` when you want a fixed-arrival-rate backend pressure test.

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

- mini-swe-agent calibration
