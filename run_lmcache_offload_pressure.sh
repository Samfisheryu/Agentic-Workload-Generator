#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
GENERATOR_DIR="${GENERATOR_DIR:-${SCRIPT_DIR}}"
VLLM_DIR="${VLLM_DIR:-${ROOT_DIR}/HF_Prometheus}"
CONFIG="${CONFIG:-${GENERATOR_DIR}/configs/pressure_4090_offload.json}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
RUN_ID="${1:-lmcache_tp2_offload_pressure_$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="${RUN_DIR:-${GENERATOR_DIR}/results/${RUN_ID}}"
LOG_DIR="${RUN_DIR}/logs"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-16384}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.80}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-64}"
VLLM_DISABLE_LOG_STATS="${VLLM_DISABLE_LOG_STATS:-1}"
LMCACHE_CPU_SIZE_GB="${LMCACHE_CPU_SIZE_GB:-100}"
LMCACHE_DISK_SIZE_GB="${LMCACHE_DISK_SIZE_GB:-500}"
LMCACHE_DISK_PATH="${LMCACHE_DISK_PATH:-/data1/lmcache_kv/${RUN_ID}/gpu0,/data1/lmcache_kv/${RUN_ID}/gpu1}"
LMCACHE_DISK_PATH_SHARDING="${LMCACHE_DISK_PATH_SHARDING:-by_gpu}"
LMCACHE_CONFIG_FILE="${LMCACHE_CONFIG_FILE:-${RUN_DIR}/lmcache_config.yaml}"
VLLM_KV_OFFLOAD_GB="${VLLM_KV_OFFLOAD_GB:-${LMCACHE_CPU_SIZE_GB}}"
REPLAY_MAX_WORKERS="${REPLAY_MAX_WORKERS:-512}"
REPLAY_TIMEOUT_S="${REPLAY_TIMEOUT_S:-2400}"
REPLAY_MODE="${REPLAY_MODE:-closed-loop}"

mkdir -p "${LOG_DIR}"
cd "${ROOT_DIR}"

source /home/nengneng/miniconda3/etc/profile.d/conda.sh
set +u
conda activate agent-dmi
set -u

export PYTHONHASHSEED=0
export CUDA_VISIBLE_DEVICES=0,1
export PROMETHEUS_MULTIPROC_DIR="/tmp/lmcache_prometheus_${RUN_ID}"
export LMCACHE_INTERNAL_API_SERVER_ENABLED=true
export LMCACHE_INTERNAL_API_SERVER_PORT_START=6999
export LMCACHE_INTERNAL_API_SERVER_HOST=127.0.0.1
export LMCACHE_MAX_LOCAL_CPU_SIZE="${LMCACHE_CPU_SIZE_GB}"
export LMCACHE_LOCAL_CPU=true
export LMCACHE_LOCAL_DISK="${LMCACHE_DISK_PATH}"
export LMCACHE_LOCAL_DISK_PATH_SHARDING="${LMCACHE_DISK_PATH_SHARDING}"
export LMCACHE_MAX_LOCAL_DISK_SIZE="${LMCACHE_DISK_SIZE_GB}"
export LMCACHE_CONFIG_FILE

IFS=',' read -ra LMCACHE_DISK_PATHS <<< "${LMCACHE_DISK_PATH}"
for disk_path in "${LMCACHE_DISK_PATHS[@]}"; do
  if ! mkdir -p "${disk_path}" 2>/dev/null; then
    echo "[run] failed to create LMCache disk path: ${disk_path}" >&2
    echo "[run] please remount /data1 read-write or set LMCACHE_DISK_PATH to a writable path." >&2
    exit 1
  fi
done

cat > "${LMCACHE_CONFIG_FILE}" <<EOF
chunk_size: 256
local_cpu: true
max_local_cpu_size: ${LMCACHE_CPU_SIZE_GB}
local_disk: ${LMCACHE_DISK_PATH}
local_disk_path_sharding: ${LMCACHE_DISK_PATH_SHARDING}
max_local_disk_size: ${LMCACHE_DISK_SIZE_GB}
remote_url: null
remote_serde: naive
internal_api_server_enabled: true
internal_api_server_port_start: 6999
internal_api_server_host: 127.0.0.1
EOF

rm -rf "${PROMETHEUS_MULTIPROC_DIR}"
mkdir -p "${PROMETHEUS_MULTIPROC_DIR}"

echo "[run] run_id=${RUN_ID}"
echo "[run] run_dir=${RUN_DIR}"
echo "[run] lmcache_config=${LMCACHE_CONFIG_FILE}"
echo "[run] lmcache_cpu_gb=${LMCACHE_CPU_SIZE_GB}"
echo "[run] lmcache_disk_gb=${LMCACHE_DISK_SIZE_GB}"
echo "[run] lmcache_disk_path=${LMCACHE_DISK_PATH}"
echo "[run] generating trace"
python "${GENERATOR_DIR}/generate_trace.py" \
  --config "${CONFIG}" \
  --out-dir "${RUN_DIR}" | tee "${LOG_DIR}/generate_trace.log"
cp "${CONFIG}" "${RUN_DIR}/config.json"

python "${GENERATOR_DIR}/analyze_trace.py" \
  --trace "${RUN_DIR}/trace.jsonl" \
  --working-set-window-s 30 \
  2>&1 | tee "${LOG_DIR}/analyze_trace.log"

python "${GENERATOR_DIR}/monitoring/init_run.py" \
  --mode agentic-lmcache-offload-pressure-tp2 \
  --dataset "${RUN_DIR}/trace.jsonl" \
  --model "${MODEL}" \
  --vllm-url http://127.0.0.1:8000/v1 \
  --out "${RUN_DIR}" \
  --run-id "${RUN_ID}" \
  --max-tokens 512 \
  --temperature 0.0 \
  --notes "TP2 LMCache offload pressure: CPU ${LMCACHE_CPU_SIZE_GB}GB, disk ${LMCACHE_DISK_SIZE_GB}GB at ${LMCACHE_DISK_PATH}, ignore_eos, internal LMCache metrics enabled" \
  2>&1 | tee "${LOG_DIR}/init_run.log"

VLLM_PID=""
VLLM_WATCHER_PID=""
MONITOR_PIDS=()

stop_monitors() {
  set +e
  for pid in "${MONITOR_PIDS[@]:-}"; do
    if kill -0 "${pid}" >/dev/null 2>&1; then
      kill -TERM "${pid}" >/dev/null 2>&1
    fi
  done
}

wait_monitors() {
  set +e
  for pid in "${MONITOR_PIDS[@]:-}"; do
    wait "${pid}" >/dev/null 2>&1 || true
  done
  MONITOR_PIDS=()
}

cleanup() {
  set +e
  if [[ -n "${VLLM_WATCHER_PID}" ]] && kill -0 "${VLLM_WATCHER_PID}" >/dev/null 2>&1; then
    kill -TERM "${VLLM_WATCHER_PID}" >/dev/null 2>&1
    wait "${VLLM_WATCHER_PID}" >/dev/null 2>&1 || true
  fi
  stop_monitors
  wait_monitors
  if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" >/dev/null 2>&1; then
    kill -INT "${VLLM_PID}" >/dev/null 2>&1
    for _ in $(seq 1 60); do
      kill -0 "${VLLM_PID}" >/dev/null 2>&1 || break
      sleep 1
    done
    kill -0 "${VLLM_PID}" >/dev/null 2>&1 && kill "${VLLM_PID}" >/dev/null 2>&1
  fi
}
trap cleanup EXIT

echo "[run] starting vLLM + LMCache"
VLLM_ARGS=(
  serve "${MODEL}"
  --tensor-parallel-size 2
  --max-model-len "${VLLM_MAX_MODEL_LEN}"
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}"
  --max-num-seqs "${VLLM_MAX_NUM_SEQS}"
  --kv-offloading-size "${VLLM_KV_OFFLOAD_GB}"
  --kv-offloading-backend lmcache
  --disable-hybrid-kv-cache-manager
  --port 8000
)
if [[ "${VLLM_DISABLE_LOG_STATS}" == "1" ]]; then
  VLLM_ARGS+=(--disable-log-stats)
fi
(
  cd "${VLLM_DIR}"
  vllm "${VLLM_ARGS[@]}"
) >"${LOG_DIR}/vllm.log" 2>&1 &
VLLM_PID=$!
echo "${VLLM_PID}" > "${RUN_DIR}/vllm.pid"

echo "[run] waiting for vLLM health"
for i in $(seq 1 180); do
  if curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then
    echo "[run] vLLM ready after ${i} checks"
    break
  fi
  if ! kill -0 "${VLLM_PID}" >/dev/null 2>&1; then
    echo "[run] vLLM exited before ready; tailing log"
    tail -200 "${LOG_DIR}/vllm.log"
    exit 1
  fi
  if [[ "${i}" == "180" ]]; then
    echo "[run] vLLM did not become healthy"
    tail -200 "${LOG_DIR}/vllm.log"
    exit 1
  fi
  sleep 5
done

echo "[run] starting monitors"
python "${GENERATOR_DIR}/monitoring/monitor_vllm_metrics.py" \
  --manifest "${RUN_DIR}" \
  --out "${RUN_DIR}/vllm_metrics.jsonl" \
  --interval 1 \
  >"${LOG_DIR}/monitor_vllm_metrics.log" 2>&1 &
MONITOR_PIDS+=("$!")

python "${GENERATOR_DIR}/monitoring/monitor_lmcache_metrics.py" \
  --manifest "${RUN_DIR}" \
  --out "${RUN_DIR}/lmcache_metrics.jsonl" \
  --interval 1 \
  --url http://127.0.0.1:6999/metrics \
  --url http://127.0.0.1:7000/metrics \
  --url http://127.0.0.1:7001/metrics \
  >"${LOG_DIR}/monitor_lmcache_metrics.log" 2>&1 &
MONITOR_PIDS+=("$!")

python "${GENERATOR_DIR}/monitoring/monitor_gpu_nvml.py" \
  --manifest "${RUN_DIR}" \
  --out "${RUN_DIR}/gpu.jsonl" \
  --interval 1 \
  >"${LOG_DIR}/monitor_gpu.log" 2>&1 &
MONITOR_PIDS+=("$!")

python "${GENERATOR_DIR}/monitoring/monitor_pcie_nvml.py" \
  --manifest "${RUN_DIR}" \
  --out "${RUN_DIR}/pcie.jsonl" \
  --interval 0.25 \
  >"${LOG_DIR}/monitor_pcie.log" 2>&1 &
MONITOR_PIDS+=("$!")

python "${GENERATOR_DIR}/monitoring/monitor_cpu_proc.py" \
  --manifest "${RUN_DIR}" \
  --out "${RUN_DIR}/cpu.jsonl" \
  --interval 1 \
  --match "vllm serve" \
  >"${LOG_DIR}/monitor_cpu.log" 2>&1 &
MONITOR_PIDS+=("$!")

(
  while [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" >/dev/null 2>&1; do
    sleep 2
  done
  echo "[run] vLLM exited; stopping monitors" >> "${LOG_DIR}/monitor_watchdog.log"
  for pid in "${MONITOR_PIDS[@]:-}"; do
    kill -TERM "${pid}" >/dev/null 2>&1 || true
  done
) &
VLLM_WATCHER_PID=$!

echo "[run] replaying trace"
python "${GENERATOR_DIR}/replay_trace.py" \
  --trace "${RUN_DIR}/trace.jsonl" \
  --prefix-bank "${RUN_DIR}/prefix_bank.jsonl" \
  --results "${RUN_DIR}/client_results.jsonl" \
  --base-url http://127.0.0.1:8000/v1 \
  --model "${MODEL}" \
  --mode "${REPLAY_MODE}" \
  --endpoint completions \
  --max-workers "${REPLAY_MAX_WORKERS}" \
  --timeout-s "${REPLAY_TIMEOUT_S}" \
  --ignore-eos \
  2>&1 | tee "${LOG_DIR}/replay_trace.log"

echo "[run] stopping monitors"
stop_monitors
wait_monitors

echo "[run] plotting"
conda deactivate || true
MPLCONFIGDIR=/tmp/matplotlib python \
  "${GENERATOR_DIR}/monitoring/plot_workload_run.py" \
  --run-dir "${RUN_DIR}" \
  2>&1 | tee "${LOG_DIR}/plot.log"

echo "[run] completed ${RUN_DIR}"
