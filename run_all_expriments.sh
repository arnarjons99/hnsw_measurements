#!/usr/bin/env bash

set -uo pipefail

COMPOSE_FILE="docker/docker-compose.yml"
CONFIG_DIR="configs/experiments"
PYTHON=".venv/bin/python"
export PYTHONUNBUFFERED=1

CONFIGS=(
    "glove_baseline.yaml"
    "glove_vacuum.yaml"
    "glove_reindex.yaml"
    "glove_reindex_vacuum.yaml"
    "sift1m_baseline.yaml"
    "sift1m_vacuum.yaml"
    "sift1m_reindex.yaml"
    "sift1m_reindex_vacuum.yaml"
)

for config in "${CONFIGS[@]}"; do
    config_path="${CONFIG_DIR}/${config}"

    # e.g. glove_baseline.yaml -> glove_baseline
    run_id="run0_${config%.yaml}"

    echo
    echo "============================================================"
    echo "Starting experiment: ${run_id}"
    echo "Config: ${config_path}"
    echo "============================================================"

    echo "[1/4] Removing previous PostgreSQL container and volume..."
    docker compose -f "$COMPOSE_FILE" down --volumes

    echo "[2/4] Starting fresh PostgreSQL and waiting for healthcheck..."
    docker compose -f "$COMPOSE_FILE" up -d --wait

    echo "[3/4] PostgreSQL is healthy. Waiting another 20 seconds..."
    sleep 20

    echo "[4/4] Running ${run_id}..."
    "$PYTHON" -m hnsw_bench.run \
        "$config_path" \
        --run-id "$run_id"

    echo
    echo "Finished: ${run_id}"
done

echo
echo "============================================================"
echo "All experiments completed successfully."
echo "============================================================"
