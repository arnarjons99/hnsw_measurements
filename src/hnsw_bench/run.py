"""CLI entrypoint for a full experiment run - one config, `repetitions`
repetitions, results under results/<run_id>/. See PLAN.md Sec 4/7/8.

Run: .venv/bin/python -m hnsw_bench.run configs/experiments/sift1m_baseline.yaml
"""

import argparse
import platform
import socket
import subprocess
from datetime import datetime
from pathlib import Path

from hnsw_bench.config import load_config
from hnsw_bench.datasets import partition_glove, partition_sift1m
from hnsw_bench.db import connect
from hnsw_bench.experiment import run_repetition
from hnsw_bench.results_writer import ResultsWriter

PARTITIONERS = {"sift1m": partition_sift1m, "glove": partition_glove}
DATA_PATHS = {
    "sift1m": Path("data/sift1m"),
    "glove": Path(
        "data/glove/wiki_giga_2024_300_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05_combined.txt"
    ),
}


def collect_metadata() -> dict:
    conn = connect()
    cur = conn.cursor()

    cur.execute("SELECT version()")
    row = cur.fetchone()
    if row is None:
        raise RuntimeError("Could not get PostgreSQL version")
    postgres_version = row[0]

    cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
    row = cur.fetchone()
    if row is None:
        raise RuntimeError("pgvector extension is not installed")
    pgvector_version = row[0]

    conn.close()

    git_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    return {
        "git_sha": git_sha,
        "hostname": socket.gethostname(),
        "python_version": platform.python_version(),
        "postgres_version": postgres_version,
        "pgvector_version": pgvector_version,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=Path)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    # loading the config first so we can use its dataset to load the partitioner
    config = load_config(args.config_path)
    run_id = (
        args.run_id
        or f"{config.dataset}_{config.maintenance}_{datetime.now():%Y%m%dT%H%M%S}"
    )
    run_dir = Path("results") / run_id

    # loading the actual datasets
    print(
        f"Loading {config.dataset} (active_size={config.active_size}, seed={config.seed})..."
    )
    partition = PARTITIONERS[config.dataset](
        DATA_PATHS[config.dataset], active_size=config.active_size, seed=config.seed
    )

    # running the experiment repeatedly, writing results to disk as we go
    print(f"Running experiment with {config.repetitions} repetitions...")
    writer = ResultsWriter(run_dir, config, collect_metadata())
    for repetition in range(config.repetitions):
        if repetition in writer.completed_repetitions:
            print(f"\n=== repetition {repetition + 1}/{config.repetitions} already complete, skipping ===")
            continue
        print(f"\n=== repetition {repetition + 1}/{config.repetitions} ===")
        run_repetition(config, partition, writer, repetition)
    writer.close()

    print(f"\ndone - results in {run_dir}")


if __name__ == "__main__":
    main()
