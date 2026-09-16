"""Incremental, crash-safe result writing. Flushed to disk after every
checkpoint, so a crash loses at most one in-progress checkpoint - see
PLAN.md Sec 6/7 for the file layout this implements.
"""

import csv
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

from hnsw_bench.config import ExperimentConfig

MEASUREMENT_FIELDS = [
    "repetition",
    "round",
    "phase",
    "recall_at_10",
    "latency_p50_ms",
    "latency_p95_ms",
    "latency_mean_ms",
    "table_size_bytes",
    "index_size_bytes",
    "dead_tuples",
    "delete_s",
    "insert_s",
    "vacuum_s",
    "reindex_s",
]
QUERY_FIELDS = [
    "repetition",
    "round",
    "phase",
    "query_index",
    "latency_ms",
    "true_dist_k",
    "returned_dists",
]


class ResultsWriter:
    def __init__(self, run_dir: Path, config: ExperimentConfig, metadata: dict) -> None:
        self.run_dir = run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.yaml").write_text(yaml.safe_dump(asdict(config)))
        (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

        self._measurements = open(run_dir / "measurements.csv", "a", newline="")
        self._measurements_writer = csv.DictWriter(
            self._measurements, fieldnames=MEASUREMENT_FIELDS, restval=""
        )
        if self._measurements.tell() == 0:
            self._measurements_writer.writeheader()

        self._queries = open(run_dir / "queries.csv", "a", newline="")
        self._queries_writer = csv.DictWriter(
            self._queries, fieldnames=QUERY_FIELDS, restval=""
        )
        if self._queries.tell() == 0:
            self._queries_writer.writeheader()

        self._status: dict = {"completed_repetitions": []}
        self._write_status()

    def write_measurement(self, row: dict) -> None:
        self._measurements_writer.writerow(row)
        self._measurements.flush()
        os.fsync(self._measurements.fileno())

    def write_queries(self, rows: list[dict]) -> None:
        self._queries_writer.writerows(rows)
        self._queries.flush()
        os.fsync(self._queries.fileno())

    def update_status(self, **fields) -> None:
        self._status.update(fields)
        self._write_status()

    def mark_repetition_complete(self, repetition: int) -> None:
        self._status["completed_repetitions"].append(repetition)
        self._status["current_repetition"] = None
        self._status["current_phase"] = None
        self._write_status()

    def _write_status(self) -> None:
        self._status["last_updated"] = datetime.now(timezone.utc).isoformat()
        (self.run_dir / "status.json").write_text(json.dumps(self._status, indent=2))

    def close(self) -> None:
        self._measurements.close()
        self._queries.close()
