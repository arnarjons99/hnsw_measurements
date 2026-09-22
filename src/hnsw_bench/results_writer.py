"""Incremental, crash-safe result writing. Flushed to disk after every
checkpoint, so a crash loses at most one in-progress checkpoint - see
PLAN.md Sec 6/7 for the file layout this implements.
"""

import csv
import hashlib
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

from hnsw_bench.config import ExperimentConfig

# Timestamps are UTC ISO-8601. Every *_s duration has a matching *_start, so
# an operation can be placed on a wall clock and not just measured.
MEASUREMENT_FIELDS = [
    "repetition",
    "round",
    "phase",
    "checkpoint_start",
    "checkpoint_end",
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
    "warmup_start",
    "warmup_end",
    "measured_start",
    "measured_end",
    "delete_start",
    "insert_start",
    "vacuum_start",
    "reindex_start",
]
# t_offset_ms is milliseconds from this checkpoint's measured_start.
QUERY_FIELDS = [
    "repetition",
    "round",
    "phase",
    "query_index",
    "t_offset_ms",
    "latency_ms",
    "true_dist_k",
    "returned_dists",
]


class ResultsWriter:
    def __init__(self, run_dir: Path, config: ExperimentConfig, metadata: dict) -> None:
        self.run_dir = run_dir
        run_dir.mkdir(parents=True, exist_ok=True)

        config_hash = hashlib.sha256(
            json.dumps(asdict(config), sort_keys=True).encode()
        ).hexdigest()
        status_path = run_dir / "status.json"
        if status_path.exists():
            self._status: dict = json.loads(status_path.read_text())
            existing_hash = self._status.get("config_hash")
            if existing_hash is not None and existing_hash != config_hash:
                raise ValueError(
                    f"{run_dir} already has results from a different config "
                    "(params changed since this run-id was last used) - use a "
                    "different --run-id or delete the directory to start fresh"
                )
        else:
            self._status = {"completed_repetitions": []}
        self._status["config_hash"] = config_hash

        (run_dir / "config.yaml").write_text(yaml.safe_dump(asdict(config)))
        (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

        self._measurements = self._open_pruned(
            run_dir / "measurements.csv", MEASUREMENT_FIELDS
        )
        self._measurements_writer = csv.DictWriter(
            self._measurements, fieldnames=MEASUREMENT_FIELDS, restval=""
        )
        if self._measurements.tell() == 0:
            self._measurements_writer.writeheader()

        self._queries = self._open_pruned(run_dir / "queries.csv", QUERY_FIELDS)
        self._queries_writer = csv.DictWriter(
            self._queries, fieldnames=QUERY_FIELDS, restval=""
        )
        if self._queries.tell() == 0:
            self._queries_writer.writeheader()

        self._write_status()

    @property
    def completed_repetitions(self) -> set[int]:
        return set(self._status["completed_repetitions"])

    def _open_pruned(self, path: Path, fields: list[str]):
        """On resume, drop rows from a repetition that crashed mid-way so its
        redo isn't mixed with stale partial rows - PLAN.md Sec 7 resume policy."""
        if path.exists():
            with open(path, newline="") as f:
                kept = [
                    row
                    for row in csv.DictReader(f)
                    if int(row["repetition"]) in self.completed_repetitions
                ]
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields, restval="")
                writer.writeheader()
                writer.writerows(kept)
        return open(path, "a", newline="")

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
