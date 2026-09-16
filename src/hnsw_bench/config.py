"""Experiment config: one YAML file per run, loaded into a validated
dataclass. See PLAN.md Sec 5/10 - deliberately not pydantic."""

from dataclasses import dataclass
from pathlib import Path

import yaml

DATASETS = {"sift1m", "glove"}
DISTANCE_METRICS = {"l2", "cosine"}
MAINTENANCE_CONDITIONS = {"none", "vacuum", "reindex", "reindex_vacuum"}


@dataclass
class ExperimentConfig:
    dataset: str
    distance_metric: str
    seed: int
    active_size: int
    churn_pct: float
    rounds: int
    repetitions: int
    maintenance: str
    m: int
    ef_construction: int
    ef_search: int
    k: int
    n_recall_queries: int
    n_latency_queries: int

    def __post_init__(self) -> None:
        assert self.dataset in DATASETS, f"unknown dataset: {self.dataset!r}"
        assert self.distance_metric in DISTANCE_METRICS, f"unknown distance_metric: {self.distance_metric!r}"
        assert self.maintenance in MAINTENANCE_CONDITIONS, f"unknown maintenance: {self.maintenance!r}"
        assert 0 < self.churn_pct <= 100, "churn_pct must be in (0, 100]"
        assert self.rounds > 0
        assert self.repetitions > 0
        assert self.active_size > 0
        assert self.m > 0 and self.ef_construction > 0 and self.ef_search > 0
        assert self.k > 0
        assert self.n_recall_queries > 0 and self.n_latency_queries > 0
        assert self.n_recall_queries <= self.n_latency_queries, (
            "recall subset can't exceed the latency query set"
        )


def load_config(path: Path, base_path: Path = Path("configs/base.yaml")) -> ExperimentConfig:
    """Loads base_path, then applies path's keys on top of it - lets
    per-dataset experiment files override just what differs."""
    with open(base_path) as f:
        data = yaml.safe_load(f)
    with open(path) as f:
        data.update(yaml.safe_load(f))
    return ExperimentConfig(**data)
