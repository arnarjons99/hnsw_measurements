"""Stage 3 experiment loop: one repetition is build -> baseline measure
-> 5x [churn -> measure -> maintenance -> measure]. Churn, warm-up and
maintenance rules follow the PLAN.md Sec 11 resolutions.
"""

import time
from typing import LiteralString, cast

import numpy as np
from psycopg import sql

from hnsw_bench.config import ExperimentConfig
from hnsw_bench.datasets import Partition
from hnsw_bench.db import connect
from hnsw_bench.ground_truth import compute_ground_truth, recall_at_k
from hnsw_bench.metrics import collect_metrics
from hnsw_bench.results_writer import ResultsWriter

DIST_OPS = {"l2": ("vector_l2_ops", "<->"), "cosine": ("vector_cosine_ops", "<=>")}


def build_index(
    cur, config: ExperimentConfig, active_ids: np.ndarray, active_vectors: np.ndarray
) -> None:
    op_class, _ = DIST_OPS[config.distance_metric]
    dim = active_vectors.shape[1]
    cur.execute("DROP TABLE IF EXISTS active")
    cur.execute(
        sql.SQL(
            "CREATE TABLE active (id BIGINT PRIMARY KEY, embedding VECTOR({}))"
        ).format(sql.Literal(dim))
    )
    with cur.copy("COPY active (id, embedding) FROM STDIN") as copy:
        for vec_id, vec in zip(active_ids, active_vectors):
            copy.write_row((int(vec_id), vec))
    cur.execute(
        sql.SQL(
            "CREATE INDEX active_hnsw_idx ON active USING hnsw (embedding {}) "
            "WITH (m = {}, ef_construction = {})"
        ).format(
            sql.SQL(cast(LiteralString, op_class)),
            sql.Literal(config.m),
            sql.Literal(config.ef_construction),
        )
    )
    cur.execute(
        sql.SQL("SET hnsw.ef_search = {}").format(sql.Literal(config.ef_search))
    )


def run_queries(
    cur, op: str, queries: np.ndarray, k: int
) -> list[tuple[float, np.ndarray]]:
    """Returns (latency_ms, returned_dists) for each query, in order."""
    query_sql = sql.SQL(
        "SELECT embedding {} %s AS dist FROM active ORDER BY dist LIMIT %s"
    ).format(sql.SQL(cast(LiteralString, op)))
    results = []
    for q in queries:
        t0 = time.perf_counter()
        cur.execute(query_sql, (q, k))
        returned_dists = np.array([row[0] for row in cur.fetchall()])
        results.append(((time.perf_counter() - t0) * 1000, returned_dists))
    return results


def measure(
    cur, config: ExperimentConfig, active_vectors: np.ndarray, latency_query: np.ndarray
):
    """Runs the latency query set twice - the first pass warms the cache and is discarded,
    the second is what gets recorded and feeds recall (over its first n_recall_queries).
    """
    n_recall = config.n_recall_queries
    _, op = DIST_OPS[config.distance_metric]
    _, true_dists = compute_ground_truth(
        active_vectors,
        latency_query[:n_recall],
        k=config.k,
        metric=config.distance_metric,
    )

    # warm up and discard the first pass, then measure the second pass
    run_queries(cur, op, latency_query, config.k)
    measured = run_queries(cur, op, latency_query, config.k)

    latencies, recalls, query_rows = [], [], []
    for i, (latency_ms, returned_dists) in enumerate(measured):
        latencies.append(latency_ms)
        true_dist_k = None
        if i < n_recall:
            recalls.append(recall_at_k(returned_dists, true_dists[i], config.k))
            true_dist_k = float(true_dists[i][config.k - 1])
        query_rows.append(
            {
                "query_index": i,
                "latency_ms": round(latency_ms, 4),
                "true_dist_k": true_dist_k,
                "returned_dists": ",".join(f"{d:.6f}" for d in returned_dists),
            }
        )

    latencies_arr = np.array(latencies)
    summary = {
        "recall_at_10": float(np.mean(recalls)),
        "latency_p50_ms": float(np.percentile(latencies_arr, 50)),
        "latency_p95_ms": float(np.percentile(latencies_arr, 95)),
        "latency_mean_ms": float(np.mean(latencies_arr)),
    }
    return summary, query_rows


def apply_maintenance(cur, condition: str) -> dict:
    durations = {}
    if condition in ("reindex", "reindex_vacuum"):
        t0 = time.perf_counter()
        cur.execute("REINDEX INDEX active_hnsw_idx")
        durations["reindex_s"] = time.perf_counter() - t0
    if condition in ("vacuum", "reindex_vacuum"):
        t0 = time.perf_counter()
        cur.execute("VACUUM active")
        durations["vacuum_s"] = time.perf_counter() - t0
    return durations


def run_repetition(
    config: ExperimentConfig,
    partition: Partition,
    writer: ResultsWriter,
    repetition: int,
) -> None:
    rng = np.random.default_rng(config.seed + repetition)
    active_ids = partition.active_ids.copy()
    active_vectors = partition.active.copy()
    pool_ids = partition.replacement_pool_ids
    pool_vectors = partition.replacement_pool
    pool_available = np.ones(len(pool_ids), dtype=bool)
    latency_query = partition.query[: config.n_latency_queries]
    n_churn = int(config.active_size * config.churn_pct / 100)

    conn = connect()
    cur = conn.cursor()

    def checkpoint(round_num: int, phase: str, extra: dict | None = None) -> None:
        summary, query_rows = measure(cur, config, active_vectors, latency_query)
        row = {
            "repetition": repetition,
            "round": round_num,
            "phase": phase,
            **summary,
            **collect_metrics(cur),
            **(extra or {}),
        }
        writer.write_measurement(row)
        writer.write_queries(
            [
                {"repetition": repetition, "round": round_num, "phase": phase, **q}
                for q in query_rows
            ]
        )
        writer.update_status(
            current_repetition=repetition, current_round=round_num, current_phase=phase
        )

    build_index(cur, config, active_ids, active_vectors)
    checkpoint(0, "baseline")

    for round_num in range(1, config.rounds + 1):
        # decide what to delete and what to insert
        delete_positions = rng.choice(len(active_ids), size=n_churn, replace=False)
        available_positions = np.flatnonzero(pool_available)
        insert_positions = rng.choice(available_positions, size=n_churn, replace=False)
        pool_available[insert_positions] = False

        # perform the churn: delete from active, insert from pool, update active_ids/vectors
        delete_ids = active_ids[delete_positions]
        insert_ids = pool_ids[insert_positions]
        insert_vectors = pool_vectors[insert_positions]

        # deleting
        t0 = time.perf_counter()
        cur.execute("DELETE FROM active WHERE id = ANY(%s)", (delete_ids.tolist(),))
        delete_s = time.perf_counter() - t0

        # inserting
        t0 = time.perf_counter()
        with cur.copy("COPY active (id, embedding) FROM STDIN") as copy:
            for vec_id, vec in zip(insert_ids, insert_vectors):
                copy.write_row((int(vec_id), vec))
        insert_s = time.perf_counter() - t0

        # update the active_ids and active_vectors arrays to reflect the churn
        keep_mask = np.ones(len(active_ids), dtype=bool)
        keep_mask[delete_positions] = False
        active_ids = np.concatenate([active_ids[keep_mask], insert_ids])
        active_vectors = np.concatenate([active_vectors[keep_mask], insert_vectors])

        checkpoint(
            round_num, "post_churn", {"delete_s": delete_s, "insert_s": insert_s}
        )

        # apply maintenance if requested, then measure again
        if config.maintenance != "none":
            durations = apply_maintenance(cur, config.maintenance)
            checkpoint(round_num, "post_maintenance", durations)

    conn.close()
    writer.mark_repetition_complete(repetition)
