"""Stage 2 pilot: one churn round + VACUUM/REINDEX, to prove the
delete->insert->measure->maintain->measure loop works before committing to
full-scale automation. See PLAN.md Stage 2.

Ground truth gets recomputed against the post-churn active set, not reused
from baseline - PLAN.md's top correctness item (the active set is
different vectors the moment the first delete happens).

Run: .venv/bin/python scripts/pilot_churn.py --churn-pct 10
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
from psycopg import sql

from hnsw_bench.datasets import partition_sift1m
from hnsw_bench.db import connect
from hnsw_bench.ground_truth import compute_ground_truth, recall_at_k

N_RECALL_QUERIES = 1_000  # matches the locked methodology's recall subset
K = 10
M, EF_CONSTRUCTION, EF_SEARCH = 16, 64, 80  # frozen from the param-sweep pilot
RESULTS_FILE = Path("results/pilot_churn.csv")
# Wide header so every row has the same shape
FIELDNAMES = [
    "phase",
    "recall_at_10",
    "latency_p50_ms",
    "latency_p95_ms",
    "delete_s",
    "insert_s",
    "vacuum_s",
    "reindex_s",
]


def append_result(row: dict) -> None:
    RESULTS_FILE.parent.mkdir(exist_ok=True)
    is_new = not RESULTS_FILE.exists()
    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def measure(cur, active_vectors: np.ndarray, query: np.ndarray, k: int):
    """Recomputes ground truth against the CURRENT active set, then runs
    the query set. Returns (recall, latencies_ms)."""
    _, true_dists = compute_ground_truth(active_vectors, query, k=k, metric="l2")
    recalls, latencies = [], []
    for i, q in enumerate(query):
        t0 = time.perf_counter()
        cur.execute(
            "SELECT embedding <-> %s AS dist FROM pilot ORDER BY dist LIMIT %s",
            (q, k),
        )
        returned_dists = np.array([row[0] for row in cur.fetchall()])
        latencies.append(time.perf_counter() - t0)
        recalls.append(recall_at_k(returned_dists, true_dists[i], k))
    return float(np.mean(recalls)), np.array(latencies) * 1000


def report(
    phase: str, recall: float, latencies_ms: np.ndarray, extra: dict | None = None
) -> None:
    row = {
        "phase": phase,
        "recall_at_10": round(recall, 4),
        "latency_p50_ms": round(float(np.percentile(latencies_ms, 50)), 3),
        "latency_p95_ms": round(float(np.percentile(latencies_ms, 95)), 3),
        **(extra or {}),
    }
    print(
        f"[{phase}] recall@10={row['recall_at_10']:.4f} p50={row['latency_p50_ms']:.2f}ms "
        f"p95={row['latency_p95_ms']:.2f}ms" + (f"  {extra}" if extra else "")
    )
    append_result(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-size", type=int, default=50_000)
    parser.add_argument("--churn-pct", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    print(f"Loading SIFT1M, taking first {args.pilot_size} of the active set...")
    partition = partition_sift1m(Path("data/sift1m"))
    active_vectors = partition.active[: args.pilot_size].copy()
    active_ids = partition.active_ids[: args.pilot_size].copy()
    pool_vectors = partition.replacement_pool
    pool_ids = partition.replacement_pool_ids
    query = partition.query[:N_RECALL_QUERIES]

    conn = connect()
    cur = conn.cursor()

    print(f"Loading {args.pilot_size} vectors into Postgres...")
    cur.execute("DROP TABLE IF EXISTS pilot")
    cur.execute("CREATE TABLE pilot (id BIGINT PRIMARY KEY, embedding VECTOR(128))")
    with cur.copy("COPY pilot (id, embedding) FROM STDIN") as copy:
        for vec_id, vec in zip(active_ids, active_vectors):
            copy.write_row((int(vec_id), vec))

    print(f"Building HNSW index (m={M}, ef_construction={EF_CONSTRUCTION})...")
    cur.execute(
        sql.SQL(
            "CREATE INDEX pilot_hnsw_idx ON pilot USING hnsw (embedding vector_l2_ops) "
            "WITH (m = {}, ef_construction = {})"
        ).format(sql.Literal(M), sql.Literal(EF_CONSTRUCTION))
    )
    cur.execute(sql.SQL("SET hnsw.ef_search = {}").format(sql.Literal(EF_SEARCH)))

    print("\n--- baseline ---")
    recall, lat = measure(cur, active_vectors, query, K)
    report("baseline", recall, lat)

    n_churn = int(args.pilot_size * args.churn_pct / 100)
    print(f"\n--- churn round: delete+insert {n_churn} ({args.churn_pct}%) ---")
    delete_positions = rng.choice(len(active_ids), size=n_churn, replace=False)
    delete_ids = active_ids[delete_positions]
    insert_positions = rng.choice(len(pool_ids), size=n_churn, replace=False)
    insert_ids = pool_ids[insert_positions]
    insert_vectors = pool_vectors[insert_positions]

    t0 = time.perf_counter()
    cur.execute("DELETE FROM pilot WHERE id = ANY(%s)", (delete_ids.tolist(),))
    delete_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    with cur.copy("COPY pilot (id, embedding) FROM STDIN") as copy:
        for vec_id, vec in zip(insert_ids, insert_vectors):
            copy.write_row((int(vec_id), vec))
    insert_s = time.perf_counter() - t0
    print(f"  delete: {delete_s:.2f}s  insert: {insert_s:.2f}s")

    # Mirror the same churn on our own tracked active set so ground truth
    # is recomputed against exactly what's now in the table.
    keep_mask = np.ones(len(active_ids), dtype=bool)
    keep_mask[delete_positions] = False
    active_vectors = np.concatenate([active_vectors[keep_mask], insert_vectors])
    active_ids = np.concatenate([active_ids[keep_mask], insert_ids])

    print("\n--- after churn (before maintenance) ---")
    recall, lat = measure(cur, active_vectors, query, K)
    report(
        "post_churn",
        recall,
        lat,
        {"delete_s": round(delete_s, 3), "insert_s": round(insert_s, 3)},
    )

    print("\n--- maintenance: VACUUM + REINDEX ---")
    t0 = time.perf_counter()
    cur.execute("VACUUM pilot")
    vacuum_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    cur.execute("REINDEX INDEX pilot_hnsw_idx")
    reindex_s = time.perf_counter() - t0
    print(f"  vacuum: {vacuum_s:.2f}s  reindex: {reindex_s:.2f}s")

    print("\n--- after maintenance ---")
    recall, lat = measure(cur, active_vectors, query, K)
    report(
        "post_maintenance",
        recall,
        lat,
        {"vacuum_s": round(vacuum_s, 3), "reindex_s": round(reindex_s, 3)},
    )

    conn.close()


if __name__ == "__main__":
    main()
