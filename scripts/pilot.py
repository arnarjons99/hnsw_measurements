"""Stage 2 pilot: one build->query pass at small scale (50K vectors), to
prove the load->index->query->recall loop works end to end and compare
HNSW param combinations before committing to full-scale automation.
See PLAN.md Stage 2.

Run: .venv/bin/python scripts/pilot.py --m 16 --ef-construction 64 --ef-search 40
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from hnsw_bench.datasets import partition_sift1m
from hnsw_bench.db import connect
from hnsw_bench.ground_truth import compute_ground_truth

PILOT_SIZE = 50_000
N_RECALL_QUERIES = 1_000  # matches the locked methodology's recall subset
K = 10
RESULTS_FILE = Path("results/pilot.csv")


def recall_at_k(returned_dists: np.ndarray, true_dists: np.ndarray, k: int) -> float:
    # Distance-threshold, tie-tolerant recall (PLAN.md Eq. 1), not ID
    # overlap: any returned point within the k-th true neighbor's distance
    # counts, so boundary ties don't get unfairly penalized.
    threshold = true_dists[k - 1] + 1e-4
    return float((returned_dists <= threshold).sum()) / k


def append_result(row: dict) -> None:
    RESULTS_FILE.parent.mkdir(exist_ok=True)
    is_new = not RESULTS_FILE.exists()
    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=64)
    parser.add_argument("--ef-search", type=int, default=40)
    args = parser.parse_args()

    print(f"Loading SIFT1M, taking first {PILOT_SIZE} of the active set...")
    partition = partition_sift1m(Path("data/sift1m"))
    pilot_vectors = partition.active[:PILOT_SIZE]
    pilot_ids = partition.active_ids[:PILOT_SIZE]
    query = partition.query[:N_RECALL_QUERIES]

    print("Computing ground truth for this pilot subset...")
    _, true_dists = compute_ground_truth(pilot_vectors, query, k=K, metric="l2")

    conn = connect()
    cur = conn.cursor()

    print(f"Loading {PILOT_SIZE} vectors into Postgres...")
    cur.execute("DROP TABLE IF EXISTS pilot")
    cur.execute("CREATE TABLE pilot (id BIGINT PRIMARY KEY, embedding VECTOR(128))")
    with cur.copy("COPY pilot (id, embedding) FROM STDIN") as copy:
        for vec_id, vec in zip(pilot_ids, pilot_vectors):
            copy.write_row((int(vec_id), vec))

    print(f"Building HNSW index (m={args.m}, ef_construction={args.ef_construction})...")
    t0 = time.perf_counter()
    cur.execute(
        f"CREATE INDEX ON pilot USING hnsw (embedding vector_l2_ops) "
        f"WITH (m = {args.m}, ef_construction = {args.ef_construction})"
    )
    build_s = time.perf_counter() - t0
    print(f"  build time: {build_s:.2f}s")

    cur.execute(f"SET hnsw.ef_search = {args.ef_search}")

    print(f"Running {N_RECALL_QUERIES} queries...")
    recalls = []
    latencies = []
    for i, q in enumerate(query):
        t0 = time.perf_counter()
        cur.execute(
            "SELECT embedding <-> %s AS dist FROM pilot ORDER BY dist LIMIT %s",
            (q, K),
        )
        returned_dists = np.array([row[0] for row in cur.fetchall()])
        latencies.append(time.perf_counter() - t0)
        recalls.append(recall_at_k(returned_dists, true_dists[i], K))

    latencies_ms = np.array(latencies) * 1000
    recall_mean = float(np.mean(recalls))
    p50 = float(np.percentile(latencies_ms, 50))
    p95 = float(np.percentile(latencies_ms, 95))
    mean_ms = float(np.mean(latencies_ms))

    print(f"\nrecall@{K}: {recall_mean:.4f}")
    print(f"latency ms: p50={p50:.2f} p95={p95:.2f} mean={mean_ms:.2f}")

    append_result(
        {
            "m": args.m,
            "ef_construction": args.ef_construction,
            "ef_search": args.ef_search,
            "pilot_size": PILOT_SIZE,
            "build_s": round(build_s, 3),
            "recall_at_10": round(recall_mean, 4),
            "latency_p50_ms": round(p50, 3),
            "latency_p95_ms": round(p95, 3),
            "latency_mean_ms": round(mean_ms, 3),
        }
    )
    print(f"appended to {RESULTS_FILE}")

    conn.close()


if __name__ == "__main__":
    main()
