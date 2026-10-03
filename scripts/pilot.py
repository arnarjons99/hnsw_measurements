"""Stage 2 pilot: one build->query pass at small scale (50K vectors), to
prove the load->index->query->recall loop works end to end and compare
HNSW param combinations before committing to full-scale automation.
See PLAN.md Stage 2.

Run: .venv/bin/python scripts/pilot.py --dataset glove --ef-search 80 150 200
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
from psycopg import sql

from hnsw_bench.db import connect
from hnsw_bench.experiment import DIST_OPS
from hnsw_bench.ground_truth import compute_ground_truth, recall_at_k
from hnsw_bench.run import DATA_PATHS, PARTITIONERS

N_RECALL_QUERIES = 1_000  # matches the locked methodology's recall subset
K = 10
METRICS = {"sift1m": "l2", "glove": "cosine"}


def append_result(results_file: Path, row: dict) -> None:
    results_file.parent.mkdir(exist_ok=True)
    is_new = not results_file.exists()
    with open(results_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(METRICS), default="sift1m")
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=64)
    parser.add_argument("--ef-search", type=int, nargs="+", default=[40])
    parser.add_argument("--pilot-size", type=int, default=50_000)
    args = parser.parse_args()
    pilot_size = args.pilot_size
    metric = METRICS[args.dataset]
    op_class, op = DIST_OPS[metric]
    results_file = Path(
        "results/pilot.csv" if args.dataset == "sift1m" else f"results/pilot_{args.dataset}.csv"
    )

    print(f"Loading {args.dataset}, taking first {pilot_size} of the active set...")
    partition = PARTITIONERS[args.dataset](DATA_PATHS[args.dataset])
    pilot_vectors = np.asarray(partition.active[:pilot_size])
    pilot_ids = partition.active_ids[:pilot_size]
    query = np.asarray(partition.query[:N_RECALL_QUERIES])
    dim = pilot_vectors.shape[1]

    print("Computing ground truth for this pilot subset...")
    _, true_dists = compute_ground_truth(pilot_vectors, query, k=K, metric=metric)

    conn = connect()
    cur = conn.cursor()

    print(f"Loading {pilot_size} vectors into Postgres...")
    cur.execute("DROP TABLE IF EXISTS pilot")
    cur.execute(
        sql.SQL("CREATE TABLE pilot (id BIGINT PRIMARY KEY, embedding VECTOR({}))").format(
            sql.Literal(dim)
        )
    )
    with cur.copy("COPY pilot (id, embedding) FROM STDIN") as copy:
        for vec_id, vec in zip(pilot_ids, pilot_vectors):
            copy.write_row((int(vec_id), vec))

    print(
        f"Building HNSW index (m={args.m}, ef_construction={args.ef_construction})..."
    )
    t0 = time.perf_counter()
    cur.execute(
        sql.SQL(
            "CREATE INDEX ON pilot USING hnsw (embedding {}) "
            "WITH (m = {}, ef_construction = {})"
        ).format(
            sql.SQL(op_class), sql.Literal(args.m), sql.Literal(args.ef_construction)
        )
    )
    build_s = time.perf_counter() - t0
    print(f"  build time: {build_s:.2f}s")

    query_sql = sql.SQL("SELECT embedding {} %s AS dist FROM pilot ORDER BY dist LIMIT %s").format(
        sql.SQL(op)
    )
    for ef_search in args.ef_search:
        cur.execute(sql.SQL("SET hnsw.ef_search = {}").format(sql.Literal(ef_search)))

        print(f"\nef_search={ef_search}: running {N_RECALL_QUERIES} queries...")
        recalls = []
        latencies = []
        for i, q in enumerate(query):
            t0 = time.perf_counter()
            cur.execute(query_sql, (q, K))
            returned_dists = np.array([row[0] for row in cur.fetchall()])
            latencies.append(time.perf_counter() - t0)
            recalls.append(recall_at_k(returned_dists, true_dists[i], K))

        latencies_ms = np.array(latencies) * 1000
        recall_mean = float(np.mean(recalls))
        p50 = float(np.percentile(latencies_ms, 50))
        p95 = float(np.percentile(latencies_ms, 95))
        mean_ms = float(np.mean(latencies_ms))

        print(f"recall@{K}: {recall_mean:.4f}")
        print(f"latency ms: p50={p50:.2f} p95={p95:.2f} mean={mean_ms:.2f}")

        append_result(
            results_file,
            {
                "m": args.m,
                "ef_construction": args.ef_construction,
                "ef_search": ef_search,
                "pilot_size": pilot_size,
                "build_s": round(build_s, 3),
                "recall_at_10": round(recall_mean, 4),
                "latency_p50_ms": round(p50, 3),
                "latency_p95_ms": round(p95, 3),
                "latency_mean_ms": round(mean_ms, 3),
            },
        )
    print(f"appended to {results_file}")

    conn.close()


if __name__ == "__main__":
    main()
