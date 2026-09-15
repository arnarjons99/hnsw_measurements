"""Exact k-NN ground truth, computed outside Postgres (faiss brute force).

Called again after every churn round during the real experiment, not just
once - see PLAN.md's "ground truth must be recomputed" correctness item.
"""

import faiss
import numpy as np


def compute_ground_truth(
    active: np.ndarray, query: np.ndarray, k: int, metric: str
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (neighbor_ids, distances), each shape (n_query, k).
    Distances are in pgvector's own units - plain L2 for 'l2' (faiss
    itself returns squared L2), cosine distance (1 - cos sim) for
    'cosine' - so they're directly comparable to <-> / <=> output."""
    d = active.shape[1]
    if metric == "l2":
        index = faiss.IndexFlatL2(d)
        index.add(active)
        sq_dist, ids = index.search(query, k)
        return ids, np.sqrt(sq_dist)
    if metric == "cosine":
        active_n, query_n = active.copy(), query.copy()
        faiss.normalize_L2(active_n)
        faiss.normalize_L2(query_n)
        index = faiss.IndexFlatIP(d)
        index.add(active_n)
        sim, ids = index.search(query_n, k)
        return ids, 1.0 - sim
    raise ValueError(f"unknown metric: {metric!r}")
