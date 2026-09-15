"""Cross-checks compute_ground_truth against SIFT1M's own published ground
truth, on the untouched base set. The one test PLAN.md calls out as worth
keeping - if this ever fails, every recall number in the study is suspect.

Run directly: .venv/bin/python tests/test_ground_truth.py
"""

from pathlib import Path

import numpy as np

from hnsw_bench.datasets import read_fvecs, read_ivecs
from hnsw_bench.ground_truth import compute_ground_truth

SIFT_DIR = Path(__file__).parent.parent / "data" / "sift1m"


def test_sift1m_ground_truth_matches_published() -> None:
    base = read_fvecs(SIFT_DIR / "sift_base.fvecs")
    query = read_fvecs(SIFT_DIR / "sift_query.fvecs")
    published = read_ivecs(SIFT_DIR / "sift_groundtruth.ivecs")
    k = published.shape[1]

    computed_ids, _ = compute_ground_truth(base, query, k, metric="l2")

    overlaps = np.array(
        [len(set(computed_ids[i]) & set(published[i])) / k for i in range(published.shape[0])]
    )
    print(f"mean top-{k} overlap with published ground truth: {overlaps.mean():.4%}")
    print(f"worst-case query overlap: {overlaps.min():.4%}")
    # Ties at the k-th boundary mean exact 100% isn't guaranteed even when
    # correct - a wrong implementation would be nowhere near this close.
    assert overlaps.mean() > 0.99, "ground truth oracle diverges from published SIFT1M ground truth"


if __name__ == "__main__":
    test_sift1m_ground_truth_matches_published()
    print("PASSED")
