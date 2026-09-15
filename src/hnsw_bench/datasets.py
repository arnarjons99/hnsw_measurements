"""Load SIFT1M/GloVe vectors and partition them into active/replacement/query pools."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Partition:
    active: np.ndarray
    active_ids: np.ndarray
    replacement_pool: np.ndarray
    replacement_pool_ids: np.ndarray
    query: np.ndarray


def read_fvecs(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int32)
    dim = int(raw[0])
    raw = raw.reshape(-1, dim + 1)
    assert (raw[:, 0] == dim).all(), f"{path}: inconsistent per-vector dimension header"
    # Bit-reinterpret, not value-cast: the payload columns are float32 bytes
    # read as int32, so .view() (not .astype()) is required for correctness.
    return raw[:, 1:].view(np.float32).copy()


def read_ivecs(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int32)
    dim = int(raw[0])
    raw = raw.reshape(-1, dim + 1)
    assert (raw[:, 0] == dim).all(), f"{path}: inconsistent per-vector dimension header"
    return raw[:, 1:].copy()


def read_glove_text(path: Path, cache: bool = True) -> tuple[list[str], np.ndarray]:
    # First call on a fresh machine takes ~15min (1.29M lines, pure-Python
    # parse) before the .npy cache exists; every call after is near-instant.
    vectors_cache = path.with_suffix(".npy")
    words_cache = path.with_suffix(".words.txt")
    if cache and vectors_cache.exists() and words_cache.exists():
        return words_cache.read_text().splitlines(), np.load(vectors_cache)

    words: list[str] = []
    vectors: list[list[str]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split(" ")
            words.append(parts[0])
            vectors.append(parts[1:])
    vector_array = np.array(vectors, dtype=np.float32)

    if cache:
        np.save(vectors_cache, vector_array)
        words_cache.write_text("\n".join(words))

    return words, vector_array


def partition_sift1m(sift_dir: Path, active_size: int = 500_000, seed: int = 42) -> Partition:
    """Active/replacement split drawn from base+learn pooled together (600K);
    query is SIFT1M's own disjoint 10K query file. See PLAN.md open item 6."""
    base = read_fvecs(sift_dir / "sift_base.fvecs")
    learn = read_fvecs(sift_dir / "sift_learn.fvecs")
    query = read_fvecs(sift_dir / "sift_query.fvecs")

    pool = np.concatenate([base, learn], axis=0)
    ids = np.arange(pool.shape[0])
    np.random.default_rng(seed).shuffle(ids)

    active_ids, replacement_ids = ids[:active_size], ids[active_size:]
    return Partition(
        active=pool[active_ids],
        active_ids=active_ids,
        replacement_pool=pool[replacement_ids],
        replacement_pool_ids=replacement_ids,
        query=query,
    )


def partition_glove(
    glove_path: Path, active_size: int = 500_000, query_size: int = 10_000, seed: int = 42
) -> Partition:
    """No upstream split exists, so active/replacement/query are all carved
    from the same flat vocabulary. See PLAN.md open item 7."""
    _, vectors = read_glove_text(glove_path)
    ids = np.arange(vectors.shape[0])
    np.random.default_rng(seed).shuffle(ids)

    query_ids = ids[:query_size]
    active_ids = ids[query_size : query_size + active_size]
    replacement_ids = ids[query_size + active_size :]
    return Partition(
        active=vectors[active_ids],
        active_ids=active_ids,
        replacement_pool=vectors[replacement_ids],
        replacement_pool_ids=replacement_ids,
        query=vectors[query_ids],
    )
