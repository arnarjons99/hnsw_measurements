"""Load SIFT1M/GloVe vectors and partition them into active/replacement/query pools."""

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _debug(msg: str) -> None:
    # flush=True so the last line survives an OOM kill.
    rss_gb = int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e9
    print(f"[debug rss={rss_gb:.2f}GB] {msg}", flush=True)


class LazyRows:
    """Rows of `base` picked by `ids`, copied into memory only when indexed."""

    def __init__(self, base: np.ndarray, ids: np.ndarray):
        self.base, self.ids = base, ids

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, positions) -> np.ndarray:
        return np.asarray(self.base[self.ids[positions]])


@dataclass
class Partition:
    active: np.ndarray
    active_ids: np.ndarray
    replacement_pool: np.ndarray | LazyRows
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


GLOVE_BATCH_LINES = 10_000


def read_glove_text(path: Path) -> np.ndarray:
    """Returns the vectors as a read-only memmap of the .npy cache, building the
    cache first if missing by streaming the text file in GLOVE_BATCH_LINES batches."""
    vectors_cache = path.with_suffix(".npy")
    words_cache = path.with_suffix(".words.txt")
    if vectors_cache.exists() and words_cache.exists():
        _debug(f"loading cached {vectors_cache}")
        return np.load(vectors_cache, mmap_mode="r")

    _debug(f"no cache, counting lines in {path}")
    with open(path, encoding="utf-8") as f:
        dim = len(f.readline().rstrip("\n").split(" ")) - 1
        n_lines = 1 + sum(1 for _ in f)
    _debug(f"{n_lines:,} lines, dim={dim}")

    # Write to .partial and rename at the end, so a killed run leaves no valid-looking cache.
    vectors_tmp = vectors_cache.with_name(vectors_cache.name + ".partial")
    words_tmp = words_cache.with_name(words_cache.name + ".partial")
    out = np.lib.format.open_memmap(vectors_tmp, mode="w+", dtype=np.float32, shape=(n_lines, dim))
    batch = np.empty((GLOVE_BATCH_LINES, dim), dtype=np.float32)
    row = filled = 0
    with open(path, encoding="utf-8") as f, open(words_tmp, "w", encoding="utf-8") as words_out:
        for line in f:
            parts = line.rstrip("\n").split(" ")
            words_out.write(parts[0] + "\n")
            batch[filled] = parts[1:]
            filled += 1
            if filled == GLOVE_BATCH_LINES:
                out[row : row + filled] = batch
                row, filled = row + filled, 0
                if row % 100_000 == 0:
                    _debug(f"parsed {row:,}/{n_lines:,} lines")
        out[row : row + filled] = batch[:filled]
        row += filled
    assert row == n_lines, f"{path}: expected {n_lines} lines, parsed {row}"
    out.flush()
    del out

    os.replace(words_tmp, words_cache)
    os.replace(vectors_tmp, vectors_cache)
    _debug(f"wrote cache {vectors_cache}")
    return np.load(vectors_cache, mmap_mode="r")


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
    _debug(f"partition_glove start: {glove_path}")
    vectors = read_glove_text(glove_path)
    _debug(f"vectors memmapped, shape={vectors.shape}")
    ids = np.arange(vectors.shape[0])
    np.random.default_rng(seed).shuffle(ids)

    query_ids = ids[:query_size]
    active_ids = ids[query_size : query_size + active_size]
    replacement_ids = ids[query_size + active_size :]
    _debug(f"split: query={len(query_ids):,} active={len(active_ids):,} replacement={len(replacement_ids):,}")
    partition = Partition(
        active=vectors[active_ids],
        active_ids=active_ids,
        replacement_pool=LazyRows(vectors, replacement_ids),
        replacement_pool_ids=replacement_ids,
        query=vectors[query_ids],
    )
    _debug("partition_glove done")
    return partition
