# hnsw_measurements

Evaluating HNSW indexes in pgvector under repeated data replacement and PostgreSQL maintenance (KTH II2202). See [PLAN.md](PLAN.md) for methodology and architecture.

## Environment

Postgres 18.6 + pgvector 0.8.6 via Docker Compose:

```bash
docker compose -f docker/docker-compose.yml up -d
```

Connects on `localhost:5432`, user/db/password all `hnsw`. Tear down with:

```bash
docker compose -f docker/docker-compose.yml down
```

## Datasets

Gitignored — download and extract before running anything:

```bash
mkdir -p data/sift1m data/glove

curl -o data/sift1m/sift.tar.gz ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar -xzf data/sift1m/sift.tar.gz --strip-components=1 -C data/sift1m

curl -L -o data/glove/glove.2024.wikigiga.300d.zip https://nlp.stanford.edu/data/wordvecs/glove.2024.wikigiga.300d.zip
unzip data/glove/glove.2024.wikigiga.300d.zip -d data/glove
```

The zip extracts to `wiki_giga_2024_300_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05_combined.txt`, not a name matching the zip - that's the path `DATA_PATHS["glove"]` in `run.py` points at. First load parses ~1.29M lines in pure Python (~15min) and caches a `.npy` + `.words.txt` next to it; every load after that is near-instant.

## Python environment

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
```

## Running an experiment

Results land in `results/<run_id>/` (`--run-id` to name it yourself). **Try the dry-run configs first** - each runs the full pipeline on 2,000 vectors in seconds:

```bash
.venv/bin/python -m hnsw_bench.run configs/experiments/sift1m_dryrun.yaml --run-id sift1m_pilot
.venv/bin/python -m hnsw_bench.run configs/experiments/glove_dryrun.yaml --run-id glove_pilot
```

Then a real run:

```bash
.venv/bin/python -m hnsw_bench.run configs/experiments/sift1m_baseline.yaml
.venv/bin/python -m hnsw_bench.run configs/experiments/glove_baseline.yaml
```



