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

## Python environment

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
```

## Running an experiment

Results land in `results/<run_id>/` (`--run-id` to name it yourself). **Try `configs/experiments/sift1m_dryrun.yaml` first** - it runs the full pipeline on 2,000 vectors in seconds.

```bash
.venv/bin/python -m hnsw_bench.run configs/experiments/sift1m_baseline.yaml
```



