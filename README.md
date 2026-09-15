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

