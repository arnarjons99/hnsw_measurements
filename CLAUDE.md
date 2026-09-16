# CLAUDE.md

Research-methodology project (KTH II2202) measuring pgvector HNSW index degradation under repeated delete+insert churn, and how VACUUM/REINDEX recover it. See [PLAN.md](PLAN.md) for full methodology and architecture.

## Comments

Keep inline comments short — 1-2 lines. This project has a lot of scientifically-motivated parameter choices (why a GUC deviates from its default, why a churn scheme was picked, etc.), and the temptation is to justify all of it inline. Resist that: if a choice needs more than 1-2 lines to explain, put the full reasoning in PLAN.md or the commit message instead, not in the file itself. Code and config should stay easy to scan.
