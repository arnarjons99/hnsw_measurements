# pgvector HNSW replacement/maintenance experiment — architecture review & implementation plan

## Context

This is a university research project measuring how repeated DELETE+INSERT replacement cycles degrade pgvector HNSW search quality/performance on Postgres 18.6 + pgvector 0.8.6, and how VACUUM/REINDEX recover it afterward. The repo is currently empty (stub README only) — this is a greenfield design, not a refactor. The goal of this phase is to critique the proposed architecture and lock down a concrete plan *before* writing code, per the user's explicit request. Nothing has been implemented yet.

The core tension to design around: this is a scientific instrument, not a product. Every extra moving part (a second process, a database-backed results store, a UI feature) is something that can introduce timing noise, desync bugs, or silent invalidation of months of overnight runs. The plan below repeatedly favors "one obviously-correct sequential process + flat files" over anything fancier.

This plan has been cross-checked against the actual submitted **project proposal** (`project_proposal_HNSW.pdf`, 04.09.2026) and **research plan** (`II2202_Research_Plan.pdf`, 11.09.2026) for KTH II2202. That changes scope in one big way and several small ones — see the callout below and inline notes marked **[proposal/plan]**.

## What the actual proposal & research plan change

**Scope, not a countdown.** The research plan's dated milestones (env ready 17 Sept, automation 23 Sept, data collection 28 Sept, report 20 Oct) and the 56-hour estimate are standard course-deliverable scaffolding for II2202 — a Gantt-style table the template asks for, not a strict per-stage budget to optimize against. The real constraint is just: everything done and written up by 20 October. That said, the stage sequence itself (setup → pilot → automate → full runs → analysis → report) is sensible regardless of dates, and it's worth building this at the complexity a 2-person, few-week project actually needs rather than a general-purpose research platform — see §8/§9, now reasoned from project size rather than a manufactured deadline.

**Already settled by your own documents (no need to revisit):**
- Single benchmark client, not two — the research plan refers to "the benchmark client" (singular, §2.3.1), running on the same host as the database.
- Ground-truth-recompute-per-round cost — already solved: recall is measured over a fixed 1,000-query subset specifically *because* it needs brute-force ground truth every round, while the full 10,000-query set (no ground truth needed) carries the latency numbers. The plan below is written to that exact split.
- Container RAM risk — already in your own risk register (research plan §3.3); confirms rather than adds to the concern in item 4 below.

**New, concrete issues surfaced by close reading of the two documents**, detailed inline below: a likely arithmetic inconsistency in the churn model, an unspecified GloVe query/train split, a maintenance-condition-count mismatch between the two documents (5 vs. 4), an unstated distance metric for GloVe, two different warm-up descriptions, and a "single host" vs. "local machines" ambiguity given two team members. See the new subsection in §1–2 and the updated §11.

---

## 1–2. Architecture feedback & scientific problems

The overall shape (Docker + pinned versions, YAML-driven Python orchestrator, incremental flat-file results, UI decoupled from execution, analysis decoupled from execution) is sound and appropriately sized for the task. The problems are not architectural — they're in methodological details that the current design is silent on. These would silently invalidate results if not addressed **before** any overnight run:

### Correctness-critical (must fix before first real run)

1. **Ground truth must be recomputed after every replacement round, not just once.** Recall@10 needs the *true* k nearest neighbors within the *currently active* set. After replacement round 1, the active set is no longer the original SIFT1M/GloVe base set — some original vectors are gone, new ones are in. Static ann-benchmarks ground truth (computed against the untouched base set) becomes wrong the moment the first delete happens. This means you need your own exact brute-force k-NN oracle (numpy or `faiss.IndexFlatL2`/`IndexFlatIP`, run outside Postgres) that recomputes ground truth against whatever is active at each checkpoint. This is the single biggest correctness gap in the current plan and needs its own pipeline stage, validated in isolation against published ground truth on the untouched dataset before trusting it after replacement.

2. **Distance metric / operator class must match the dataset.** SIFT1M is evaluated with L2 (`vector_l2_ops`); GloVe is conventionally evaluated with cosine/angular distance (`vector_cosine_ops`). The index operator class, the query's distance operator (`<->` vs `<=>`), and the ground-truth oracle's metric all have to agree per-dataset. Needs to be an explicit, validated per-dataset config, not an implicit default.

3. **Autovacuum must be disabled (or tightly controlled).** If autovacuum fires mid-experiment, it contaminates "before maintenance" measurements and makes VACUUM timing non-reproducible. Set `autovacuum = off` (or off for the target table) and only vacuum when the experiment explicitly triggers it.

4. **A held-out replacement pool is needed per dataset, and it isn't free.** "Insert new vectors" needs a source of vectors that are neither in the active set nor the query set. SIFT1M's corpus conveniently ships a 100K "learn" set that's perfect for this. GloVe (ann-benchmarks HDF5 format) does not ship a separate reserve — you'll need to carve one out of the training vectors up front (e.g., reserve X% as replacement pool, never load it into the initial active set). This needs to be sized so it doesn't run out across the planned cycles × replacement% and must be documented as a per-dataset partitioning decision.

5. **HNSW graph construction is not fully deterministic**, even with a fixed Python-level seed. pgvector's HNSW level-assignment randomness is internal to the extension and (to our knowledge) not exposed as a seedable GUC — parallel index builds add further nondeterminism. "Fixed random seeds" in the plan should be understood as controlling *your* sampling (which vectors are deleted/inserted/queried), not Postgres's internal graph structure. Worth an early empirical check: build the same index twice from identical data and confirm whether recall/latency vary — this justifies (and calibrates expectations for) the 3–5 repetitions rather than assuming they're a formality, and is worth a sentence in your report's limitations section. Consider also setting `max_parallel_maintenance_workers = 0` if you want the closest thing to deterministic serial builds.

### Specific corrections from your actual proposal/research plan text **[proposal/plan]**

6. **"10% of the dataset" is ambiguous, and the two readings change whether the proposal's own claim holds.** The proposal says: "Each subsequent round deletes 10% of the dataset (measured relative to the full dataset size)... Five rounds are performed, so that all initially active vectors are eventually replaced." If "10%" means 10% of the 500,000-vector active set (50,000/round): 5 rounds touch 50% of the table's capacity in total, and if deletions are uniform-random over the *current* active set each round (the natural default), the expected fraction of *original* vectors still untouched after 5 rounds is 0.9⁵ ≈ 59% — not 0%, contradicting "all initially active vectors are eventually replaced." (Worth noting the more recent research plan quietly drops this specific claim.) If instead "10%" means 10% of the *full source corpus* (1,000,000 for SIFT1M / 1,200,000 for GloVe → 100,000/round, i.e. 20% of the active set), a full turnover in 5 rounds is arithmetically possible, but only via a stratified/disjoint-block deletion schedule — uniform-random at 20%/round still only yields ≈33% expected survival, not 0%. Either way, "uniform random each round" and "guaranteed full turnover by round 5" are in tension. Decide explicitly: uniform-random (simpler, easier to justify, "round number" isn't exactly "% of original replaced") vs. stratified blocks (guarantees full turnover, more bookkeeping) — and use whichever reading of "10%" matches.
7. **GloVe likely needs a from-scratch query/train split.** SIFT1M ships a dedicated 10,000-vector query file, disjoint from its 1M base vectors by construction — "drawn from the dataset's test split" (proposal §2.3.2) is a natural fit there. The 2024 GloVe Wikipedia+Gigaword release ("A new pair of gloves") is, as far as I'm aware, distributed like classic GloVe releases — a flat vocabulary+vector file with no predefined train/query split — worth confirming its actual distribution format before assuming one exists. If there isn't one, you'll need to carve a disjoint 10,000-vector query set out of the 1.2M pool yourselves, on top of the 500K-active / rest-as-replacement-pool split already described — i.e., the 3-way partition (active / replacement pool / query pool) needs to be an explicit step in `datasets.py` for GloVe, not something that falls out of the raw file the way it does for SIFT1M.
8. **Maintenance condition count differs between the two documents.** The proposal lists 5 (including "periodic maintenance: best-performing strategy applied periodically"); the research plan lists 4 (baseline, VACUUM, REINDEX, REINDEX+VACUUM) — periodic maintenance was dropped. Recommend building to the research plan's 4 (more recent, more precisely specified — "periodic" was underspecified anyway: periodic at what interval?), but worth a quick confirmation of which is the operative scope before it's encoded into the config schema.
9. **Distance metric per dataset isn't stated in either document.** Cosine is conventional for GloVe-style embeddings, L2 for SIFT — but "conventional" isn't "written down," and it determines both the HNSW operator class and how the brute-force ground truth is computed. A silent mismatch between the two would produce wrong recall numbers without any obvious error.
10. **Warm-up procedure is described two different ways.** "Each query set is executed twice per measurement point, with the first execution discarded" (proposal §2.3.2) vs. "a fixed warm-up workload will be executed before latency measurements" (research plan §2.4). These might be the same thing described loosely, or genuinely different (repeating the *measured* query set itself vs. a separate warm-up set). The plan below implements the proposal's version — simpler, no extra warm-up pool needed — but worth reconciling the wording before it's written down as final methodology.
11. **"Single Linux host" (proposal) vs. "local machines," plural (research plan).** With Pavlo on environment setup and Arnar on automation (research plan §3.2), it would be easy for stage-4 "full experiments" to end up split across two laptops. Fine for development; the actual timed data-collection runs that go into the report need one consistent, otherwise-idle machine — worth agreeing explicitly on whose.

Also worth building in directly rather than treating as open: the recall definition in the research plan (Eq. 1) is a **distance-threshold, tie-tolerant** recall@k, not naive ID-set overlap — the ground-truth/metrics code needs to compute the k-th true neighbour's *distance* and count returned points within that distance, not just intersect ID sets.

### Important (affects validity of comparisons, less catastrophic than the above)

6. **Cache state must be a controlled variable.** Query latency is dominated by whether HNSW pages are warm in `shared_buffers`/OS cache. Define one fixed warm-up procedure (e.g., N throwaway queries before every measured batch) and apply it identically at every checkpoint — otherwise "latency changed" might just mean "cache state changed."
7. **Postgres memory/durability GUCs are experimental variables you haven't named yet**: `shared_buffers`, `work_mem`, `maintenance_work_mem` (dominates REINDEX/VACUUM speed), `fsync`/`synchronous_commit` (dominates INSERT/DELETE speed). These belong in `postgresql.conf`, fixed and recorded in metadata — container CPU/RAM limits alone don't control them.
8. **Dead tuple counts via `pg_stat_user_tables.n_dead_tup` are estimates**, refreshed periodically by the stats subsystem, not exact. Consider the `pgstattuple` extension for an exact (if more expensive) count at measurement checkpoints where precision matters.
9. **Fix the measurement order** (e.g., always: run benchmark queries → record recall/latency → record size/dead-tuple stats) identically at every checkpoint, so the act of measuring one thing doesn't perturb another (e.g., size-check queries touching pages right before a "cold-ish" latency measurement).
10. **Maintenance strategy needs a closed, explicit definition.** Decide up front exactly which of `{VACUUM, VACUUM FULL, REINDEX, REINDEX CONCURRENTLY, none}` (or combinations) are in scope, and pin the exact SQL for each — these have very different lock/space/performance profiles and "how maintenance affects the index" is a core research question, so this shouldn't be left implicit.
11. **Session-level GUCs**: `hnsw.ef_search` is per-session — the query path must `SET` it at the start of every session/connection, not assume it persists. Minor secondary note: Postgres flips prepared statements from a custom to a generic plan after 5 executions (`plan_cache_mode`); unlikely to matter for a plain `ORDER BY embedding <-> $1 LIMIT k`, but worth being aware of if query plans ever look inconsistent.

None of this invalidates the overall architecture — it's all detail work that fits cleanly into the "core DB operations" and "ground truth" modules in the plan below. But it changes scope: the ground-truth oracle and dataset partitioning are now first-class pipeline components, not afterthoughts.

---

## 3. Two-client architecture — recommendation: **one process, not two**

Given you've said queries and updates aren't intended to run concurrently, there's no technical or scientific reason to run them as two separate client processes/services. A single Python orchestrator process should own both, structured as two logical modules (`QueryRunner`, `UpdateRunner`) for code clarity — that gets you the separation-of-concerns benefit without any of the operational cost.

What *is* worth doing: use two separate database connections/sessions (still within one process) — one for querying, one for mutating — purely so session-level GUCs (`hnsw.ef_search`, etc.) don't bleed between phases and each connection's purpose is unambiguous in logs. That's a cheap, good idea. It is a different thing from "two client processes," which would add process coordination, synchronization, and separate failure modes for no current benefit.

If you later want to study *concurrent* query/update interference as its own research question, treat that as an explicitly separate Phase 2 experiment (different methodology: interleaving, attribution of latency spikes, thread/async coordination) — don't build it into the core architecture now. Flagged as an open decision in §11.

**[proposal/plan]** This is already settled by your own documents, not just a recommendation: the research plan refers to "the benchmark client" in the singular (§2.3.1), running on the same host as the database. Nothing to reconsider here.

---

## 4. Recommended repository structure

```
hnsw_measurements/
  docker/
    docker-compose.yml        # postgres+pgvector, pinned image digest, cpu/mem limits
    postgresql.conf           # explicit shared_buffers/work_mem/maintenance_work_mem,
                               # autovacuum=off, fsync/synchronous_commit fixed
  configs/
    base.yaml                 # shared defaults
    experiments/
      sift1m_baseline.yaml
      glove_baseline.yaml
  data/                        # gitignored; downloaded datasets
    sift1m/  glove/
  src/hnsw_bench/
    config.py                 # YAML loading + validation (pydantic or dataclasses)
    datasets.py                # load + partition active/replacement/query pools
    ground_truth.py            # brute-force exact k-NN oracle (numpy/faiss)
    db.py                      # connections, DDL, index creation, GUC helpers
    operations.py               # timed DELETE/INSERT/VACUUM/REINDEX primitives
    query_runner.py             # benchmark queries -> recall + latency
    update_runner.py            # replacement round execution
    metrics.py                   # size / dead-tuple / pg_stat_* collection
    experiment.py                 # one repetition's state machine
    results_writer.py             # incremental CSV/JSON + resume bookkeeping
    run.py                          # CLI entrypoint
  analysis/                    # reads results/*.csv only; no execution coupling
  ui/streamlit_app.py           # later milestone; reads results/, can subprocess the runner
  results/<run_id>/
    config.yaml  metadata.json  measurements.csv  queries.csv  experiment.log  status.json
  tests/
  scripts/download_datasets.sh
  pyproject.toml
```

Notably absent on purpose: no `api/`, no `services/`, no message-queue/worker directories — see §9.

---

## 5. YAML vs. code

**YAML** = anything that is a controlled variable of the study (changing it answers a different research question): dataset, seed, active size, replacement %, cycles, repetitions, HNSW `m`/`ef_construction`, `ef_search` (value or short list), `k`, query counts, maintenance strategy, distance metric, resource limits.

**Code** = anything that's "how the machinery works" rather than "what condition we're testing": the actual DDL/DML, measurement ordering, ground-truth computation method, retry/error policy, file layout, logging format. Exposing these as YAML knobs adds misconfiguration surface with no scientific upside.

Validate the config against a schema (pydantic model, or a hand-written check) at startup — fail in 2 seconds on a typo, not 3 hours into an overnight run.

---

## 6. Results structure & storage

Endorse the proposed `results/<run_id>/` layout with small changes:

- **Single state file**: fold "checkpoint/resume state" into `status.json` rather than a separate file — two files that must stay in sync is a bug waiting to happen. `status.json` carries `current_repetition`, `current_cycle`, `current_phase`, `last_updated`, and which repetitions are fully complete; both the resume logic and the future Streamlit UI read the same file.
- `measurements.csv`: one row per (repetition, checkpoint), wide columns (`recall_at_10`, `latency_p50_ms`, `latency_p95_ms`, `dead_tuples`, `table_size_bytes`, `index_size_bytes`, `delete_duration_s`, `insert_duration_s`, `vacuum_duration_s`, `reindex_duration_s`, …). Duration columns are null where not applicable (e.g. baseline). One flat table, no joins needed for basic plots.
- `queries.csv`: one row per individual query per checkpoint (latency, and — for the 1,000-query recall subset — enough to recompute recall under the tie-tolerant distance-threshold definition: the k-th ground-truth distance and the distances of the returned points, not just IDs) — lets you derive any percentile or re-verify recall later without rerunning anything.
- `config.yaml`: the fully **resolved** config (defaults applied), not just what the user typed — so a run is self-describing even if code defaults change later.
- `metadata.json`: git commit SHA of the experiment code, Docker image digest, full Postgres/pgvector version strings, Python version, dataset checksums, hostname, host CPU/mem — everything needed for the methodology section of the writeup.
- Append-and-flush after every checkpoint (`f.flush(); os.fsync(...)`) — cheap, and bounds crash loss to at most the in-progress checkpoint.

No database for results storage (see §9) — flat files are simpler and sufficient at this scale.

---

## 7. Overnight robustness, resumability, reproducibility

- **Resume granularity = one full repetition**, not sub-cycle. If a repetition completes, mark it done and skip on restart; if interrupted mid-way, restart that repetition from scratch (drop/recreate schema+index) rather than trying to resume mid-mutation. This is the "keep it simple" call: fine-grained resume would require persisting exactly which deletes/inserts landed, for little practical benefit given repetitions are the unit that gets averaged anyway.
- Make this cheap to get right: derive each repetition's replacement schedule deterministically from `(base_seed, repetition_index, cycle_index)` rather than persisting ID lists — a restarted repetition regenerates the identical sequence of deletes/inserts.
- Wrap DB calls in a small retry-with-backoff helper for transient connection hiccups (e.g., container blip) — not a full circuit breaker, just enough that a 2-second glitch doesn't kill an 8-hour run.
- **Pre-flight checks** at run start: container healthy, DB reachable, dataset files present and checksummed, config validates, rough disk-space estimate for the run. Fail fast and loud.
- **Dry-run mode**: a tiny config (thousands of vectors, 1 cycle, 1 repetition) that exercises the entire pipeline in minutes. Treat this as mandatory practice before any real overnight run, and as its own milestone (§8).
- Watch memory headroom explicitly: SIFT1M/GloVe at scale + VACUUM/REINDEX under a fixed container memory cap is a real OOM-kill risk — load-test this in the dry run, not for the first time overnight.

---

## 8. Implementation milestones — sequenced to match your work plan's stages **[proposal/plan]**

Your research plan already lays out sensible stages (setup → pilot → automate → full runs → analysis → report); the attached dates are course-template milestones, not hard walls — a pacing guide, not a countdown. What actually matters is finishing the sequence with enough runway left for analysis and writing before 20 October. The detail below fills in what each stage involves and keeps the automation at a complexity that matches a 2-person, few-week project rather than a general-purpose research platform — that's a judgment about what the project needs, not about how many hours are left on a clock.

**Stage 1 — Experimental setup**
- Docker Compose: Postgres 18.6 + pgvector 0.8.6, pinned image digest, fixed CPU/mem limits, explicit `postgresql.conf` (autovacuum off, `shared_buffers`/`work_mem`/`maintenance_work_mem` set deliberately, an explicit fsync decision).
- Dataset download + format parsers: SIFT1M (fvecs/ivecs, standard base/query/learn split) and GloVe (format TBD — confirm the actual distribution format of the 2024 release before writing this parser; see §1–2 item 7).
- Manual verification only at this stage: `psql` in, `CREATE EXTENSION vector`, build one small HNSW index by hand. No Python automation yet — that's stage 3.
- Worth starting early since everything else depends on it, but a day or two of focused work either way, not an emergency.

**Stage 2 — Pilot experiments**
- This is where `m`, `ef_construction`, and `ef_search` actually get chosen and then frozen for the whole study (research plan §2.2) — not something to sweep later. Run a handful of manual, small-scale (e.g. 50K-vector) end-to-end passes: build → query → one churn round → VACUUM/REINDEX → query again.
- This is also where the RAM-risk and per-operation timing get sanity-checked — directly addressing the risk your own §3.3 already names, and cheap to catch here vs. discovering it mid-way through stage 4.
- Cross-check the brute-force ground-truth oracle against SIFT1M's own published ground truth on the untouched base set before trusting it after churn — the single highest-value correctness check given how easy recall is to get subtly wrong, and cheap to do now regardless of how the schedule is going.

**Stage 3 — Experiment automation**
Keep this to what actually lets stage 4 run unattended and survive a crash — not a general-purpose framework. That's about matching complexity to the project, not rationing hours:
- One Python script, one YAML config (dataset, seed, active size, replacement %, rounds, maintenance strategy, HNSW/query params). A plain dataclass with a few `assert`s at startup is enough validation for one config file the two of you both wrote and will both be debugging — a pydantic schema is more machinery for the same guarantee at this scale.
- Loop: load → baseline measure → (delete+insert → measure → [maintenance → measure]) × 5 rounds, per the exact procedure in research plan §2.3.2–2.3.3.
- Recall over the fixed 1,000-query subset (tie-tolerant, distance-threshold — §1–2 above), latency over the full 10,000, warm-up per whichever of the two documented procedures gets reconciled (§1–2 item 10).
- Incremental, flushed writes to CSV/JSON after every measurement point — a few lines of code, and what makes an unattended multi-hour run safe. This also directly implements your own QA commitment to "identify and repeat failed or incomplete runs" (research plan §2.4).
- Retry/backoff wrappers, elaborate pre-flight checks, and a full test suite are worth building for infrastructure meant to outlive one project — here, the ground-truth cross-check from stage 2 plus crash-safe incremental writes cover the risks that actually matter.

**Stage 4 — Full experiments**
- Run all maintenance conditions (4 or 5 — confirm which, §1–2 item 8) × both datasets × ≥3 repetitions, on the single agreed-upon machine (§1–2 item 11).
- "Check for failed or incomplete runs" (research plan §3.1) = watch `status.json`; a repetition that didn't finish gets restarted from scratch, not resumed mid-way — the simplest thing that satisfies your own stated QA commitment.

Stages 5 (analysis) and 6 (report) aren't implementation work, but the incremental CSV/JSON layout in §6 is built so stage 5 is "load these files into pandas," not another engineering task.

**Streamlit UI**: not part of the research plan or its stages at all. Build it only if you want the nicety and have real slack after the report is in good shape — it doesn't help answer RQ1/RQ2 or get the write-up done.

---

## 9. Things to deliberately not build

- Two separate client **processes** for queries vs. updates (§3) — one process, two DB connections at most.
- Any task queue / workflow engine (Celery, Airflow, Prefect) — a plain Python loop with try/except and file checkpoints is enough for one sequential pipeline on one machine.
- A database (Postgres table, SQLite) for storing *results* — flat CSV/JSON is simpler, human-inspectable, and trivially loadable into pandas.
- A REST API between the future Streamlit UI and the runner — direct subprocess + polling result files is enough for a single-user local tool.
- Any custom HNSW implementation, or reaching into pgvector internals — it's the system under test, treat it as a black box.
- Abstraction for "support other vector extensions/databases" — this study is pgvector-specific by design.
- A metrics/observability stack (Prometheus/Grafana) — files + Streamlit polling suffice for monitoring one overnight run.
- Multi-host/distributed execution — explicitly out of scope by the experiment's own design (single physical host).
- Fine-grained sub-cycle resumability (§7) — repetition-level restart is enough.
- Concurrent query/update execution — out of scope unless deliberately promoted to its own Phase 2 research question (§3).
- An ORM — raw SQL via `psycopg` (v3) keeps exactly what's executed, and when, transparent, which matters for a timing-sensitive benchmark.
- A generic "pluggable maintenance strategy" abstraction — there are ~4–5 known strategies; a dict/enum dispatch is enough.

**[proposal/plan] Additional cuts, sized to a 2-person course project rather than a long-lived research platform:**
- VACUUM FULL and REINDEX CONCURRENTLY as extra maintenance conditions — technically interesting, but outside the approved 4-condition scope (§1–2 item 8); adding them multiplies the run matrix for a comparison that isn't part of either RQ.
- A config validation framework (pydantic/JSON Schema) — a dataclass plus a few `assert`s is enough for one config file authored by the two people who'll also be debugging it.
- Retry/backoff wrappers and elaborate pre-flight checks — the kind of thing worth building for infrastructure meant to outlive one project; here, crash-safe incremental writes plus restarting a failed repetition from scratch cover the actual risk.
- A broader unit test suite — keep only the ground-truth-vs-published-ground-truth cross-check (stage 2); it's the one bug class that would silently invalidate results.
- The Streamlit UI — not in the research plan or its stages. Build it only if stage 4 and the report are in good shape and you want the nicety.

---

## 10. Suggested stack

Python: `psycopg` (v3, not v2), `numpy`, `faiss-cpu` (ground truth + reading ann-benchmarks-style HDF5/vecs formats conveniently), `PyYAML`, `pandas` (analysis side only). Docker Compose for the Postgres+pgvector container. **[proposal/plan]** Skip `pydantic` — a plain `dataclass` plus a few `assert`s at startup is enough validation for one config file authored by the two people also debugging it. Streamlit is out of scope for now (§8/§9) — revisit only if stage 4 and the report both finish with time to spare.

---

## 11. Open decisions to settle before coding starts

Updated against the actual proposal/research plan — several of the original open questions are now resolved by those documents; what's left is sharper and more concrete.

**Resolved by your own documents (no longer open):**
- ~~`ef_search` single value vs. swept list~~ → single fixed value, chosen during stage-2 piloting and then frozen (research plan §2.2) — not swept.
- ~~Whether two client processes are needed~~ → no, one benchmark client (research plan §2.3.1).
- ~~How much engineering polish is worth it~~ → minimal, but because of project size and scope, not an imminent deadline — the dated milestones are course scaffolding, not a countdown. See §8/§9.

**Still open, now sharper:**

1. **Maintenance conditions: 4 or 5?** Proposal has 5 (adds "periodic maintenance"), research plan has 4. Confirm which is the operative scope for your course deliverable before it's encoded into the config schema — recommend defaulting to the research plan's 4 in the meantime.
2. **Churn selection scheme and what "10% of the dataset" means** (§1–2 item 6): uniform-random-over-current-active-set (simpler, "round number" ≠ "% of original replaced") vs. stratified/disjoint blocks (guarantees full turnover, matches the proposal's "all initially active vectors are eventually replaced" claim, but needs the block size worked out from whichever reading of "10%" is intended). This is the single most consequential ambiguity to resolve before writing `update_runner.py`, since it changes both the code and how "round" is interpreted as an x-axis variable in the RQ1 plots.
3. **GloVe query/train split**: once the actual distribution format of the 2024 GloVe Wikipedia+Gigaword release is confirmed (§1–2 item 7), decide exactly how the disjoint active/replacement/query partition is carved out of the 1.2M pool — there's no upstream split to inherit the way there is for SIFT1M.
4. **Distance metric for GloVe**: cosine is conventional but isn't written down anywhere in either document — needs to be an explicit choice, recorded in the config and the eventual report.
5. **Which single machine hosts the stage-4 data-collection runs** (§1–2 item 11) — a coordination decision between Pavlo and Arnar, not a technical one, but worth settling before either of you starts a "real" run on your own laptop.
6. **Warm-up procedure**: reconcile the proposal's "run twice, discard first" against the research plan's "fixed warm-up workload" (§1–2 item 10) — the plan currently implements the former.
7. **Ground-truth recompute cost**: much lower-risk now that it's known to be over a 1,000-query subset rather than the full 10,000, but still worth the stage-2 timing check before assuming stage 4's 32-hour budget comfortably fits the full matrix.
8. **Whether concurrent query/update interference is a real future goal** or fully out of scope — low priority given the timeline, but affects whether `db.py`'s connection handling is worth keeping easy to extend later.

---

## Verification approach

Each stage in §8 has its own concrete check: manual `psql`/`CREATE EXTENSION vector`/hand-built HNSW index for stage 1; the brute-force-vs-published-SIFT1M-ground-truth comparison plus RAM/timing sanity checks for stage 2; inspecting real output files from a small end-to-end pass, and a deliberate crash-and-restart test of the resume logic, for stage 3; watching `status.json` and re-running any repetition that didn't complete, plus a face-validity check that recall/latency trends move the way the hypotheses predict, for stage 4. No implementation work starts until this plan is approved — the first concrete deliverable is stage 1 (Docker/Postgres/pgvector environment).
