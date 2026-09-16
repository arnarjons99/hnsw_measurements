"""Live monitor + launcher, reading results/ and subprocessing the runner.
Purely additive - no changes to the experiment/runner logic. See PLAN.md
Sec 4/6/8 ("later milestone; reads results/, can subprocess the runner").

Run: .venv/bin/streamlit run ui/streamlit_app.py
"""

import json
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd
import psutil
import streamlit as st
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from hnsw_bench.config import ExperimentConfig, MAINTENANCE_CONDITIONS, load_config  # noqa: E402
from hnsw_bench.db import connect  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
CONFIGS_DIR = REPO_ROOT / "configs" / "experiments"

st.set_page_config(page_title="pgvector Benchmark", layout="wide")


def list_configs() -> list[Path]:
    return sorted(CONFIGS_DIR.glob("*.yaml"))


def list_runs() -> list[Path]:
    if not RESULTS_DIR.exists():
        return []
    return sorted((p for p in RESULTS_DIR.iterdir() if p.is_dir()), reverse=True)


def read_status(run_dir: Path) -> dict | None:
    path = run_dir / "status.json"
    return json.loads(path.read_text()) if path.exists() else None


def read_measurements(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "measurements.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def read_run_config(run_dir: Path) -> ExperimentConfig | None:
    """The config actually used for this run (written by ResultsWriter),
    not whatever the sidebar happens to have selected right now."""
    path = run_dir / "config.yaml"
    return ExperimentConfig(**yaml.safe_load(path.read_text())) if path.exists() else None


def checkpoints_per_repetition(config) -> int:
    return 1 + config.rounds * (1 if config.maintenance == "none" else 2)


def is_running(run_dir: Path) -> int | None:
    """Returns the runner's pid if a launch of it is still alive, else None.
    Reaps a finished child via wait(timeout=0) - pid_exists() alone stays
    True for a zombie until its parent (this app) reaps it."""
    pid_file = run_dir / "runner.pid"
    if not pid_file.exists():
        return None
    pid = int(pid_file.read_text())
    if not psutil.pid_exists(pid):
        return None
    try:
        psutil.Process(pid).wait(timeout=0)
        return None
    except psutil.TimeoutExpired:
        return pid
    except psutil.NoSuchProcess:
        return None


def postgres_status() -> tuple[bool, str]:
    try:
        conn = connect()
        version = conn.execute("select version()").fetchone()[0].split(",")[0]
        pgvector = conn.execute(
            "select extversion from pg_extension where extname = 'vector'"
        ).fetchone()
        conn.close()
        return True, f"{version} / pgvector {pgvector[0] if pgvector else '?'}"
    except Exception as e:
        return False, str(e)


def render_system_status() -> None:
    st.sidebar.subheader("System status")
    ok, detail = postgres_status()
    st.sidebar.markdown(f"{'🟢' if ok else '🔴'} **PostgreSQL** — {detail}")
    st.sidebar.progress(psutil.cpu_percent() / 100, text=f"CPU {psutil.cpu_percent()}%")
    mem = psutil.virtual_memory()
    st.sidebar.progress(mem.percent / 100, text=f"Memory {mem.percent}%")
    disk = psutil.disk_usage(str(REPO_ROOT))
    st.sidebar.progress(disk.percent / 100, text=f"Disk {disk.percent}%")


def render_metrics(df: pd.DataFrame) -> None:
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else None

    def delta(col: str, fmt: str = "{:.3f}") -> str | None:
        if prev is None or pd.isna(last[col]) or pd.isna(prev[col]):
            return None
        return fmt.format(last[col] - prev[col])

    cols = st.columns(5)
    cols[0].metric("Recall@10", f"{last['recall_at_10']:.3f}", delta("recall_at_10"))
    cols[1].metric(
        "Latency p50/p95 (ms)",
        f"{last['latency_p50_ms']:.1f} / {last['latency_p95_ms']:.1f}",
    )
    cols[2].metric(
        "Dead tuples", f"{int(last['dead_tuples']):,}", delta("dead_tuples", "{:+.0f}")
    )
    cols[3].metric(
        "Table size (MB)",
        f"{last['table_size_bytes'] / 1e6:.1f}",
        delta("table_size_bytes", "{:+.1f}"),
    )
    cols[4].metric(
        "Index size (MB)",
        f"{last['index_size_bytes'] / 1e6:.1f}",
    )


def render_charts(df: pd.DataFrame) -> None:
    df = df.reset_index(drop=True)
    df["checkpoint"] = df.index
    c1, c2, c3 = st.columns(3)
    with c1:
        st.caption("Recall@10 vs checkpoint")
        st.line_chart(df, x="checkpoint", y="recall_at_10")
    with c2:
        st.caption("Latency (ms) vs checkpoint")
        st.line_chart(df, x="checkpoint", y=["latency_p50_ms", "latency_p95_ms"])
    with c3:
        st.caption("Size (MB) vs checkpoint")
        sized = df.assign(
            table_mb=df["table_size_bytes"] / 1e6, index_mb=df["index_size_bytes"] / 1e6
        )
        st.line_chart(sized, x="checkpoint", y=["table_mb", "index_mb"])


st.sidebar.title("pgvector Benchmark")
page = st.sidebar.radio("View", ["Run experiment", "Browse results"])

st.sidebar.divider()
st.sidebar.subheader("Launch a run")
config_path = st.sidebar.selectbox(
    "Config", list_configs(), format_func=lambda p: p.name
)
base_config = load_config(config_path) if config_path else None

launch_config = base_config
if base_config:
    with st.sidebar.expander("Override parameters"):
        st.caption(
            f"m={base_config.m}, ef_construction={base_config.ef_construction}, "
            f"ef_search={base_config.ef_search}, k={base_config.k} - frozen from "
            "the stage-2 pilot, not editable here (PLAN.md Sec 11)."
        )
        active_size = st.number_input(
            "Active vectors", min_value=1, step=1000, value=base_config.active_size
        )
        churn_pct = st.number_input(
            "Replacement %/round",
            min_value=0.1,
            max_value=100.0,
            value=float(base_config.churn_pct),
        )
        rounds = st.number_input("Rounds", min_value=1, step=1, value=base_config.rounds)
        repetitions = st.number_input(
            "Repetitions", min_value=1, step=1, value=base_config.repetitions
        )
        maintenance_options = sorted(MAINTENANCE_CONDITIONS)
        maintenance = st.selectbox(
            "Maintenance",
            maintenance_options,
            index=maintenance_options.index(base_config.maintenance),
        )
        n_recall_queries = st.number_input(
            "Recall query count",
            min_value=1,
            step=100,
            value=base_config.n_recall_queries,
        )
        n_latency_queries = st.number_input(
            "Latency query count",
            min_value=1,
            step=500,
            value=base_config.n_latency_queries,
        )
        if n_recall_queries > n_latency_queries:
            st.error("Recall query count can't exceed the latency query count.")
            launch_config = None
        else:
            launch_config = replace(
                base_config,
                active_size=int(active_size),
                churn_pct=float(churn_pct),
                rounds=int(rounds),
                repetitions=int(repetitions),
                maintenance=maintenance,
                n_recall_queries=int(n_recall_queries),
                n_latency_queries=int(n_latency_queries),
            )
            n_checkpoints = checkpoints_per_repetition(launch_config)
            st.caption(
                f"→ {launch_config.repetitions} rep(s) x {n_checkpoints} checkpoints = "
                f"{launch_config.repetitions * n_checkpoints} checkpoints total"
            )

default_run_id = (
    f"{launch_config.dataset}_{launch_config.maintenance}_ui" if launch_config else ""
)
run_id = st.sidebar.text_input("Run ID", value=default_run_id)
run_dir = RESULTS_DIR / run_id if run_id else None

active_pid = is_running(run_dir) if run_dir else None
start_disabled = launch_config is None or not run_id or active_pid is not None
if st.sidebar.button("▶ Start experiment", disabled=start_disabled, type="primary"):
    run_dir.mkdir(parents=True, exist_ok=True)
    launch_config_path = run_dir / "launch_config.yaml"
    launch_config_path.write_text(yaml.safe_dump(asdict(launch_config)))
    log_path = run_dir / "runner.log"
    with open(log_path, "w") as log_file:
        proc = subprocess.Popen(
            [sys.executable, "-m", "hnsw_bench.run", str(launch_config_path), "--run-id", run_id],
            cwd=REPO_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    (run_dir / "runner.pid").write_text(str(proc.pid))
    st.session_state.watching = run_id
    st.rerun()

if active_pid is not None and st.sidebar.button("■ Stop experiment"):
    psutil.Process(active_pid).terminate()
    (run_dir / "runner.pid").unlink(missing_ok=True)
    st.rerun()

render_system_status()

if page == "Run experiment":
    watching = st.session_state.get("watching") or run_id
    watch_dir = RESULTS_DIR / watching if watching else None

    if not watch_dir or not watch_dir.exists():
        st.info("Pick a config and click **Start experiment** to begin.")
    else:
        status = read_status(watch_dir)
        pid = is_running(watch_dir)
        df = read_measurements(watch_dir)
        run_config = read_run_config(watch_dir)

        if pid is not None:
            st.success(f"🟢 Running - `{watching}` (pid {pid})")
        elif status:
            st.info(f"⚪ Not running - `{watching}` (last state shown below)")

        if status:
            total_reps = run_config.repetitions if run_config else "?"
            if status.get("current_repetition") is not None:
                rep_num = status["current_repetition"] + 1
                progress_label = f"round {status['current_round']} - {status['current_phase']}"
            else:
                completed = status["completed_repetitions"]
                rep_num = completed[-1] + 1 if completed else 0
                progress_label = "finishing up..." if pid is not None else "complete"
            st.subheader(f"Repetition {rep_num}/{total_reps} - {progress_label}")
            if run_config and not df.empty:
                total_checkpoints = run_config.repetitions * checkpoints_per_repetition(
                    run_config
                )
                st.progress(min(len(df) / total_checkpoints, 1.0))

        if not df.empty:
            current_rep = status.get("current_repetition") if status else None
            if current_rep is None and status and status["completed_repetitions"]:
                current_rep = status["completed_repetitions"][-1]
            rep_df = df[df["repetition"] == current_rep] if current_rep is not None else df
            render_metrics(rep_df if not rep_df.empty else df)
            render_charts(rep_df if not rep_df.empty else df)

        log_path = watch_dir / "runner.log"
        if log_path.exists():
            st.subheader("Experiment log")
            lines = log_path.read_text().splitlines()[-30:]
            st.code("\n".join(lines) or "(no output yet)", language="text")

        if pid is not None:
            time.sleep(2)
            st.rerun()

else:
    runs = list_runs()
    if not runs:
        st.info("No runs under results/ yet.")
    else:
        chosen = st.selectbox("Run", runs, format_func=lambda p: p.name)
        df = read_measurements(chosen)
        cfg_path = chosen / "config.yaml"
        meta_path = chosen / "metadata.json"

        with st.expander("Config / metadata"):
            c1, c2 = st.columns(2)
            if cfg_path.exists():
                c1.code(cfg_path.read_text(), language="yaml")
            if meta_path.exists():
                c2.json(json.loads(meta_path.read_text()))

        if df.empty:
            st.warning("No measurements recorded for this run yet.")
        else:
            reps = sorted(df["repetition"].unique())
            rep_choice = st.selectbox("Repetition", ["All"] + list(reps))
            rep_df = df if rep_choice == "All" else df[df["repetition"] == rep_choice]
            render_charts(rep_df)
            st.dataframe(df, use_container_width=True)
            st.download_button(
                "Download measurements.csv",
                df.to_csv(index=False),
                file_name=f"{chosen.name}_measurements.csv",
            )
