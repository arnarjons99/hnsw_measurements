"""Descriptive analysis of run0: metrics vs churn round, recovery by maintenance, maintenance cost.

The repetition (n=3) is the unit of replication; no significance tests are applied.
Outputs (results/aj/analysis/): summary_table.csv, recovery.csv, maintenance_cost.csv,
<metric>_vs_round.png.
"""
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RUN_DIR = "results/aj/run0"
OUT = Path("results/aj/analysis")
OUT.mkdir(parents=True, exist_ok=True)
STRATS = ["vacuum", "reindex", "reindex + vacuum"]
ORDER = ["baseline"] + STRATS
DATASETS = ["glove", "sift1m"]
METRICS = {"recall_at_10": "Recall@10", "latency_p50_ms": "p50 latency (ms)",
           "latency_p95_ms": "p95 latency (ms)", "index_mb": "Index size (MiB)",
           "dead_tuples": "Dead tuples"}


def load():
    frames = []
    for f in sorted(glob.glob(f"{RUN_DIR}/*/measurements.csv")):
        dataset, strategy = Path(f).parent.name.removeprefix("run0_").split("_", 1)
        frames.append(pd.read_csv(f).assign(dataset=dataset, strategy=strategy.replace("_", " + ")))
    d = pd.concat(frames, ignore_index=True)
    d["index_mb"] = d.index_size_bytes / 2**20
    return d


df = load()
LAST = int(df["round"].max())


def series(ds, strat):
    """Rounds 0..LAST per repetition: round 0 baseline, then post_churn (no maintenance) or post_maintenance."""
    phase = "post_churn" if strat == "baseline" else "post_maintenance"
    s = df[(df.dataset == ds) & (df.strategy == strat) & (df.phase.isin(["baseline", phase]))]
    return s.sort_values(["repetition", "round"])


def mmm(v, prefix):
    return {f"{prefix}_mean": v.mean(), f"{prefix}_min": v.min(), f"{prefix}_max": v.max()}


# ---- Round 0 vs final round, per metric (RQ1, and RQ2 values) ----------------------------
summ = []
for ds in DATASETS:
    for s in ORDER:
        ser = series(ds, s)
        r0 = ser[ser["round"] == 0].set_index("repetition")
        rl = ser[ser["round"] == LAST].set_index("repetition")
        for col in METRICS:
            summ.append(dict(dataset=ds, strategy=s, metric=col, n_reps=len(rl),
                             **mmm(r0[col], "round0"), **mmm(rl[col], f"round{LAST}"),
                             **mmm(rl[col] - r0[col], "change")))
pd.DataFrame(summ).to_csv(OUT / "summary_table.csv", index=False)

# ---- Recovery: recall regained vs no maintenance (primary), and as share of recall lost ---
# Graph builds are independent across runs, so repetitions are not paired; the conservative
# range is strategy min - baseline max .. strategy max - baseline min.
rec = []
for ds in DATASETS:
    final = {s: series(ds, s).query("round == @LAST").recall_at_10 for s in ORDER}
    r0 = series(ds, "baseline").query("round == 0").recall_at_10.mean()
    base = final["baseline"]
    lost = r0 - base.mean()
    for s in STRATS:
        reg = final[s].mean() - base.mean()
        rec.append(dict(dataset=ds, strategy=s, round=LAST, recall_round0=r0,
                        recall_no_maintenance=base.mean(), recall_strategy=final[s].mean(),
                        recall_strategy_min=final[s].min(), recall_strategy_max=final[s].max(),
                        recall_lost=lost, recall_regained=reg,
                        recall_regained_min=final[s].min() - base.max(),
                        recall_regained_max=final[s].max() - base.min(),
                        recovery=reg / lost))
pd.DataFrame(rec).to_csv(OUT / "recovery.csv", index=False)

# ---- Maintenance cost (seconds, per repetition) ------------------------------------------
cost = []
for ds in DATASETS:
    for s in STRATS:
        m = df[(df.dataset == ds) & (df.strategy == s) & (df.phase == "post_maintenance")]
        per_rep = m.assign(total_s=m.vacuum_s.fillna(0) + m.reindex_s.fillna(0)).groupby("repetition")
        per_round, total = per_rep.total_s.mean(), per_rep.total_s.sum()
        cost.append(dict(dataset=ds, strategy=s, **mmm(per_round, "per_round_s"), **mmm(total, "total_s")))
pd.DataFrame(cost).to_csv(OUT / "maintenance_cost.csv", index=False)

# ---- Figures: mean over repetitions with min-max band ------------------------------------
# Distinct colour (Okabe-Ito), marker and line style per strategy so grayscale prints stay readable.
style = {"baseline": dict(color="#000000", marker="o", ls="-"),
         "vacuum": dict(color="#0072B2", marker="s", ls="--"),
         "reindex": dict(color="#E69F00", marker="^", ls=":"),
         "reindex + vacuum": dict(color="#009E73", marker="D", ls="-.")}
labels = {"baseline": "Baseline: no maintenance (post-churn)", "vacuum": "VACUUM (post-maintenance)",
          "reindex": "REINDEX (post-maintenance)", "reindex + vacuum": "REINDEX + VACUUM (post-maintenance)"}


def draw(ax, ds, col):
    for s in ORDER:
        piv = series(ds, s).pivot(index="round", columns="repetition", values=col)
        ax.fill_between(piv.index, piv.min(axis=1), piv.max(axis=1), color=style[s]["color"], alpha=0.15, lw=0)
        ax.plot(piv.index, piv.mean(axis=1), lw=1.8, markersize=5, label=labels[s], **style[s])
    ax.grid(alpha=0.25)


def figure(col, ylabel):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for ax, ds in zip(axes, DATASETS):
        draw(ax, ds, col)
        ax.set(title={"glove": "GloVe", "sift1m": "SIFT1M"}[ds], xlabel="Churn round", ylabel=ylabel,
               xticks=range(0, LAST + 1, 2))
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=12, frameon=False)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    name = col.replace("_at_10", "").replace("latency_", "")
    fig.savefig(OUT / f"{name}_vs_round.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


for col, ylabel in METRICS.items():
    if col != "dead_tuples":  # deterministic (50k per round); table only
        figure(col, ylabel)

for f in ("recovery", "maintenance_cost"):
    print(f"\n{f}")
    print(pd.read_csv(OUT / f"{f}.csv").to_string(index=False, float_format=lambda v: f"{v:.4g}"))
print(f"\nsaved outputs to {OUT}")
