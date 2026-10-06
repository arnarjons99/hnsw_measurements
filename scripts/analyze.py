"""Descriptive analysis of run0: metrics vs churn round, recovery by maintenance, maintenance cost.

The repetition (n=3) is the unit of replication; no significance tests are applied.
Outputs (results/aj/analysis/): summary_table.csv, recovery.csv, maintenance_cost.csv, p95_across_rounds.csv,
<metric>_vs_round.png, recovery_cost.png, round0_vs_round10.tex,
maintenance_table.tex.
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

# ---- p95 latency across rounds: min/max/avg of the per-round mean over repetitions --------
p95 = []
for ds in DATASETS:
    for s in ORDER:
        per_round = series(ds, s).groupby("round").latency_p95_ms.mean()
        p95.append(dict(dataset=ds, strategy=s, min_ms=per_round.min(), max_ms=per_round.max(),
                        avg_ms=per_round.mean(), round_of_min=per_round.idxmin(),
                        round_of_max=per_round.idxmax()))
pd.DataFrame(p95).to_csv(OUT / "p95_across_rounds.csv", index=False)

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

def recovery_cost_figure():
    """Cost-effectiveness scatter: time per round vs recall regained; bars = min-max over repetitions."""
    rec_df, cost_df = pd.read_csv(OUT / "recovery.csv"), pd.read_csv(OUT / "maintenance_cost.csv")
    names = {"glove": "GloVe", "sift1m": "SIFT1M"}
    short = {"vacuum": "VACUUM", "reindex": "REINDEX", "reindex + vacuum": "REINDEX + VACUUM"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for ax, ds in zip(axes, DATASETS):
        r = rec_df[rec_df.dataset == ds].set_index("strategy").loc[STRATS]
        c = cost_df[cost_df.dataset == ds].set_index("strategy").loc[STRATS]
        for s in STRATS:
            x, y = c.loc[s, "per_round_s_mean"], r.loc[s, "recall_regained"]
            ax.errorbar(x, y, xerr=[[x - c.loc[s, "per_round_s_min"]], [c.loc[s, "per_round_s_max"] - x]],
                        yerr=[[y - r.loc[s, "recall_regained_min"]], [r.loc[s, "recall_regained_max"] - y]],
                        color=style[s]["color"], marker=style[s]["marker"], ms=7, capsize=4, lw=1.2, alpha=0.85,
                        label=short[s])
        ax.set(title=names[ds], xlabel="Maintenance time per round (s)", ylabel="Recall@10 gain over no maintenance")
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=12, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(OUT / "recovery_cost.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


recovery_cost_figure()
def round0_vs_round10_table():
    """LaTeX table of round 0 vs final round (recall, p95, index size); round-final cells are mean (min-max)."""
    d = pd.read_csv(OUT / "summary_table.csv").set_index(["dataset", "strategy", "metric"])
    names = {"glove": "GloVe", "sift1m": "SIFT1M"}
    rows = {"baseline": "No maintenance", "vacuum": "VACUUM", "reindex": "REINDEX",
            "reindex + vacuum": "REINDEX + VACUUM"}
    cols = [("recall_at_10", 3, True), ("latency_p95_ms", 2, True), ("index_mb", 0, False)]
    out = [r"\begin{table}[H]", r"\centering", r"\footnotesize",
           r"\caption{Round 0 versus round %d. Round 0 shows the mean; round %d shows mean over three "
           r"repetitions, with min--max ranges for recall and p95.}" % (LAST, LAST), r"\label{tab:round0-vs-round%d}" % LAST,
           r"\begin{tabular}{l" + "cc" * len(cols) + "}", r"\hline",
           r"& \multicolumn{2}{c}{\textbf{Recall@10}} & \multicolumn{2}{c}{\textbf{p95 (ms)}} "
           r"& \multicolumn{2}{c}{\textbf{Index size (MiB)}} \\",
           r"\textbf{Strategy} & Round 0 & Round %d & Round 0 & Round %d & Round 0 & Round %d \\" % (LAST, LAST, LAST)]
    for ds in DATASETS:
        out += [r"\hline", r"\multicolumn{7}{l}{\textit{%s}} \\" % names[ds]]
        for s, label in rows.items():
            cells = [label]
            for m, nd, rng in cols:
                r = d.loc[(ds, s, m)]
                cells.append(f"{r.round0_mean:.{nd}f}")
                last = f"{r[f'round{LAST}_mean']:.{nd}f}"
                if rng:
                    last += f" ({r[f'round{LAST}_min']:.{nd}f}--{r[f'round{LAST}_max']:.{nd}f})"
                cells.append(last)
            out.append(" & ".join(cells) + r" \\")
    out += [r"\hline", r"\end{tabular}", r"\end{table}"]
    (OUT / "round0_vs_round10.tex").write_text("\n".join(out) + "\n")


round0_vs_round10_table()

def maintenance_table():
    """LaTeX table of recall gain, % of lost recall regained, and maintenance time per strategy."""
    rec_df, cost_df = pd.read_csv(OUT / "recovery.csv"), pd.read_csv(OUT / "maintenance_cost.csv")
    names = {"glove": "GloVe", "sift1m": "SIFT1M"}
    short = {"vacuum": "VACUUM", "reindex": "REINDEX", "reindex + vacuum": "REINDEX + VACUUM"}

    def mmm_cell(r, p, nd):
        return f"{r[p + '_mean']:.{nd}f} ({r[p + '_min']:.{nd}f}--{r[p + '_max']:.{nd}f})"

    out = [r"\begin{table}[H]", r"\centering", r"\footnotesize",
           r"\caption{Maintenance effectiveness and cost after %d churn rounds. Recall gain is relative to no "
           r"maintenance at round %d; \%% regained is the fraction of recall loss recovered. Values are means over "
           r"three repetitions; maintenance times include min--max ranges.}" % (LAST, LAST),
           r"\label{tab:maintenance}", r"\begin{tabular}{lcccc}", r"\hline",
           r"\textbf{Strategy} & \textbf{Recall@10 gain} & \textbf{\% regained} & "
           r"\textbf{Maintenance/round (s)} & \textbf{Total maintenance (s)} \\"]
    for ds in DATASETS:
        r0 = rec_df[rec_df.dataset == ds].iloc[0]
        out += [r"\hline", r"\multicolumn{5}{l}{\textit{%s} (recall lost without maintenance: %.3f)} \\"
                % (names[ds], r0.recall_lost)]
        for s in STRATS:
            r = rec_df[(rec_df.dataset == ds) & (rec_df.strategy == s)].iloc[0]
            c = cost_df[(cost_df.dataset == ds) & (cost_df.strategy == s)].iloc[0]
            gain = f"{r.recall_regained:.3f}"
            out.append(" & ".join([short[s], gain, f"{100 * r.recovery:.0f}\\%",
                                   mmm_cell(c, "per_round_s", 0), mmm_cell(c, "total_s", 0)]) + r" \\")
    out += [r"\hline", r"\end{tabular}", r"\end{table}"]
    (OUT / "maintenance_table.tex").write_text("\n".join(out) + "\n")


maintenance_table()

for f in ("recovery", "maintenance_cost"):
    print(f"\n{f}")
    print(pd.read_csv(OUT / f"{f}.csv").to_string(index=False, float_format=lambda v: f"{v:.4g}"))
print(f"\nsaved outputs to {OUT}")
