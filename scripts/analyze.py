"""Analysis of run0: recall/latency/index size vs churn round, and recovery by VACUUM/REINDEX.

Unit of replication is the repetition (n=3); rounds within a repetition are never treated as
independent. No multiple-comparison correction is applied (raw p-values only).
Outputs (results/aj/analysis/): baseline_slope_results.csv, final_round_comparisons.csv, summary_table.csv,
recall_vs_round.{png,pdf}.
"""
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

RUN_DIR = "results/aj/run0"
OUT = Path("results/aj/analysis")
OUT.mkdir(parents=True, exist_ok=True)
STRATS = ["vacuum", "reindex", "reindex + vacuum"]
ORDER = ["baseline"] + STRATS
DATASETS = ["glove", "sift1m"]


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
    """Per-repetition rows for rounds 0..LAST: round 0 baseline, then post_churn (baseline)
    or post_maintenance (maintenance strategies)."""
    phase = "post_churn" if strat == "baseline" else "post_maintenance"
    s = df[(df.dataset == ds) & (df.strategy == strat) & (df.phase.isin(["baseline", phase]))]
    return s.sort_values(["repetition", "round"])


def mean_ci(v):
    v = np.asarray(v, float)
    sd = v.std(ddof=1)
    h = stats.t.ppf(0.975, len(v) - 1) * sd / np.sqrt(len(v))
    return v.mean(), sd, v.mean() - h, v.mean() + h


# ---- Statistical tests -------------------------------------------------------------------
def fmt(v):
    return "; ".join(f"{x:.6g}" for x in v)


# RQ1: baseline degradation. One OLS slope of recall@10 on round per repetition (rounds 0..LAST),
# then a one-sample t-test of the 3 slopes against 0.
slope_rows = []
for ds in DATASETS:
    fits = [stats.linregress(g["round"], g.recall_at_10) for _, g in series(ds, "baseline").groupby("repetition")]
    slopes = [f.slope for f in fits]
    m, sd, lo, hi = mean_ci(slopes)
    t = stats.ttest_1samp(slopes, 0.0)
    slope_rows.append({
        "dataset": ds,
        "n_repetitions": len(slopes),
        "slope_per_repetition (recall@10 per round)": fmt(slopes),
        "mean_slope (recall@10 per round)": m,
        "sd_of_slopes": sd,
        "ci95_low": lo,
        "ci95_high": hi,
        "mean_R2_of_per_repetition_fits": np.mean([f.rvalue ** 2 for f in fits]),
        "t_statistic": t.statistic,
        "degrees_of_freedom": len(slopes) - 1,
        "p_value (uncorrected)": t.pvalue,
        "null_hypothesis": "mean slope = 0",
        "rounds_used": f"0-{LAST}",
    })
slope_res = pd.DataFrame(slope_rows)
slope_res.to_csv(OUT / "baseline_slope_results.csv", index=False)

# RQ2: end state after LAST rounds. Strategy (post_maintenance) vs baseline (post_churn),
# Welch t-test on the 3 repetition values per group.
final_rows = []
for ds in DATASETS:
    end = {s: series(ds, s).query("round == @LAST").sort_values("repetition").recall_at_10.values
           for s in ORDER}
    for s in STRATS:
        t = stats.ttest_ind(end[s], end["baseline"], equal_var=False)
        ci = t.confidence_interval()
        final_rows.append({
            "dataset": ds,
            "strategy": s,
            "round": LAST,
            "n_per_group": len(end[s]),
            "strategy_recall_per_repetition": fmt(end[s]),
            "strategy_mean_recall": end[s].mean(),
            "strategy_sd_recall": end[s].std(ddof=1),
            "baseline_recall_per_repetition": fmt(end["baseline"]),
            "baseline_mean_recall": end["baseline"].mean(),
            "baseline_sd_recall": end["baseline"].std(ddof=1),
            "mean_difference (strategy - baseline)": end[s].mean() - end["baseline"].mean(),
            "ci95_low": ci.low,
            "ci95_high": ci.high,
            "welch_t_statistic": t.statistic,
            "degrees_of_freedom": t.df,
            "p_value (uncorrected)": t.pvalue,
            "null_hypothesis": "mean difference = 0",
        })
final_res = pd.DataFrame(final_rows)
final_res.to_csv(OUT / "final_round_comparisons.csv", index=False)

for name, r in (("baseline slopes", slope_res), ("final-round comparisons", final_res)):
    print(f"\n{name}")
    print(r.drop(columns=[c for c in r.columns if "per_repetition" in c]).to_string(
        index=False, float_format=lambda v: f"{v:.4g}"))

# ---- Descriptive summary table -----------------------------------------------------------
metrics = {"recall_at_10": "recall@10", "latency_p50_ms": "p50_ms", "latency_p95_ms": "p95_ms",
           "index_mb": "index_MiB", "dead_tuples": "dead_tuples"}
summ = []
for ds in DATASETS:
    for s in ORDER:
        ser = series(ds, s)
        row_maint = df[(df.dataset == ds) & (df.strategy == s) & (df.phase == "post_maintenance")]
        per_rep = row_maint.groupby("repetition")[["vacuum_s", "reindex_s"]].mean()
        for r in (0, LAST):
            g = ser[ser["round"] == r]
            row = dict(dataset=ds, strategy=s, round=r, n_reps=len(g))
            for c, name in metrics.items():
                row[f"{name}_mean"], row[f"{name}_sd"] = g[c].mean(), g[c].std(ddof=1)
            for c in ("vacuum_s", "reindex_s"):
                if r == LAST and s != "baseline":
                    row[f"{c}_per_round_mean"], row[f"{c}_per_round_sd"] = per_rep[c].mean(), per_rep[c].std(ddof=1)
            summ.append(row)
summ = pd.DataFrame(summ)
summ.to_csv(OUT / "summary_table.csv", index=False)

# ---- Figures -----------------------------------------------------------------------------
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
        ax.plot(piv.index, piv.mean(axis=1), lw=1.8, markersize=5, label=labels[s], **style[s])
    ax.grid(alpha=0.25)


def figure(col, ylabel, fname):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for ax, ds in zip(axes, DATASETS):
        draw(ax, ds, col)
        ax.set(title={"glove": "GloVe", "sift1m": "SIFT1M"}[ds], xlabel="Churn round", ylabel=ylabel, xticks=range(0, LAST + 1, 2))
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=12, frameon=False)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{fname}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


figure("recall_at_10", "Recall@10", "recall_vs_round")
print(f"saved outputs to {OUT}")
