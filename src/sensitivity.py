"""Sensitivity of the CV results to the history window (max lag = rolling window).

Development data only. The locked split (test weeks, embargo, stations) is
read from disk and never redrawn, and every window is scored on the same rows:
the development rows that are complete under the longest window tested.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from . import features, models, splits, validation

WINDOWS = [3, 5, 7, 10]
KEY = ["station_id", "target_date"]


def window_config(cfg, w):
    if w > cfg["embargo_days"]:
        raise ValueError("a window longer than the locked embargo would leak test-week values")
    return {**cfg, "max_lag_days": w, "rolling_window_days": w, "lag_sets": sorted({1, 3, w})}


def run(panel, station_split, roles, cfg, windows=WINDOWS):
    groups, days = splits.cv_layout(roles, station_split, cfg)
    longest = max(windows)
    main = validation.development_rows(
        features.build_supervised(panel, station_split, roles, window_config(cfg, longest)))
    scores, coverage = [], []
    for w in windows:
        cw = window_config(cfg, w)
        own = validation.development_rows(features.build_supervised(panel, station_split, roles, cw))
        coverage.append({"window": w, "complete_development_rows": len(own),
                         f"extra_rows_vs_{longest}_day": len(own) - len(main)})
        common = own.merge(main[KEY], on=KEY).sort_values(KEY).reset_index(drop=True)
        cands = models.candidates(cw, station_split)
        cell_scores, _ = validation.run_cv(common, cands, groups, days, cfg)
        summary = validation.summarize(cell_scores, cands)
        summary["window"] = w
        sel = validation.one_se_select(summary)
        sel["window"] = w
        sel["primary"] = sel["family"] == validation.designate_primary(sel, summary)
        sel["n_rows_scored"] = len(common)
        scores.append((summary, sel))
    summary = pd.concat([s for s, _ in scores], ignore_index=True)
    best = pd.concat([b for _, b in scores], ignore_index=True)
    return summary, best, pd.DataFrame(coverage)


def window_figure(summary, fig_path):
    """Best OLS CV MAE per feature set, one line per window, +-1 SE."""
    ols = summary[summary["family"] == "ols"]
    best = ols.loc[ols.groupby(["window", "feature_set"])["cv_mae"].idxmin()]
    pers = summary.loc[summary["family"] == "persistence", "cv_mae"].iloc[0]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, (w, g) in enumerate(best.groupby("window")):
        g = g.sort_values("feature_set")
        ax.errorbar([x + (i - 1) * 0.06 for x in range(len(g))], g["cv_mae"], yerr=g["cv_se"],
                    marker="o", capsize=3, label=f"{w}-day history window")
    ax.axhline(pers, c="k", ls="--", lw=1, label=f"persistence ({pers:.2f})")
    ax.set_xticks(range(4), ["F1", "F2", "F3", "F4"])
    ax.set(xlabel="feature set", ylabel="CV station-average MAE (ppb), lower is better",
           title="OLS: error by history window (best lag choice per point)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(fig_path, dpi=150); plt.close(fig)
