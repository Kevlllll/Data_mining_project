"""Step 7: blocked, station-grouped cross-validation with embargo, and
one-standard-error selection within each model family."""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from .models import BASELINES, build
from .splits import cv_cells


def development_rows(sup):
    """Development stations x development target days, complete rows only.
    Every candidate is scored on exactly these rows (common sample)."""
    m = (sup["role"] == "development") & (sup["date_role"] == "development") & sup["complete"]
    return sup[m].reset_index(drop=True)


def station_macro_mae(station_ids, err):
    return pd.Series(np.abs(err)).groupby(np.asarray(station_ids)).mean().mean()


def _score(cid, g, f, val, pred):
    err = pred - val["y"].to_numpy()
    return {"candidate_id": cid, "station_group": g, "day_fold": f, "n_val": len(val),
            "n_val_stations": val["station_id"].nunique(),
            "station_macro_mae": station_macro_mae(val["station_id"], err),
            "sum_abs": np.abs(err).sum(), "sum_sq": (err ** 2).sum(), "sum_err": err.sum()}


def _run_cell(dev, cands, g, f, val_st, val_days, excluded, keep_preds=False):
    is_val = dev["station_id"].isin(val_st) & dev["target_date"].isin(val_days)
    is_train = ~dev["station_id"].isin(val_st) & ~dev["target_date"].isin(excluded)
    tr, va = dev[is_train], dev[is_val]
    rows, preds = [], []

    def record(c, pred):
        rows.append(_score(c["id"], g, f, va, pred))
        if keep_preds:
            preds.append(pd.DataFrame({"candidate_id": c["id"], "station_id": va["station_id"],
                                       "target_date": va["target_date"], "y_true": va["y"],
                                       "y_pred": pred, "station_group": g, "day_fold": f}))

    knn = [c for c in cands if c["family"] == "knn"]
    for c in cands:
        if c["family"] != "knn":
            record(c, build(c).fit(tr[c["columns"]], tr["y"]).predict(va[c["columns"]]))
    # kNN: one neighbour search per feature spec, then average the first k neighbours
    by_spec = {}
    for c in knn:
        by_spec.setdefault(tuple(c["columns"]), []).append(c)
    ytr = tr["y"].to_numpy()
    for cols, group in by_spec.items():
        cols = list(cols)
        sc = StandardScaler().fit(tr[cols])
        kmax = max(c["params"]["k"] for c in group)
        nn = KNeighborsRegressor(n_neighbors=kmax).fit(sc.transform(tr[cols]), ytr)
        idx = nn.kneighbors(sc.transform(va[cols]), return_distance=False)
        for c in group:
            record(c, ytr[idx[:, :c["params"]["k"]]].mean(axis=1))
    return rows, preds


def run_cv(dev, cands, groups, days, cfg, keep_preds=False, n_jobs=-1):
    cells = list(cv_cells(groups, days, cfg))
    out = Parallel(n_jobs=n_jobs)(
        delayed(_run_cell)(dev, cands, g, f, vs, vd, ex, keep_preds) for g, f, vs, vd, ex in cells)
    scores = pd.DataFrame([r for rows, _ in out for r in rows])
    preds = pd.concat([p for _, ps in out for p in ps]) if keep_preds else None
    return scores, preds


def summarize(scores, cands):
    meta = pd.DataFrame([{"candidate_id": c["id"], "family": c["family"], "feature_set": c["fs"],
                          "p": c["p"], "n_features": len(c["columns"]),
                          "params": c["params"], "complexity": c["complexity"]} for c in cands])
    agg = scores.groupby("candidate_id").agg(
        cv_mae=("station_macro_mae", "mean"), cv_sd=("station_macro_mae", "std"),
        n_cells=("station_macro_mae", "size"), n=("n_val", "sum"),
        sum_abs=("sum_abs", "sum"), sum_sq=("sum_sq", "sum"), sum_err=("sum_err", "sum"))
    agg["cv_se"] = agg["cv_sd"] / np.sqrt(agg["n_cells"])
    agg["pooled_mae"] = agg["sum_abs"] / agg["n"]
    agg["pooled_rmse"] = np.sqrt(agg["sum_sq"] / agg["n"])
    agg["mean_error"] = agg["sum_err"] / agg["n"]
    agg = agg.drop(columns=["sum_abs", "sum_sq", "sum_err"]).reset_index()
    return meta.merge(agg, on="candidate_id")


def one_se_select(summary):
    """Per family: best = lowest CV MAE; selected = simplest candidate whose CV MAE
    is within one SE of the best (Class16 one-standard-error rule)."""
    rows = []
    for fam, s in summary.groupby("family"):
        best = s.loc[s["cv_mae"].idxmin()]
        within = s[s["cv_mae"] <= best["cv_mae"] + best["cv_se"]]
        chosen = within.sort_values(["complexity", "cv_mae"]).iloc[0]
        rows.append({"family": fam, "selected_id": chosen["candidate_id"],
                     "selected_cv_mae": chosen["cv_mae"], "selected_cv_se": chosen["cv_se"],
                     "best_id": best["candidate_id"], "best_cv_mae": best["cv_mae"],
                     "best_cv_se": best["cv_se"], "one_se_threshold": best["cv_mae"] + best["cv_se"],
                     "n_within_one_se": len(within)})
    return pd.DataFrame(rows).sort_values("selected_cv_mae").reset_index(drop=True)


# fewer tuning parameters first when the number of inputs ties
FAMILY_ORDER = ["ols", "ridge", "lasso", "pcr", "spline", "knn"]


def designate_primary(selection, summary):
    """One-SE rule again, across the families' selected specs: among those within
    one SE of the best selected CV MAE, take the one with the fewest model inputs
    (PCR counts its components), then the family with fewer tuning parameters."""
    s = selection[~selection["family"].isin(BASELINES)].merge(
        summary[["candidate_id", "n_features", "params"]], left_on="selected_id",
        right_on="candidate_id")
    s["n_inputs"] = [p.get("n_components", n) for p, n in zip(s["params"], s["n_features"])]
    best = s.loc[s["selected_cv_mae"].idxmin()]
    within = s[s["selected_cv_mae"] <= best["selected_cv_mae"] + best["selected_cv_se"]].copy()
    within["rank"] = within["family"].map(FAMILY_ORDER.index)
    return within.sort_values(["n_inputs", "rank"]).iloc[0]["family"]


def paired_differences(scores, ref_id, other_ids):
    """Compare candidates cell by cell (same validation rows). Positive mean_diff =
    the other candidate has LOWER error than the reference. Cells that share days
    are correlated, so the paired SE is approximate."""
    piv = scores.pivot_table(index=["station_group", "day_fold"], columns="candidate_id",
                             values="station_macro_mae")
    rows = []
    for oid in dict.fromkeys(other_ids):
        if oid == ref_id:
            continue
        d = piv[ref_id] - piv[oid]
        rows.append({"reference": ref_id, "other": oid, "mean_diff": d.mean(),
                     "paired_se": d.std() / np.sqrt(len(d)), "z": d.mean() / (d.std() / np.sqrt(len(d))),
                     "cells_other_better": int((d > 0).sum()), "n_cells": len(d)})
    return pd.DataFrame(rows)
