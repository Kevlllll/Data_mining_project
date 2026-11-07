"""Steps 8-9: freeze the selected models, then the one-time final test."""
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .models import build
from .validation import development_rows


def param_hash(model):
    return joblib.hash(model)


def fit_frozen(sup, selected, cand_by_id, model_dir):
    dev = development_rows(sup)
    meta = {}
    for fam, cid in selected.items():
        c = cand_by_id[cid]
        model = build(c).fit(dev[c["columns"]], dev["y"])
        h = param_hash(model)
        joblib.dump({"model": model, "candidate": c, "param_hash": h}, model_dir / f"{fam}.joblib")
        meta[fam] = {"candidate_id": cid, "columns": c["columns"], "params": c["params"],
                     "n_train_rows": len(dev), "n_train_stations": dev["station_id"].nunique(),
                     "train_target_dates": [str(dev["target_date"].min().date()),
                                            str(dev["target_date"].max().date())],
                     "param_hash": h}
    return meta


EVAL_CELLS = {
    "temporal_test": ("development", "test"),       # seen stations, unseen days
    "spatial_test": ("held_out", "development"),    # unseen stations, development days
    "spatiotemporal_test": ("held_out", "test"),    # unseen stations, unseen days
}


def predict_frozen(sup, frozen):
    """Frozen models only: no fit / partial_fit anywhere in this function."""
    parts = []
    for cell, (role, date_role) in EVAL_CELLS.items():
        rows = sup[(sup["role"] == role) & (sup["date_role"] == date_role)]
        n_all = len(rows)
        rows = rows[rows["complete"]]  # common keys for every model
        for fam, obj in frozen.items():
            c = obj["candidate"]
            pred = obj["model"].predict(rows[c["columns"]])
            if np.isnan(pred).any():
                raise RuntimeError(f"{fam} produced NaN predictions in {cell}")
            parts.append(pd.DataFrame({
                "station_id": rows["station_id"].to_numpy(), "origin_date": rows["origin_date"].to_numpy(),
                "target_date": rows["target_date"].to_numpy(), "split": cell, "model_id": fam,
                "test_season": rows["test_season"].to_numpy(), "y_true": rows["y"].to_numpy(),
                "y_pred": pred, "baseline_pred": rows["lag1"].to_numpy(),
                "n_rows_before_completeness": n_all}))
    out = pd.concat(parts, ignore_index=True)
    out["error"] = out["y_pred"] - out["y_true"]
    out["absolute_error"] = out["error"].abs()
    out["squared_error"] = out["error"] ** 2
    return out


def metrics(pred, by):
    def f(g):
        st = g.groupby("station_id")["absolute_error"].mean()
        base_st = (g["baseline_pred"] - g["y_true"]).abs().groupby(g["station_id"]).mean()
        return pd.Series({
            "n_rows": len(g), "n_stations": g["station_id"].nunique(),
            "station_macro_mae": st.mean(), "pooled_mae": g["absolute_error"].mean(),
            "rmse": np.sqrt(g["squared_error"].mean()), "mean_error": g["error"].mean(),
            "baseline_station_macro_mae": base_st.mean(),
            "improvement_vs_persistence": 1 - st.mean() / base_st.mean(),
        })
    return pred.groupby(by).apply(f, include_groups=False).reset_index()


def forward_only_check(sup, selected, cand_by_id, roles, season="fall"):
    """Refit on development rows whose target precedes the chosen test block,
    then score that block on development stations (a pure forward forecast)."""
    start = roles.index[roles["test_season"] == season].min()
    dev = development_rows(sup)
    dev = dev[dev["target_date"] < start]
    test = sup[(sup["role"] == "development") & (sup["test_season"] == season) & sup["complete"]]
    rows = []
    for fam, cid in selected.items():
        c = cand_by_id[cid]
        pred = build(c).fit(dev[c["columns"]], dev["y"]).predict(test[c["columns"]])
        err = np.abs(pred - test["y"].to_numpy())
        rows.append({"model_id": fam, "season": season, "train_rows": len(dev),
                     "station_macro_mae_forward_only": pd.Series(err).groupby(
                         test["station_id"].to_numpy()).mean().mean()})
    return pd.DataFrame(rows)


def error_vs_distance(pred, station_split, model_id, fig_path):
    st = (pred[(pred["model_id"] == model_id) & (pred["split"] == "spatiotemporal_test")]
          .groupby("station_id")["absolute_error"].mean().rename("mae").reset_index()
          .merge(station_split[["station_id", "nearest_dev_km"]], on="station_id"))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(st["nearest_dev_km"], st["mae"])
    ax.set(xlabel="distance to nearest development station (km)", ylabel="MAE (ppb)",
           title=f"Held-out stations, test weeks: {model_id}")
    fig.tight_layout(); fig.savefig(fig_path, dpi=150); plt.close(fig)
    return st
