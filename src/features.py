"""Supervised rows: one per (station, target_date). Inputs use only days up to
the forecast origin t = target_date - 1, built on a calendar-complete index so
a missing day is never mistaken for 'yesterday'."""
import numpy as np
import pandas as pd

from .common import dates
from .splits import distance_matrix_km

DOY_PERIOD = 366  # 2024 is a leap year

FEATURE_DESCRIPTIONS = {
    "lag{k}": "MDA8 (ppb) on day t-{km1}, i.e. {k} day(s) before the target date",
    "roll_mean{R}": "mean MDA8 over t-{Rm1}..t (all {R} days required)",
    "roll_sd{R}": "SD (ddof=1) of MDA8 over t-{Rm1}..t (all {R} days required)",
    "doy_sin": "sin(2*pi*day_of_year(target_date)/366)",
    "doy_cos": "cos(2*pi*day_of_year(target_date)/366)",
    "latitude": "station latitude",
    "longitude": "station longitude",
    "cluster_{c}": "1 if station is in feature cluster {c} (cluster 0 is the reference)",
    "nbr_mean": "mean MDA8 on day t at the k nearest development stations reporting that day (self excluded)",
}


def wide_matrix(panel, stations, cfg):
    """dates x stations matrix of MDA8 over the full study window (NaN = missing)."""
    p = panel[panel["station_id"].isin(stations)]
    return (p.pivot(index="date", columns="station_id", values="mda8_ppb")
            .reindex(index=dates(cfg["study_start"], cfg["study_end"]), columns=stations))


def neighbor_mean(W, station_split, cfg):
    """Day-t mean over the k nearest *development* stations that reported on day t."""
    k = cfg["neighbor_k"]
    dev = station_split[station_split["role"] == "development"]
    D = distance_matrix_km(station_split["latitude"], station_split["longitude"],
                           dev["latitude"], dev["longitude"])
    Wdev = W[dev["station_id"]].to_numpy()
    out = np.full(W.shape, np.nan)
    for i, sid in enumerate(station_split["station_id"]):
        order = np.argsort(D[i])
        order = order[dev["station_id"].to_numpy()[order] != sid]  # exclude self
        V = Wdev[:, order]
        ok = ~np.isnan(V)
        take = ok & (np.cumsum(ok, axis=1) <= k)
        n = take.sum(axis=1)
        s = np.where(take, V, 0).sum(axis=1)
        out[:, W.columns.get_loc(sid)] = np.where(n == k, s / np.maximum(n, 1), np.nan)
    return pd.DataFrame(out, index=W.index, columns=W.columns)


def build_supervised(panel, station_split, roles, cfg):
    """Return the supervised table for every eligible station and study day."""
    stations = station_split["station_id"].tolist()
    W = wide_matrix(panel, stations, cfg)
    P = cfg["max_lag_days"]
    R = cfg["rolling_window_days"]
    nbr = neighbor_mean(W, station_split, cfg)

    def stack(frame, name):
        return frame.stack(future_stack=True).rename(name)

    # shift by calendar days: row T of shift(k) holds the value from day T-k
    cols = [stack(W, "y")]
    cols += [stack(W.shift(k), f"lag{k}") for k in range(1, P + 1)]
    origin = W.shift(1)  # day t = T-1
    cols.append(stack(origin.rolling(R, min_periods=R).mean(), f"roll_mean{R}"))
    cols.append(stack(origin.rolling(R, min_periods=R).std(ddof=1), f"roll_sd{R}"))
    cols.append(stack(nbr.shift(1), "nbr_mean"))
    X = pd.concat(cols, axis=1).reset_index()
    X.columns = ["target_date", "station_id"] + X.columns[2:].tolist()

    doy = X["target_date"].dt.dayofyear
    X["doy_sin"] = np.sin(2 * np.pi * doy / DOY_PERIOD)
    X["doy_cos"] = np.cos(2 * np.pi * doy / DOY_PERIOD)
    st = station_split.set_index("station_id")
    X = X.join(st[["latitude", "longitude", "role", "feature_cluster"]], on="station_id")
    for c in sorted(station_split["feature_cluster"].unique())[1:]:
        X[f"cluster_{c}"] = (X["feature_cluster"] == c).astype(float)
    X = X.join(roles, on="target_date")
    X["origin_date"] = X["target_date"] - pd.Timedelta(days=1)
    X = X[X["date_role"] != "burn_in"]
    X["complete"] = X[all_feature_columns(cfg, station_split) + ["y"]].notna().all(axis=1)
    return X.sort_values(["station_id", "target_date"]).reset_index(drop=True)


def cluster_columns(station_split):
    return [f"cluster_{c}" for c in sorted(station_split["feature_cluster"].unique())[1:]]


def feature_columns(fs, p, cfg, station_split):
    """Feature set F1..F4 with lag set 1..p."""
    R = cfg["rolling_window_days"]
    cols = [f"lag{k}" for k in range(1, p + 1)]
    if fs == "F1":
        return cols
    # the rolling mean is an exact average of lags 1..R, so drop it when p >= R
    cols += ([f"roll_mean{R}"] if p < R else []) + [f"roll_sd{R}"]
    if fs == "F2":
        return cols
    cols += ["doy_sin", "doy_cos", "latitude", "longitude"] + cluster_columns(station_split)
    if fs == "F3":
        return cols
    return cols + ["nbr_mean"]


def all_feature_columns(cfg, station_split):
    return feature_columns("F4", 1, cfg, station_split) + [
        f"lag{k}" for k in range(2, cfg["max_lag_days"] + 1)]


def feature_dictionary(cfg, station_split):
    rows = []
    for col in all_feature_columns(cfg, station_split):
        if col.startswith("lag"):
            k = int(col[3:])
            desc = FEATURE_DESCRIPTIONS["lag{k}"].format(k=k, km1=k - 1)
        elif col.startswith("roll_"):
            R = cfg["rolling_window_days"]
            key = col.replace(str(R), "{R}")
            desc = FEATURE_DESCRIPTIONS[key].format(R=R, Rm1=R - 1)
        elif col.startswith("cluster_"):
            desc = FEATURE_DESCRIPTIONS["cluster_{c}"].format(c=col.split("_")[1])
        else:
            desc = FEATURE_DESCRIPTIONS[col]
        sets = [fs for fs in cfg["feature_sets"]
                if any(col in feature_columns(fs, p, cfg, station_split) for p in cfg["lag_sets"])]
        rows.append({"feature": col, "description": desc, "in_feature_sets": ",".join(sets)})
    return pd.DataFrame(rows)
