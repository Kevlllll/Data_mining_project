"""Candidate models (Step 6). Each candidate is a dict with an id, a family,
its feature columns, its hyperparameters, and a complexity key for the
one-standard-error rule (smaller key = simpler model)."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from .features import feature_columns

SPLINE_COLUMN = "lag1"  # the nonlinear term: tomorrow vs today's level


class Persistence:
    """prediction = today's MDA8 (lag1). Nothing to fit."""

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.asarray(X["lag1"], dtype=float)


class NaiveBlend:
    """prediction = (today + rolling mean) / 2. A no-fit baseline that captures
    pull-toward-the-weekly-level without estimating anything."""

    def __init__(self, mean_col):
        self.mean_col = mean_col

    def fit(self, X, y):
        return self

    def predict(self, X):
        return 0.5 * np.asarray(X["lag1"], dtype=float) + 0.5 * np.asarray(X[self.mean_col], dtype=float)


BASELINES = ("persistence", "naive_blend")


def build(c):
    fam, prm = c["family"], c["params"]
    if fam == "persistence":
        return Persistence()
    if fam == "naive_blend":
        return NaiveBlend(c["columns"][1])
    if fam == "ols":
        return LinearRegression()
    if fam == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=prm["alpha"]))
    if fam == "lasso":
        return make_pipeline(StandardScaler(), Lasso(alpha=prm["alpha"], max_iter=50_000, tol=1e-5))
    if fam == "spline":
        others = [col for col in c["columns"] if col != SPLINE_COLUMN]
        spline = SplineTransformer(n_knots=prm["n_knots"], degree=3, knots="quantile",
                                   extrapolation="linear")
        parts = [("spline", spline, [SPLINE_COLUMN])]
        if others:
            parts.append(("linear", StandardScaler(), others))
        return make_pipeline(ColumnTransformer(parts), Ridge(alpha=prm["alpha"]))
    if fam == "knn":
        return make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=prm["k"]))
    if fam == "pcr":
        return make_pipeline(StandardScaler(), PCA(prm["n_components"]), LinearRegression())
    raise ValueError(fam)


def candidates(cfg, station_split):
    g = cfg["grids"]
    out = [dict(id="persistence", family="persistence", fs="-", p=1, columns=["lag1"],
                params={}, complexity=(0,))]
    R = cfg["rolling_window_days"]
    out.append(dict(id="naive_blend", family="naive_blend", fs="-", p=1,
                    columns=["lag1", f"roll_mean{R}"], params={}, complexity=(0,)))

    def add(family, fs, p, params, complexity):
        cols = feature_columns(fs, p, cfg, station_split)
        pid = "_".join(f"{k}={v}" for k, v in params.items())
        out.append(dict(id=f"{family}|{fs}|p{p}" + (f"|{pid}" if pid else ""), family=family,
                        fs=fs, p=p, columns=cols, params=params,
                        complexity=(len(cols),) + complexity))

    for fs in cfg["feature_sets"]:
        for p in cfg["lag_sets"]:
            add("ols", fs, p, {}, ())
            for a in g["ridge_alpha"]:
                add("ridge", fs, p, {"alpha": a}, (-a,))
            for a in g["lasso_alpha"]:
                add("lasso", fs, p, {"alpha": a}, (-a,))
            for nk in g["spline_n_knots"]:
                for a in g["spline_alpha"]:
                    add("spline", fs, p, {"n_knots": nk, "alpha": a}, (nk, -a))
            for k in g["knn_k"]:
                add("knn", fs, p, {"k": k}, (-k,))
    pcr_cols = feature_columns(g["pcr_feature_set"], max(cfg["lag_sets"]), cfg, station_split)
    for n in range(1, len(pcr_cols) + 1):
        out.append(dict(id=f"pcr|{g['pcr_feature_set']}|p{max(cfg['lag_sets'])}|n_components={n}",
                        family="pcr", fs=g["pcr_feature_set"], p=max(cfg["lag_sets"]),
                        columns=pcr_cols, params={"n_components": n}, complexity=(n,)))
    return out
