"""Leakage and correctness checks (requirements §17). Synthetic data only,
except where noted, so these run without the raw CSV."""
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import SplineTransformer

from src import evaluation, features, models, splits, validation

CFG = {
    "random_seed": 42, "study_start": "2024-01-01", "study_end": "2024-03-31",
    "max_lag_days": 7, "rolling_window_days": 7, "test_block_days": 7, "embargo_days": 7,
    "season_windows": {"a": ["2024-01-08", "2024-02-10"], "b": ["2024-02-11", "2024-03-31"]},
    "neighbor_k": 2, "feature_sets": ["F1", "F2", "F3", "F4"], "lag_sets": [1, 3, 7],
    "cv": {"n_station_groups": 2, "n_day_folds": 2, "day_block_days": 7},
}
STATIONS = ["S1", "S2", "S3", "S4"]


def make_panel(drop=()):
    days = pd.date_range(CFG["study_start"], CFG["study_end"])
    rows = [{"station_id": s, "date": d, "mda8_ppb": 100 * i + k}
            for i, s in enumerate(STATIONS) for k, d in enumerate(days)]
    p = pd.DataFrame(rows)
    return p[~p.set_index(["station_id", "date"]).index.isin(list(drop))].reset_index(drop=True)


def make_station_split():
    return pd.DataFrame({"station_id": STATIONS, "latitude": [40, 40.1, 40.2, 45],
                         "longitude": [-90, -90.1, -90.2, -100],
                         "role": ["development", "development", "development", "held_out"],
                         "feature_cluster": [0, 0, 1, 1]})


def supervised(panel=None):
    roles = splits.date_roles(CFG)
    return features.build_supervised(panel if panel is not None else make_panel(),
                                     make_station_split(), roles, CFG)


def row(sup, sid, date):
    return sup[(sup["station_id"] == sid) & (sup["target_date"] == pd.Timestamp(date))].iloc[0]


# 1. lags line up with calendar days and stay within a station
def test_lags_are_calendar_days_within_station():
    sup = supervised()
    r = row(sup, "S2", "2024-01-20")  # S2 value on day k (0-based) is 100 + k
    k = (pd.Timestamp("2024-01-20") - pd.Timestamp("2024-01-01")).days
    assert r["y"] == 100 + k
    for lag in range(1, 8):
        assert r[f"lag{lag}"] == 100 + k - lag
    assert r["roll_mean7"] == np.mean([100 + k - j for j in range(1, 8)])
    first = sup[sup["station_id"] == "S2"].iloc[0]
    assert first["target_date"] == pd.Timestamp("2024-01-08") and first["lag7"] == 100


# 2. a missing day is not treated as "yesterday"
def test_missing_day_breaks_lags():
    gap = pd.Timestamp("2024-01-20")
    sup = supervised(make_panel(drop=[("S1", gap)]))
    nxt = row(sup, "S1", "2024-01-21")
    assert np.isnan(nxt["lag1"]) and np.isnan(nxt["roll_mean7"]) and not nxt["complete"]
    assert np.isnan(row(sup, "S1", "2024-01-22")["lag2"])


# 5. changing a future value never changes earlier rows' inputs
def test_future_perturbation_invariance():
    base = supervised()
    p = make_panel()
    d = pd.Timestamp("2024-02-15")
    p.loc[(p["station_id"] == "S1") & (p["date"] == d), "mda8_ppb"] += 999
    pert = supervised(p)
    cols = features.all_feature_columns(CFG, make_station_split())
    early = base["target_date"] <= d
    pd.testing.assert_frame_equal(base.loc[early, cols], pert.loc[early, cols])


# neighbour feature: k nearest *development* stations, never self, never held-out
def test_neighbor_mean_uses_development_stations_only():
    sup = supervised()
    r = row(sup, "S1", "2024-01-20")  # nearest dev stations to S1: S2, S3
    k = (pd.Timestamp("2024-01-19") - pd.Timestamp("2024-01-01")).days  # origin day t
    assert r["nbr_mean"] == np.mean([100 + k, 200 + k])
    r4 = row(sup, "S4", "2024-01-20")  # held-out S4 must not appear in anyone's pool
    assert r4["nbr_mean"] == np.mean([200 + k, 100 + k])  # S3 and S2 are nearest to S4


# 3. embargo: no development row's input window touches a test day
def test_development_inputs_never_touch_test_days():
    roles = splits.date_roles(CFG)
    test_days = set(roles.index[roles["date_role"] == "test"])
    for T in roles.index[roles["date_role"] == "development"]:
        window = {T - pd.Timedelta(days=j) for j in range(1, CFG["max_lag_days"] + 1)}
        assert not window & test_days, T


def test_cv_training_inputs_never_touch_validation_days():
    roles = splits.date_roles(CFG)
    groups, days = splits.cv_layout(roles, make_station_split(), CFG)
    for g, f, val_st, val_days, excluded in splits.cv_cells(groups, days, CFG):
        train_days = set(days.index) - set(excluded)
        for T in train_days:
            window = {T - pd.Timedelta(days=j) for j in range(1, CFG["max_lag_days"] + 1)}
            assert not window & set(val_days)
        assert not set(val_st) & set(groups.index[groups != g])


# 4. held-out stations never enter development rows or CV
def test_held_out_stations_excluded_from_development():
    dev = validation.development_rows(supervised())
    assert "S4" not in set(dev["station_id"])
    groups, _ = splits.cv_layout(splits.date_roles(CFG), make_station_split(), CFG)
    assert "S4" not in set(groups.index)


def test_test_blocks_fixed_and_separated():
    a, b = splits.draw_test_blocks(CFG), splits.draw_test_blocks(CFG)
    assert all((a[s] == b[s]).all() for s in a)
    end_a = a["a"][-1] + pd.Timedelta(days=CFG["embargo_days"])
    assert end_a < b["b"][0]


# 6. frozen models do not change during prediction
def test_frozen_parameters_unchanged(tmp_path):
    sup = supervised()
    sup.loc[sup["date_role"] == "embargo", "date_role"] = "development"
    ss = make_station_split()
    cands = {c["id"]: c for c in models.candidates(
        {**CFG, "grids": {"ridge_alpha": [1.0], "lasso_alpha": [0.1], "spline_n_knots": [4],
                          "spline_alpha": [1.0], "knn_k": [2], "pcr_feature_set": "F2"}}, ss)}
    sel = {"ridge": "ridge|F2|p1|alpha=1.0", "persistence": "persistence"}
    evaluation.fit_frozen(sup, sel, cands, tmp_path)
    import joblib
    frozen = {f: joblib.load(tmp_path / f"{f}.joblib") for f in sel}
    before = {f: evaluation.param_hash(o["model"]) for f, o in frozen.items()}
    evaluation.predict_frozen(sup, frozen)
    assert before == {f: evaluation.param_hash(o["model"]) for f, o in frozen.items()}


# 7. metric arithmetic and sign convention (error = prediction - truth)
def test_metrics_small_example():
    pred = pd.DataFrame({"station_id": ["a", "a", "b"], "y_true": [10.0, 20.0, 30.0],
                         "y_pred": [12.0, 18.0, 33.0], "baseline_pred": [10.0, 10.0, 30.0],
                         "model_id": "m"})
    pred["error"] = pred["y_pred"] - pred["y_true"]
    pred["absolute_error"] = pred["error"].abs()
    pred["squared_error"] = pred["error"] ** 2
    m = evaluation.metrics(pred, ["model_id"]).iloc[0]
    assert m["station_macro_mae"] == pytest.approx((2 + 3) / 2)
    assert m["pooled_mae"] == pytest.approx(7 / 3)
    assert m["rmse"] == pytest.approx(np.sqrt((4 + 4 + 9) / 3))
    assert m["mean_error"] == pytest.approx((2 - 2 + 3) / 3)
    assert m["baseline_station_macro_mae"] == pytest.approx((5 + 0) / 2)
    assert m["improvement_vs_persistence"] == pytest.approx(1 - 2.5 / 2.5)


def test_one_se_rule_picks_simplest_within_one_se():
    s = pd.DataFrame({"candidate_id": ["a", "b", "c"], "family": "ridge",
                      "cv_mae": [5.0, 5.04, 5.2], "cv_se": [0.05, 0.05, 0.05],
                      "complexity": [(3, -1.0), (2, -1.0), (1, -1.0)]})
    sel = validation.one_se_select(s).iloc[0]
    assert sel["best_id"] == "a" and sel["selected_id"] == "b"


# 8. spline extrapolates linearly with knots fixed from training data
def test_spline_linear_extrapolation():
    x = np.linspace(10, 60, 200)[:, None]
    st = SplineTransformer(n_knots=4, degree=3, knots="quantile", extrapolation="linear").fit(x)
    w = np.random.default_rng(0).normal(size=st.n_features_out_)
    far = np.array([[80.0], [90.0], [100.0], [110.0]])
    f = st.transform(far) @ w
    assert np.allclose(np.diff(f, 2), 0, atol=1e-8)
    knots_before = st.bsplines_[0].t.copy()
    st.transform(np.array([[500.0]]))
    assert np.array_equal(knots_before, st.bsplines_[0].t)


def test_locked_test_blocks_are_used_and_checked():
    locked = {**CFG, "locked_test_blocks": {"a": "2024-01-20", "b": "2024-03-01"}}
    blocks = splits.draw_test_blocks(locked)
    assert blocks["a"][0] == pd.Timestamp("2024-01-20") and len(blocks["b"]) == CFG["test_block_days"]
    with pytest.raises(ValueError):  # embargo after block a would run into block b
        splits.draw_test_blocks({**CFG, "locked_test_blocks": {"a": "2024-02-20", "b": "2024-03-01"}})
    with pytest.raises(ValueError):  # block before the burn-in ends
        splits.draw_test_blocks({**CFG, "locked_test_blocks": {"a": "2024-01-03", "b": "2024-03-01"}})


def test_naive_blend_is_average_of_today_and_rolling_mean():
    X = pd.DataFrame({"lag1": [40.0, 10.0], "roll_mean7": [30.0, 20.0]})
    c = next(c for c in models.candidates({**CFG, "grids": {"ridge_alpha": [], "lasso_alpha": [],
                                                            "spline_n_knots": [], "spline_alpha": [],
                                                            "knn_k": [], "pcr_feature_set": "F2"}},
                                          make_station_split()) if c["family"] == "naive_blend")
    assert np.allclose(models.build(c).fit(X, None).predict(X), [35.0, 15.0])
