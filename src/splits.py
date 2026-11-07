"""Date split (seasonal test blocks + embargo), station eligibility, clustering,
station holdout, and the cross-validation fold layout.

Everything here is decided once and saved; later stages only read the saved files.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from .common import dates

EARTH_RADIUS_KM = 6371.0


# ---------------------------------------------------------------- date split
def draw_test_blocks(cfg):
    """One contiguous test block per season, drawn with the fixed seed.

    A block (plus its embargo, except for the last season) must fit inside its
    season window, so blocks and embargoes never overlap the next season's window.
    Depends only on the config, not on the data.
    """
    rng = np.random.default_rng(cfg["random_seed"])
    L, E = cfg["test_block_days"], cfg["embargo_days"]
    seasons = list(cfg["season_windows"])
    blocks = {}
    for i, season in enumerate(seasons):
        a, b = (pd.Timestamp(x) for x in cfg["season_windows"][season])
        span = L + (E if i < len(seasons) - 1 else 0)
        n_starts = (b - a).days + 1 - span + 1
        if n_starts < 1:
            raise ValueError(f"season window {season} too short for block+embargo")
        start = a + pd.Timedelta(days=int(rng.integers(0, n_starts)))
        blocks[season] = pd.date_range(start, periods=L, freq="D")
    return blocks


def date_roles(cfg, blocks=None):
    """Label every study day as burn_in / test / embargo / development."""
    blocks = blocks or draw_test_blocks(cfg)
    days = dates(cfg["study_start"], cfg["study_end"])
    role = pd.Series("development", index=days, name="date_role")
    season = pd.Series(pd.NA, index=days, name="test_season", dtype="object")
    role.iloc[: cfg["max_lag_days"]] = "burn_in"
    for s, blk in blocks.items():
        after = pd.date_range(blk[-1] + pd.Timedelta(days=1), periods=cfg["embargo_days"])
        role.loc[role.index.isin(after)] = "embargo"
    for s, blk in blocks.items():
        role.loc[blk] = "test"
        season.loc[blk] = s
    return pd.concat([role, season], axis=1).rename_axis("date")


# ------------------------------------------------------- stations & clusters
def eligible_stations(panel, blocks, cfg):
    rule = cfg["eligibility"]
    p = panel[panel["date"].between(cfg["study_start"], cfg["study_end"])]
    total = p.groupby("station_id").size()
    ok = total[total >= rule["min_total_days"]].index
    for blk in blocks.values():
        n = p[p["date"].isin(blk)].groupby("station_id").size()
        ok = ok.intersection(n[n >= rule["min_days_per_test_block"]].index)
    return sorted(ok)


def station_profiles(panel, stations, test_days, cfg):
    """Per-station summary used for clustering, from non-test days only."""
    p = panel[panel["station_id"].isin(stations) & ~panel["date"].isin(test_days)
              & panel["date"].between(cfg["study_start"], cfg["study_end"])].copy()
    season_of = {}
    for s, (a, b) in cfg["season_windows"].items():
        for d in dates(a, b):
            season_of[d] = s
    # profile seasons cover the whole study period (burn-in days count as winter)
    p["season"] = p["date"].map(season_of).fillna("winter")
    prof = p.pivot_table(index="station_id", columns="season", values="mda8_ppb", aggfunc="mean")
    prof = prof[list(cfg["season_windows"])].add_prefix("mean_")
    prof["sd"] = p.groupby("station_id")["mda8_ppb"].std()
    # lag-1 autocorrelation using true calendar neighbours (not row neighbours)
    prev = p[["station_id", "date", "mda8_ppb"]].copy()
    prev["date"] += pd.Timedelta(days=1)
    pairs = p.merge(prev, on=["station_id", "date"], suffixes=("", "_prev"))
    prof["acf1"] = pairs.groupby("station_id").apply(
        lambda g: g["mda8_ppb"].corr(g["mda8_ppb_prev"]), include_groups=False)
    loc = panel.drop_duplicates("station_id").set_index("station_id")[["latitude", "longitude"]]
    return prof.join(loc).loc[stations]


def choose_k(X, cfg):
    lo, hi = cfg["cluster_k_range"]
    scores = {}
    for k in range(lo, hi + 1):
        lab = KMeans(k, n_init=20, random_state=cfg["random_seed"]).fit_predict(X)
        scores[k] = calinski_harabasz_score(X, lab)
    return max(scores, key=scores.get), scores


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def distance_matrix_km(lat_a, lon_a, lat_b, lon_b):
    return haversine_km(np.asarray(lat_a)[:, None], np.asarray(lon_a)[:, None],
                        np.asarray(lat_b)[None, :], np.asarray(lon_b)[None, :])


def split_stations(panel, blocks, cfg):
    """Eligibility -> stratification clusters -> stratified holdout ->
    feature clusters fitted on development stations only."""
    seed = cfg["random_seed"]
    test_days = pd.DatetimeIndex(np.concatenate([b.values for b in blocks.values()]))
    stations = eligible_stations(panel, blocks, cfg)
    prof = station_profiles(panel, stations, test_days, cfg)

    # 1) clusters over all eligible stations, used only to stratify the holdout
    X_all = StandardScaler().fit_transform(prof)
    k, ch_scores = choose_k(X_all, cfg)
    strat = KMeans(k, n_init=20, random_state=seed).fit_predict(X_all)
    counts = np.bincount(strat)
    strat_label = np.where(counts[strat] >= 2, strat, -1)  # singleton clusters pooled

    sss = StratifiedShuffleSplit(1, test_size=cfg["holdout_station_fraction"], random_state=seed)
    dev_idx, held_idx = next(sss.split(prof, strat_label))
    role = np.array(["development"] * len(prof), dtype=object)
    role[held_idx] = "held_out"

    # 2) feature clusters: scaler + centroids fitted on development stations only,
    #    held-out stations assigned to the nearest centroid
    dev = prof.iloc[dev_idx]
    scaler = StandardScaler().fit(dev)
    km = KMeans(k, n_init=20, random_state=seed).fit(scaler.transform(dev))
    feat_cluster = km.predict(scaler.transform(prof))

    out = prof.reset_index()[["station_id", "latitude", "longitude"]].copy()
    out["role"] = role
    out["strat_cluster"] = strat
    out["feature_cluster"] = feat_cluster
    devmask = out["role"] == "development"
    D = distance_matrix_km(out["latitude"], out["longitude"],
                           out.loc[devmask, "latitude"], out.loc[devmask, "longitude"])
    D[np.where(devmask)[0], np.arange(devmask.sum())] = np.inf  # ignore self-distance
    out["nearest_dev_km"] = D.min(axis=1)
    info = {"n_eligible": len(stations), "k": int(k),
            "calinski_harabasz": {int(a): float(b) for a, b in ch_scores.items()},
            "n_development": int(devmask.sum()), "n_held_out": int((~devmask).sum()),
            "cluster_sizes_all": {int(c): int(n) for c, n in enumerate(counts)}}
    return out, prof, info


# -------------------------------------------------------------- CV layout
def cv_layout(roles, station_split, cfg):
    """Station groups x day folds. Day folds are contiguous calendar blocks
    (cfg day_block_days) assigned round-robin so every fold spans the year."""
    c = cfg["cv"]
    rng = np.random.default_rng(cfg["random_seed"])
    dev_st = station_split.loc[station_split["role"] == "development", "station_id"].to_numpy()
    groups = pd.Series(np.arange(len(dev_st)) % c["n_station_groups"], index=rng.permutation(dev_st),
                       name="station_group").rename_axis("station_id").sort_index()
    dev_days = roles.index[roles["date_role"] == "development"]
    block_id = (dev_days - dev_days[0]).days // c["day_block_days"]
    day_fold = pd.Series(np.asarray(block_id) % c["n_day_folds"], index=dev_days,
                         name="day_fold").rename_axis("date")
    day_block = pd.Series(np.asarray(block_id), index=dev_days, name="day_block").rename_axis("date")
    return groups, pd.concat([day_fold, day_block], axis=1)


def cv_cells(groups, days, cfg):
    """Yield (station_group, day_fold, val_stations, val_days, train_excluded_days).

    Training for a cell drops the validation stations, the validation days, and
    the embargo_days after each validation block (whose lag windows would contain
    validation-day values)."""
    E = cfg["embargo_days"]
    for f in range(cfg["cv"]["n_day_folds"]):
        val_days = days.index[days["day_fold"] == f]
        excluded = set(val_days)
        for b in days.loc[val_days, "day_block"].unique():
            end = days.index[days["day_block"] == b].max()
            excluded |= set(pd.date_range(end + pd.Timedelta(days=1), periods=E))
        for g in range(cfg["cv"]["n_station_groups"]):
            yield g, f, groups.index[groups == g], val_days, pd.DatetimeIndex(sorted(excluded))
