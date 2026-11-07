"""Step 4 exploration. Uses development stations on non-test days only."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler

from .features import wide_matrix

MAX_ACF_LAG = 14


def pooled_acf_pacf(Z, max_lag=MAX_ACF_LAG):
    """Pooled across stations, using calendar-day shifts within each station.
    PACF(k) = coefficient on lag k in a regression on lags 1..k."""
    frames = {k: Z.shift(k).stack(future_stack=True) for k in range(max_lag + 1)}
    L = pd.concat(frames, axis=1).dropna()
    y = L[0].to_numpy()
    acf, pacf = [], []
    for k in range(1, max_lag + 1):
        acf.append(np.corrcoef(y, L[k])[0, 1])
        A = np.c_[np.ones(len(y)), L[list(range(1, k + 1))].to_numpy()]
        pacf.append(np.linalg.lstsq(A, y, rcond=None)[0][k])
    return pd.DataFrame({"lag": range(1, max_lag + 1), "acf": acf, "pacf": pacf}), len(L)


def per_station_acf1(W):
    return pd.Series({s: W[s].corr(W[s].shift(1)) for s in W.columns})


def run(panel, station_split, roles, profiles, cfg, od, fig_dir):
    dev_st = station_split.loc[station_split["role"] == "development", "station_id"].tolist()
    W = wide_matrix(panel, dev_st, cfg)
    test_days = roles.index[roles["date_role"] == "test"]
    W.loc[W.index.isin(test_days)] = np.nan  # never look at test days

    # anomalies: subtract each station's calendar-month mean (removes the seasonal level)
    Z = W - W.groupby(W.index.month).transform("mean")
    acf_raw, n_raw = pooled_acf_pacf(W - W.mean())
    acf_anom, n_anom = pooled_acf_pacf(Z)
    out = acf_raw.merge(acf_anom, on="lag", suffixes=("_raw", "_anomaly"))
    out.to_csv(od / "acf_pacf_pooled.csv", index=False)
    band = 1.96 / np.sqrt(W.notna().sum().median())

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, kind in zip(axes, ["acf", "pacf"]):
        ax.bar(out["lag"] - 0.2, out[f"{kind}_raw"], 0.4, label="raw (station-demeaned)")
        ax.bar(out["lag"] + 0.2, out[f"{kind}_anomaly"], 0.4, label="anomaly (station-month demeaned)")
        ax.axhspan(-band, band, color="grey", alpha=0.2, label="±1.96/√n per station")
        ax.axvline(cfg["max_lag_days"] + 0.5, ls="--", c="k", lw=0.8)
        ax.set(title=kind.upper() + " of MDA8, pooled over development stations", xlabel="lag (days)")
    axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(fig_dir / "eda_acf_pacf.png", dpi=150); plt.close(fig)

    # distribution + series
    vals = W.stack().to_numpy()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(vals, bins=80)
    ax.axvline(cfg["exceedance_threshold_ppb"], c="r", ls="--", label="70 ppb standard")
    ax.set(xlabel="MDA8 ozone (ppb)", ylabel="station-days", title="Development MDA8 distribution")
    ax.legend(); fig.tight_layout(); fig.savefig(fig_dir / "eda_distribution.png", dpi=150); plt.close(fig)

    rng = np.random.default_rng(cfg["random_seed"])
    show = rng.choice(dev_st, 6, replace=False)
    fig, axes = plt.subplots(6, 1, figsize=(11, 10), sharex=True)
    for ax, s in zip(axes, show):
        ax.plot(W.index, W[s], lw=0.8)
        for blk_start in roles.index[(roles["date_role"] == "test")
                                     & (roles["date_role"].shift() != "test")]:
            ax.axvspan(blk_start, blk_start + pd.Timedelta(days=cfg["test_block_days"]),
                       color="orange", alpha=0.3)
        ax.set_ylabel(s, fontsize=8)
    axes[0].set_title("MDA8 (ppb) at six development stations; test weeks shaded (values hidden)")
    fig.tight_layout(); fig.savefig(fig_dir / "eda_series.png", dpi=150); plt.close(fig)

    # station map + MDS of station profiles, coloured by cluster
    ss = station_split
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for c in sorted(ss["feature_cluster"].unique()):
        for role, mk in [("development", "o"), ("held_out", "x")]:
            m = (ss["feature_cluster"] == c) & (ss["role"] == role)
            axes[0].scatter(ss.loc[m, "longitude"], ss.loc[m, "latitude"], marker=mk, s=18,
                            color=f"C{c}", label=f"cluster {c}, {role}")
    axes[0].set(xlabel="longitude", ylabel="latitude", title="Stations by cluster (x = held out)")
    axes[0].legend(fontsize=7)
    prof = profiles.loc[ss["station_id"]]
    emb = MDS(2, random_state=cfg["random_seed"], normalized_stress="auto").fit_transform(
        StandardScaler().fit_transform(prof))
    axes[1].scatter(emb[:, 0], emb[:, 1], c=[f"C{c}" for c in ss["feature_cluster"]], s=14)
    axes[1].set(title="MDS of standardized station profiles", xlabel="MDS 1", ylabel="MDS 2")
    fig.tight_layout(); fig.savefig(fig_dir / "eda_station_clusters.png", dpi=150); plt.close(fig)

    acf1 = per_station_acf1(W)
    summary = {
        "n_development_stations": len(dev_st),
        "n_station_days": int(W.notna().sum().sum()),
        "mda8_ppb_mean": float(np.nanmean(vals)), "mda8_ppb_sd": float(np.nanstd(vals)),
        "share_days_over_70ppb": float(np.mean(vals > cfg["exceedance_threshold_ppb"])),
        "per_station_acf1_median": float(acf1.median()),
        "pooled_rows_raw": n_raw, "pooled_rows_anomaly": n_anom,
        "per_station_band": float(band),
        "acf_pacf": out.round(3).to_dict(orient="records"),
        "cluster_profile_means": (profiles.loc[ss["station_id"]]
                                  .groupby(ss["feature_cluster"].to_numpy()).mean().round(2)
                                  .to_dict(orient="index")),
    }
    return summary
