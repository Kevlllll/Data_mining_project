"""Read the raw EPA AQS daily file, audit it, and build the cleaned daily panel
(one row per station x date, primary POC only)."""
import numpy as np
import pandas as pd

from .common import ROOT, out_dir, write_json
from .splits import draw_test_blocks

CODE_COLS = ["State Code", "County Code", "Site Num", "Parameter Code", "Method Code"]


def read_raw(cfg):
    df = pd.read_csv(ROOT / cfg["raw_data_path"], dtype={c: str for c in CODE_COLS},
                     low_memory=False)
    # pandas reads the literal Event Type "None" as missing; restore it
    df["Event Type"] = df["Event Type"].fillna("None")
    df["station_id"] = df["State Code"] + "-" + df["County Code"] + "-" + df["Site Num"]
    df["date"] = pd.to_datetime(df["Date Local"])
    df["row_id"] = np.arange(len(df))
    return df


def audit_and_clean(cfg):
    """Return (panel, stations) and write the audit outputs."""
    od = out_dir(cfg, "audit")
    raw = read_raw(cfg)
    excluded = []

    def exclude(mask, reason):
        excluded.append(raw.loc[mask, ["row_id", "station_id", "POC", "date"]].assign(reason=reason))

    # 1) measurement definition (expect a single definition in this file)
    defn_ok = ((raw["Sample Duration"] == cfg["sample_duration"])
               & (raw["Pollutant Standard"] == cfg["pollutant_standard"])
               & (raw["Units of Measure"] == cfg["units"]))
    exclude(~defn_ok, "measurement_definition")
    # 2) study window
    in_window = raw["date"].between(cfg["study_start"], cfg["study_end"])
    exclude(defn_ok & ~in_window, "outside_study_window")
    df = raw[defn_ok & in_window].copy()

    # 3) exceptional-event duplicates: same station/POC/day reported twice,
    #    once with events Included and once Excluded -> keep the as-measured row
    key = ["station_id", "POC", "date"]
    dup = df.duplicated(key, keep=False)
    drop_event = dup & (df["Event Type"] == "Excluded")
    exclude(raw["row_id"].isin(df.loc[drop_event, "row_id"]), "event_type_excluded_duplicate")
    df = df[~drop_event]
    assert not df.duplicated(key).any(), "unexplained station/POC/day duplicates"

    # 4) one POC per station: most days outside the test blocks (ties -> lowest POC)
    test_days = np.concatenate([b.values for b in draw_test_blocks(cfg).values()])
    n_days = (df[~df["date"].isin(test_days)].groupby(["station_id", "POC"]).size()
              .rename("n").reset_index().sort_values(["station_id", "n", "POC"],
                                                     ascending=[True, False, True]))
    primary = n_days.drop_duplicates("station_id").set_index("station_id")["POC"]
    df["is_primary"] = df["POC"] == df["station_id"].map(primary)

    multi = df[df.groupby(["station_id", "date"])["POC"].transform("nunique") > 1]
    conflicts = (multi.pivot_table(index=["station_id", "date"], columns="POC",
                                   values="1st Max Value", aggfunc="first")
                 .add_prefix("mda8_poc").reset_index())
    conflicts["primary_poc"] = conflicts["station_id"].map(primary)
    vals = conflicts.filter(like="mda8_poc")
    conflicts["max_abs_diff_ppm"] = (vals.max(axis=1) - vals.min(axis=1)).round(6)
    conflicts.to_csv(od / "duplicate_conflicts.csv", index=False)
    exclude(raw["row_id"].isin(df.loc[~df["is_primary"], "row_id"]), "non_primary_poc")
    df = df[df["is_primary"]]

    # 5) build the panel; flags only, no rows removed for quality
    scale = cfg["target_scale_to_ppb"]
    panel = pd.DataFrame({
        "station_id": df["station_id"], "date": df["date"],
        "mda8_ppb": df[cfg["target_column"]] * scale,
        "mean_ppb": df["Arithmetic Mean"] * scale,
        "poc": df["POC"], "method_code": df["Method Code"], "method_name": df["Method Name"],
        "observation_count": df["Observation Count"],
        "observation_percent": df["Observation Percent"],
        "event_type": df["Event Type"],
        "latitude": df["Latitude"], "longitude": df["Longitude"],
    }).sort_values(["station_id", "date"]).reset_index(drop=True)
    panel["flag_low_completeness"] = panel["observation_count"] < cfg["low_completeness_obs_count"]
    panel["flag_nonpositive"] = panel["mda8_ppb"] <= 0
    panel["flag_event"] = panel["event_type"] != "None"
    # one location per station (first record of the primary POC)
    loc = panel.groupby("station_id")[["latitude", "longitude"]].transform("first")
    panel[["latitude", "longitude"]] = loc

    stations = (df.groupby("station_id")
                .agg(state=("State Name", "first"), county=("County Name", "first"),
                     site_name=("Local Site Name", "first"), cbsa=("CBSA Name", "first"))
                .join(panel.groupby("station_id")[["latitude", "longitude"]].first())
                .join(primary.rename("primary_poc")).reset_index())

    # ---- audit outputs
    pd.concat(excluded).to_csv(od / "excluded_rows.csv", index=False)
    cov = panel.assign(month=panel["date"].dt.month).pivot_table(
        index="station_id", columns="month", values="mda8_ppb", aggfunc="size", fill_value=0)
    cov.columns = [f"days_m{m:02d}" for m in cov.columns]
    cov = stations.set_index("station_id").join(cov)
    cov["first_date"] = panel.groupby("station_id")["date"].min()
    cov["last_date"] = panel.groupby("station_id")["date"].max()
    cov["n_days"] = panel.groupby("station_id").size()
    cov["n_pocs_raw"] = raw.groupby("station_id")["POC"].nunique()
    cov.reset_index().to_csv(od / "station_coverage.csv", index=False)

    method_changes = (raw.sort_values("date").groupby(["station_id", "POC"])["Method Code"]
                      .apply(lambda s: int((s != s.shift()).sum() - 1)))
    raw_month = raw.groupby(raw["date"].dt.month)["station_id"].nunique()
    audit = {
        "raw_rows": len(raw), "raw_stations": raw["station_id"].nunique(),
        "raw_states": raw["State Code"].nunique(),
        "raw_date_range": [raw["date"].min().date(), raw["date"].max().date()],
        "raw_distinct_dates": raw["date"].nunique(),
        "sample_durations": raw["Sample Duration"].value_counts().to_dict(),
        "pollutant_standards": raw["Pollutant Standard"].value_counts().to_dict(),
        "units": raw["Units of Measure"].value_counts().to_dict(),
        "event_types": raw["Event Type"].value_counts().to_dict(),
        "observation_count_multiple_of_17_share": float((raw["Observation Count"] % 17 == 0).mean()),
        "observation_percent_range": [float(raw["Observation Percent"].min()),
                                      float(raw["Observation Percent"].max())],
        "raw_nonpositive_values": int((raw[cfg["target_column"]] <= 0).sum()),
        "raw_active_stations_by_month": {int(k): int(v) for k, v in raw_month.items()},
        "multi_poc_station_days": int(len(conflicts)),
        "multi_poc_stations": int(conflicts["station_id"].nunique()),
        "multi_poc_median_abs_diff_ppb": float(conflicts["max_abs_diff_ppm"].median() * scale),
        "within_station_poc_method_changes": int(method_changes.sum()),
        "excluded_rows_by_reason": pd.concat(excluded)["reason"].value_counts().to_dict(),
        "panel_rows": len(panel), "panel_stations": panel["station_id"].nunique(),
        "panel_flags": {c: int(panel[c].sum()) for c in panel.columns if c.startswith("flag_")},
        "target": f"{cfg['target_column']} = daily max 8-hour running average (MDA8), ppb",
    }
    write_json(audit, od / "data_audit.json")
    panel.to_parquet(od / "daily_panel.parquet", index=False)
    stations.to_csv(od / "stations.csv", index=False)
    return panel, stations, audit
