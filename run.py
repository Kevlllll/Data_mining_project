"""Pipeline entry point.

    python run.py audit      # Step 2: audit + clean -> outputs/audit/
    python run.py split      # Step 3: test blocks, stations, CV layout -> outputs/splits/
    python run.py explore    # Step 4: EDA on development data -> outputs/explore/, outputs/figures/
    python run.py develop    # Steps 5-7: features, cross-validation, one-SE selection
    python run.py finalize   # Step 8: refit selected models on all development rows, freeze
    python run.py sensitivity  # CV with 3/5/7/10-day history windows (development data only)
    python run.py all        # audit -> finalize (never runs evaluate)

    python run.py evaluate   # Step 9: ONE-TIME final test on frozen models
"""
import json
import sys

import pandas as pd

import joblib

from src.common import load_config, out_dir, read_json, write_json
from src import (data, evaluation, explore, features, figures, models, sensitivity, splits,
                 validation)


def stage_audit(cfg):
    _, _, audit = data.audit_and_clean(cfg)
    print(f"audit: {audit['panel_rows']} panel rows, {audit['panel_stations']} stations, "
          f"excluded {audit['excluded_rows_by_reason']}")


def stage_split(cfg):
    od = out_dir(cfg, "splits")
    panel = pd.read_parquet(out_dir(cfg, "audit") / "daily_panel.parquet")
    blocks = splits.draw_test_blocks(cfg)
    roles = splits.date_roles(cfg, blocks)
    station_split, profiles, info = splits.split_stations(panel, blocks, cfg)
    groups, days = splits.cv_layout(roles, station_split, cfg)

    roles.reset_index().to_csv(od / "date_roles.csv", index=False)
    station_split.merge(groups.reset_index(), on="station_id", how="left").to_csv(
        od / "station_split.csv", index=False)
    profiles.to_csv(od / "station_profiles.csv")
    days.reset_index().to_csv(od / "validation_day_folds.csv", index=False)
    counts = roles["date_role"].value_counts().to_dict()
    write_json({
        "test_blocks": {s: [str(b[0].date()), str(b[-1].date())] for s, b in blocks.items()},
        "day_counts": counts,
        "embargo_days": cfg["embargo_days"], "max_lag_days": cfg["max_lag_days"],
        "stations": info, "seed": cfg["random_seed"], "config_sha256": cfg["_config_sha256"],
    }, od / "date_split.json")
    print("split:", {s: f"{b[0].date()}..{b[-1].date()}" for s, b in blocks.items()})
    print("days:", counts, "| stations:", {k: info[k] for k in
                                            ("n_eligible", "n_development", "n_held_out", "k")})


def load_splits(cfg):
    sd = out_dir(cfg, "splits")
    station_split = pd.read_csv(sd / "station_split.csv")
    roles = pd.read_csv(sd / "date_roles.csv", parse_dates=["date"]).set_index("date")
    profiles = pd.read_csv(sd / "station_profiles.csv", index_col=0)
    return station_split, roles, profiles


def stage_explore(cfg):
    panel = pd.read_parquet(out_dir(cfg, "audit") / "daily_panel.parquet")
    station_split, roles, profiles = load_splits(cfg)
    summary = explore.run(panel, station_split, roles, profiles, cfg,
                          out_dir(cfg, "explore"), out_dir(cfg, "figures"))
    write_json(summary, out_dir(cfg, "explore") / "eda_summary.json")
    print("explore: ACF/PACF (anomaly):",
          [(r["lag"], r["acf_anomaly"], r["pacf_anomaly"]) for r in summary["acf_pacf"][:8]])


def build_features(cfg, station_split, roles):
    panel = pd.read_parquet(out_dir(cfg, "audit") / "daily_panel.parquet")
    return features.build_supervised(panel, station_split, roles, cfg)


def stage_develop(cfg):
    od, fig = out_dir(cfg, "development"), out_dir(cfg, "figures")
    station_split, roles, _ = load_splits(cfg)
    sup = build_features(cfg, station_split, roles)
    sup.to_parquet(out_dir(cfg, "features") / "supervised.parquet", index=False)
    features.feature_dictionary(cfg, station_split).to_csv(
        out_dir(cfg, "features") / "feature_dictionary.csv", index=False)

    dev = validation.development_rows(sup)
    groups, days = splits.cv_layout(roles, station_split, cfg)
    cands = models.candidates(cfg, station_split)
    cand_by_id = {c["id"]: c for c in cands}
    scores, _ = validation.run_cv(dev, cands, groups, days, cfg)
    summary = validation.summarize(scores, cands)
    selection = validation.one_se_select(summary)

    scores.to_csv(od / "cv_cell_scores.csv", index=False)
    summary.assign(params=summary["params"].map(json.dumps)).sort_values("cv_mae").to_csv(
        od / "candidate_scores.csv", index=False)
    selection.to_csv(od / "selection.csv", index=False)
    primary = validation.designate_primary(selection, summary)
    paired = validation.paired_differences(
        scores, selection.set_index("family").loc[primary, "selected_id"],
        list(selection["best_id"]) + list(selection["selected_id"]))
    paired.to_csv(od / "paired_vs_primary.csv", index=False)
    selected = dict(zip(selection["family"], selection["selected_id"]))
    write_json({"selected": selected, "primary_model": primary,
                "rule": ("one-SE within each family on CV station-macro MAE; primary = one-SE "
                         "across the family selections, fewest inputs, then fewest tuning parameters"),
                "config_sha256": cfg["_config_sha256"]}, od / "selected_specs.json")

    # validation predictions for the selected specs (for residual plots in the notebook)
    _, preds = validation.run_cv(dev, [cand_by_id[c] for c in selected.values()], groups, days,
                                 cfg, keep_preds=True)
    preds.to_parquet(od / "validation_predictions.parquet", index=False)

    figures.cv_curves(summary, selection, fig / "cv_curves.png")
    extra = figures.shrinkage_paths(dev, cand_by_id, selection, fig / "shrinkage_paths.png")
    extra.update(figures.ridge_loocv_crosscheck(dev, cand_by_id, selection, cfg))
    figures.spline_smoothness(dev, cand_by_id, selection, fig / "spline_smoothness.png")
    figures.feature_set_comparison(summary, fig / "feature_sets.png")
    write_json({"n_development_rows": len(dev), "n_development_stations": dev["station_id"].nunique(),
                "n_candidates": len(cands), "n_cv_cells": int(scores[["station_group", "day_fold"]]
                                                               .drop_duplicates().shape[0]),
                **extra}, od / "development_summary.json")
    print(selection[["family", "selected_id", "selected_cv_mae", "selected_cv_se", "best_cv_mae"]]
          .to_string(index=False))
    print(extra)


def stage_finalize(cfg):
    station_split, roles, _ = load_splits(cfg)
    sup = pd.read_parquet(out_dir(cfg, "features") / "supervised.parquet")
    selected = read_json(out_dir(cfg, "development") / "selected_specs.json")["selected"]
    cand_by_id = {c["id"]: c for c in models.candidates(cfg, station_split)}
    meta = evaluation.fit_frozen(sup, selected, cand_by_id, out_dir(cfg, "models"))
    write_json(meta, out_dir(cfg, "models") / "models_metadata.json")
    print("finalize:", {f: m["candidate_id"] for f, m in meta.items()})


def stage_evaluate(cfg):
    od, fig = out_dir(cfg, "test"), out_dir(cfg, "figures")
    station_split, roles, _ = load_splits(cfg)
    sup = pd.read_parquet(out_dir(cfg, "features") / "supervised.parquet")
    meta = read_json(out_dir(cfg, "models") / "models_metadata.json")
    frozen = {f: joblib.load(out_dir(cfg, "models") / f"{f}.joblib") for f in meta}
    before = {f: evaluation.param_hash(o["model"]) for f, o in frozen.items()}
    assert before == {f: m["param_hash"] for f, m in meta.items()}, "frozen model changed on disk"

    pred = evaluation.predict_frozen(sup, frozen)
    after = {f: evaluation.param_hash(o["model"]) for f, o in frozen.items()}
    assert before == after, "a model was modified during evaluation"
    pred.to_csv(od / "test_predictions.csv", index=False)
    evaluation.metrics(pred, ["split", "model_id"]).to_csv(od / "metrics_overall.csv", index=False)
    evaluation.metrics(pred[pred["test_season"].notna()], ["split", "model_id", "test_season"]).to_csv(
        od / "metrics_by_season.csv", index=False)
    evaluation.metrics(pred, ["split", "model_id", "station_id"]).to_csv(
        od / "metrics_by_station.csv", index=False)

    selected = read_json(out_dir(cfg, "development") / "selected_specs.json")
    cand_by_id = {c["id"]: c for c in models.candidates(cfg, station_split)}
    evaluation.forward_only_check(sup, selected["selected"], cand_by_id, roles).to_csv(
        od / "forward_only_fall_check.csv", index=False)
    evaluation.error_vs_distance(pred, station_split, selected["primary_model"],
                                 fig / "test_error_vs_distance.png")
    print(pd.read_csv(od / "metrics_overall.csv").to_string(index=False))


def stage_sensitivity(cfg):
    od = out_dir(cfg, "sensitivity")
    panel = pd.read_parquet(out_dir(cfg, "audit") / "daily_panel.parquet")
    station_split, roles, _ = load_splits(cfg)
    summary, best, coverage = sensitivity.run(panel, station_split, roles, cfg)
    summary.assign(params=summary["params"].map(json.dumps)).sort_values(["window", "cv_mae"]).to_csv(
        od / "window_candidate_scores.csv", index=False)
    best.to_csv(od / "window_best.csv", index=False)
    coverage.to_csv(od / "window_coverage.csv", index=False)
    sensitivity.window_figure(summary, out_dir(cfg, "figures") / "window_sensitivity.png")
    print(coverage.to_string(index=False))
    print(best[["window", "family", "selected_id", "selected_cv_mae", "best_id", "best_cv_mae",
                "best_cv_se", "primary"]].round(3).to_string(index=False))


STAGES = {"audit": stage_audit, "split": stage_split, "explore": stage_explore,
          "develop": stage_develop, "finalize": stage_finalize, "sensitivity": stage_sensitivity}
ONE_TIME = {"evaluate": stage_evaluate}

if __name__ == "__main__":
    cfg = load_config()
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name in (STAGES if which == "all" else [which]):
        {**STAGES, **ONE_TIME}[name](cfg)
