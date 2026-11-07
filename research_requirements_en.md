# Research Requirements for Next-Day Ozone Forecasting

Version: v3.0 (decisions locked; replaces v2.0)
Updated: 2026-11-06
Course: Data Mining, SDS 4971/5805, Final Project

v3.0 replaces the v2.0 placeholders with what the data audit found, the decisions the group made,
and the methods taught in the course slides. Every number below comes from `python run.py all`
with `config.json` as committed. Changing a value means editing `config.json` and documenting it
here **before** the final test runs.

## 1. Research Objective

Use each monitoring station's recent ozone history to forecast tomorrow's ozone level, and measure
how well that works (a) on days the models never saw, in every season, and (b) at stations that
took no part in model development. Complex methods may fail to beat a simple baseline. That is an
acceptable, reportable result.

Research questions:

1. Do the regression methods from the course (OLS, Ridge, Lasso, splines, kNN, PCR) beat the
   persistence baseline ("tomorrow = today")?
2. How does accuracy differ by season, and between seen and unseen stations?

This is a predictive study and makes no causal claims. The v2.0 measurement-method extension and the
optional exceedance classifier are dropped; §11 explains why.

## 2. Locked Decisions

| Decision | Value | Where |
|---|---|---|
| Target | MDA8 ozone on day t+1, in ppb (`1st Max Value` × 1000) | §4 |
| Study window | 2024-01-01 – 2024-09-30 (October excluded: only 80 stations report) | §3 |
| Maximum lag | 7 days (confirmed by ACF/PACF, §7) | `max_lag_days` |
| Test design | One 7-day block per season, drawn with seed 42 | §6.1 |
| Embargo | 7 days after each test block (= maximum lag) | §6.1 |
| Stations | 188 eligible; 150 development, 38 held out (20%, stratified by cluster) | §6.2 |
| Selection | Blocked, station-grouped CV; one-standard-error rule | §9 |
| Primary metric | Station-macro MAE (mean of per-station MAEs) | §9 |
| Random seed | 42 everywhere | `random_seed` |

## 3. Data

Source: U.S. EPA Air Quality System (AQS), pre-generated daily file `daily_44201_2024.csv`
(ozone, parameter 44201). Download instructions are in `data/README.md`. The raw file is not
committed.

Audit results (`outputs/audit/data_audit.json`):

- 239,414 rows; 1,189 stations (State + County + Site); 53 states/territories; 2024-01-01 to
  2024-10-31 (305 dates).
- One measurement definition only: Sample Duration `8-HR RUN AVG BEGIN HOUR`, standard
  `Ozone 8-hour 2015`, ppm. `1st Max Value` is therefore the **daily maximum 8-hour average
  (MDA8)**, not an hourly peak.
- Coverage collapses after June because many monitors are seasonal. Active stations by month:
  Jan 767, Feb 1,096, Mar–Jun about 1,140–1,150, Jul 625, Aug 585, Sep 417, Oct 80.
- 40 stations run more than one instrument (POC); 6,751 station-days have two POCs (median
  disagreement 1 ppb).
- 2 station-days appear twice because of exceptional events (`Event Type` Included and Excluded).
- `Observation Count` is out of 17 hourly windows (96% of rows are multiples of 17), but
  `Observation Percent` ranges from 6 to 2400, so it is **not** used as a 0–100 completeness filter.
- 49 values are ≤ 0 (minimum −0.001 ppm). They are kept and flagged.
- 380 within-instrument method-code changes, almost all in Jan–Feb (see §11).

## 4. Forecasting Task

- **Target:** Y(i, T) = MDA8 at station i on target date T, in ppb.
- **Forecast origin:** t = T − 1. Inputs may use station data through day t, plus other
  development stations' day-t values. This assumes day-t summaries are available on day t. It is an
  experiment on archived data, not a claim about real-time deployment.
- **No future information:** no inputs from T or later; no future AQI or quality fields; no
  one-hot station IDs (they cannot transfer to unseen stations).

## 5. Cleaning Rules (`src/data.py`)

1. Keep only the single measurement definition above (excludes nothing in this file).
2. Keep only the study window; October rows are logged as excluded.
3. Event duplicates: when a station/POC/day has both `Included` and `Excluded` rows, keep
   `Included` (the as-measured value). Flag every event day (`flag_event`, 3,788 rows).
4. One POC per station: the POC with the most days outside the test blocks (ties go to the lowest
   POC). The other POCs' rows go to `excluded_rows.csv`, and all station-days with more than one
   POC go to `duplicate_conflicts.csv` with each POC's value.
5. No rows are dropped for quality. Rows are flagged with `flag_low_completeness`
   (Observation Count < 13 of 17; 2,956 rows) and `flag_nonpositive` (49 rows).
6. No imputation. A missing day makes any row that needs it non-evaluable (§7).

Result: `daily_panel.parquet`, with one row per station × date (229,927 rows).

## 6. Sample Design (`src/splits.py`)

### 6.1 Dates: seasonal test blocks with embargo

The group wanted every season tested, not just the end of the series. One contiguous **7-day test
block per season** is drawn uniformly within its season window (seed 42). Each block plus its
embargo must fit inside its window.

| Season | Window | Test block |
|---|---|---|
| Winter | Jan 8 – Feb 29 | **Jan 11–17** |
| Spring | Mar 1 – May 31 | **May 1–7** |
| Summer | Jun 1 – Aug 31 | **Jul 22–28** |
| Fall | Sep 1 – Sep 30 | **Sep 11–17** |

- **Why 7 days:** each block covers every weekday once (ozone has a weekday/weekend pattern). The
  block length is independent of the lag.
- **Embargo:** inputs reach back 7 days, so a training row whose target is within 7 days after a
  test block would carry test-day values as inputs. Those 28 days are removed from training. No
  gap is needed *before* a block, because a test row using pre-block history is ordinary forecasting.
- **Burn-in:** Jan 1–7 have no complete 7-day history, so they are not targets.
- **Day budget (274 days):** 211 development + 28 test + 28 embargo + 7 burn-in, so training vs
  test is about **88/12**. An exact 90/10 split isn't possible with four whole blocks, because
  embargo and burn-in are fixed costs.
- **Why blocks, not scattered days:** each isolated test day would need its own 7-day embargo,
  which would remove most of the training data.
- **Caveat (must be stated in the report):** development includes days after some test blocks, so
  this measures accuracy across seasons rather than a pure forward-in-time forecast. §10 adds a
  forward-only check on the fall block to measure the difference.

### 6.2 Stations: eligibility, clusters, holdout

- **Eligibility:** POC-1 panel station with ≥ 5 of 7 days in **every** test block and ≥ 200 days in
  Jan–Sep, giving **188 stations**. This uses test-period presence, which is stated openly. It
  favors year-round monitors (seasonal Mar–Jun monitors cannot qualify).
- **Station profiles** (non-test days only): mean MDA8 per season, SD, lag-1 autocorrelation,
  latitude, longitude; all standardized.
- **Clusters (Class 9–11):** k-means with K chosen by the Calinski–Harabasz index over K = 2..8.
  Both CH and silhouette pick **K = 2**: a Western group (higher ozone, lag-1 autocorrelation 0.83)
  and an Eastern group (lag-1 0.65).
- **Holdout:** 20% of stations (**38**), stratified by those clusters (seed 42). Held-out stations
  are 4–176 km from the nearest development station (median 26 km).
- **Cluster feature:** the scaler and centroids are re-fitted on the **150 development stations
  only**. Held-out stations are assigned to the nearest centroid.

### 6.3 Row roles

| Station role | Target date role | Use |
|---|---|---|
| development | development | Training and cross-validation |
| development | test | `temporal_test` (final test) |
| held_out | development | `spatial_test` (final test) |
| held_out | test | `spatiotemporal_test` (final test) |
| any | embargo / burn-in | Never a target; values serve only as later rows' inputs |

Held-out stations never contribute training rows, scaler statistics, spline knots, cluster
centroids, neighbor features, or hyperparameter decisions. At prediction time they use their own
past MDA8 as inputs, which is legitimate history.

## 7. Features (`src/features.py`)

Features are built on a calendar-complete date × station grid, so shifts are by calendar day and a
missing day is never treated as "yesterday".

| Feature | Definition |
|---|---|
| `lag1`..`lag7` | MDA8 on days t, t−1, …, t−6 |
| `roll_mean7`, `roll_sd7` | Mean and SD (ddof = 1) over t−6..t; all 7 days required |
| `doy_sin`, `doy_cos` | Phase of the target date's day of year (period 366; 2024 is a leap year) |
| `latitude`, `longitude` | Station location |
| `cluster_1` | Station-cluster indicator (cluster 0 is the reference) |
| `nbr_mean` | Mean day-t MDA8 at the 5 nearest **development** stations reporting that day (self excluded; haversine distance) |

Feature sets, each with lag set p ∈ {1, 3, 7}:

- **F1:** lags 1..p
- **F2:** F1 + `roll_mean7` (only when p < 7, since otherwise it is an exact average of the lags)
  + `roll_sd7`
- **F3:** F2 + season + location + cluster
- **F4:** F3 + `nbr_mean`

**Common sample:** every candidate is scored on the same rows, namely those where all features
and the target exist (≈ 96% of development rows).

**Lag choice evidence** (`outputs/figures/eda_acf_pacf.png`, development data only): after removing
each station's monthly mean, lag-1 ACF = 0.47, lag-2 PACF = −0.11, and later lags are inside the
per-station noise band (±0.13). Without that removal, PACF at lags 3–8 is 0.06–0.12; this is the
slow seasonal level, which `roll_mean7` and the season terms cover. A maximum lag of 7 is therefore
enough, and CV chooses among 1, 3 and 7.

## 8. Methods (`src/models.py`)

| Method | Slides | Role and tuning |
|---|---|---|
| Persistence | — | Baseline: prediction = `lag1` |
| OLS | Class 12–13 | Linear reference |
| Ridge | Class 14 | α on a 13-point log grid; coefficient paths and df(α) |
| Lasso | Class 15 | α on a 9-point log grid; coefficient paths, zeroed features |
| Penalized cubic regression spline | Class 16–17 | Spline basis on `lag1` (quantile knots, linear extrapolation) + other features linear + ridge penalty; tuned over n_knots ∈ {4, 6, 8} × α |
| Smoothing spline (illustration) | Class 17 | `scipy.interpolate.make_smoothing_spline`, λ by GCV, univariate tomorrow vs today |
| kNN regression | Class 1–2, 20 | Standardized features; k ∈ {5, …, 200} |
| PCR | Class 5–6 | PCA on standardized F4 (p = 7), number of components 1..14 |
| k-means / MDS | Class 7–11 | Station clusters (§6.2); MDS map of station profiles (exploratory) |

Trend Filtering (v2.0) is removed. It is not in the course slides, and it has no clean next-day
forecasting rule. It can be mentioned as future work.

## 9. Validation and Selection (`src/validation.py`)

- **Folds:** the 211 development days are cut into 14-day calendar blocks, assigned round-robin to
  5 day-folds so that every fold spans the year. The 150 development stations are split at random
  into 5 station groups. Each of the **25 cells** (station group × day-fold) validates on its
  stations × its days. It trains on the *other* stations, excluding its days and the **7 days after
  each of its blocks** (the same embargo logic as §6.1). Random K-fold would be invalid for
  time-ordered data (Class 17); blocking plus the embargo removes the overlap, and station grouping
  rewards generalization to new stations.
- **Score:** station-macro MAE per cell; CV score = mean over 25 cells; SE = SD / √25.
- **Within each family: one-SE rule** (Class 16). The best candidate sets a threshold of best + 1
  SE. Among candidates under it, pick the simplest: fewer inputs first, then stronger
  regularization (larger α, larger k, fewer knots, fewer components). This replaces v2.0's
  "within 1%" rule.
- **Primary model:** the one-SE rule applied again across the families' selections, choosing the
  fewest inputs, then the family with fewer tuning parameters (OLS < Ridge < Lasso < PCR < spline <
  kNN). This was fixed before any test data was used.
- **Cross-checks (not used for selection, since rows are not exchangeable):** ridge LOOCV shortcut
  (`RidgeCV`) and a GCV smoothing spline.
- After selection, each chosen spec is refit once on all development rows (Class 17). Model
  assessment (§10) is kept separate from model selection.

## 10. Final Fitting and Evaluation (`src/evaluation.py`)

1. `finalize`: refit the selected specs on all 30,270 development rows (150 stations), save them
   with joblib, and record a parameter hash.
2. `evaluate` (**one time only**; `run.py all` never runs it): load the frozen models, check the
   hashes, predict, and check the hashes again. No fit, partial_fit or re-estimation. Inputs update
   daily from observed history; the models do not.
3. Report three cells (§6.3) on common keys for all models and persistence:
   - Station-macro MAE, pooled MAE, RMSE, mean error (prediction − truth)
   - Improvement vs persistence = 1 − MAE_model / MAE_persistence (same aggregation)
   - Results overall, **per season**, and per station
4. **Forward-only check:** refit the selected specs on development rows with targets before Sep 11,
   score the fall block, and report the gap from the main result.
5. **Seen vs unseen:** compare `temporal_test` with `spatiotemporal_test` improvements, and plot
   held-out station error against distance to the nearest development station.
6. If a model returns NaN where a target exists, stop. Rows are never silently dropped.
7. The test is not used to reselect anything. If the design changes after the test, the new results
   are exploratory.

Each season's result rests on 7 days, so per-season numbers are indicative only. No significance
claims are made, because errors are correlated across days and stations.

## 11. Dropped Extensions

- **Measurement-method change:** almost all method changes happen in Jan–Feb, leaving no
  pre-change development period. Conclusion: "These data do not support this design."
- **Exceedance classification (MDA8 > 70 ppb, logistic vs LDA):** only 0.7% of development
  station-days exceed 70 ppb. That is too few events for a stable classifier, so it is dropped.

## 12. Limitations

- No meteorology. Temperature, sunlight and wind drive ozone, and these models see only past ozone.
- One year of data with no Oct–Dec, so "fall" is September only.
- Seasonal blocks are not a pure forward forecast (§6.1); the forward-only check sizes the effect.
- 7 days per season gives noisy per-season results.
- The station panel favors year-round monitors.
- `Observation Percent` is anomalous and unused; completeness is only flagged.
- The day-t summaries are assumed to be available at forecast time (§4).

## 13. Course Deliverables

Per the project description (SDS 4971/5805): a slide draft before the talk; a **30-minute group
presentation** (12–15 content slides, followed by Q&A); a **2–5 page report** (PDF or notebook
export, excluding figures and appendix); reproducible code with fixed seeds; and a cited data source
(§3). Each member writes and presents their own sections in their own words. Follow Canvas for
dates.

Suggested slide outline: question and data → cleaning challenges → split design (seasonal blocks,
embargo, held-out stations) → EDA and ACF → clusters/MDS → models → CV and one-SE → shrinkage paths
→ spline vs linear → test results vs persistence → seasons → seen vs unseen → limitations.

## 14. Code and Outputs

```text
config.json                 every tunable value
run.py                      audit | split | explore | develop | finalize | all ; evaluate (one-time)
src/data.py                 read, audit, clean -> daily_panel
src/splits.py               test blocks, embargo, eligibility, clusters, holdout, CV layout
src/features.py             calendar-complete lag / rolling / season / location / neighbor features
src/explore.py              EDA on development data
src/models.py               candidate pipelines + one-SE complexity keys
src/validation.py           blocked station-grouped CV, one-SE selection, primary model
src/figures.py              CV curves, shrinkage paths, df(alpha), spline smoothness
src/evaluation.py           freeze, one-time evaluation, metrics, forward-only check
tests/test_leakage.py       §15 checks (pytest)
outputs/{audit,splits,explore,features,development,models,test,figures}/
```

Environment: Python 3.12 with the versions pinned in `requirements.txt`. Large or regenerable
outputs (parquet files, models, prediction CSVs) are git-ignored; rerun `python run.py all`.

## 15. Verification (`pytest tests/`)

1. Lags equal the value from exactly k calendar days earlier, within the same station.
2. A missing day makes the following rows' lags and rolling features missing.
3. No development row's 7-day input window touches a test day; no CV training row's window touches
   that cell's validation days.
4. Held-out stations never appear in development rows or CV groups.
5. Changing a future value does not change any earlier row's inputs.
6. Frozen model parameter hashes are unchanged by evaluation.
7. Metric arithmetic and error sign are correct on a hand-checked example.
8. Splines extrapolate linearly, and their knots do not move at prediction time.
9. The neighbor feature uses development stations only and never the station itself.
10. Test blocks are reproducible and separated by their embargo; the one-SE rule picks the
    simplest model within one SE.
11. Reproducibility: rerunning `python run.py all` from an empty `outputs/` gives byte-identical
    splits, scores, selections and model hashes (checked 2026-11-06).

## 16. Status (2026-11-06)

| Step | Status |
|---|---|
| 1 Repo and requirements | Done |
| 2 Audit and clean | Done |
| 3 Split and lock | Done |
| 4 Explore | Done (figures in `outputs/figures/eda_*`) |
| 5–7 Features, models, CV | Done: 543 candidates × 25 cells |
| 8 Refit and freeze | Done |
| 9 Final test | **Not run.** Waiting for group sign-off on §9 selections |
| 10 Checks | Tests pass; reproducibility confirmed |
| 11 Report and slides | To do |

Development (CV) results, station-macro MAE in ppb (lower is better):

| Family | Selected spec (one-SE) | CV MAE ± SE | Best spec in family | Best CV MAE |
|---|---|---|---|---|
| Persistence | — | 6.03 ± 0.05 | — | — |
| **OLS (primary)** | F2, lag 1 (`lag1`, `roll_mean7`, `roll_sd7`) | 5.56 ± 0.06 | F4, lags 1..3 | 5.52 |
| Ridge | F2, lag 1, α = 316 | 5.56 ± 0.06 | F4, lags 1..3 | 5.52 |
| Lasso | F2, lag 1, α = 0.1 | 5.56 ± 0.06 | F4, lags 1..3 | 5.52 |
| Spline | F2, lag 1, 4 knots, α = 1 | 5.57 ± 0.06 | F4, lags 1..3 | 5.53 |
| PCR | F4, lags 1..7, 14 components (= OLS) | 5.52 ± 0.05 | same | 5.52 |
| kNN | F2, lag 1, k = 200 | 5.67 ± 0.07 | same | 5.67 |
