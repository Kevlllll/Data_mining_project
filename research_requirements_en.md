# Research Requirements for Next-Day Ozone Forecasting

Version: v2.0 (Implementation-Ready Specification)  
Compiled on: 2026-11-06  
Applicable project: SDS4425 / SDS5425 Final Project

## 1. Research Objective

Use historical ozone information from multiple monitoring stations to develop next-day concentration forecasting models and assess their out-of-sample predictive value and generalization across locations. The project emphasizes reliable data processing, model selection, and independent evaluation, and allows for the possibility that complex methods may not outperform a simple baseline.

Proposed research questions:

1. Can linear and nonlinear regression methods outperform a simple Persistence Baseline?
2. How does the final model perform in a subsequent time period and at stations that were not involved in development?
3. If the data support it, does predictive performance change after a change in measurement method? This third question is an optional extension, not a current core deliverable.

The project studies predictive relationships and does not infer causal relationships. Its relevance to careers in quantitative analysis lies in controlling information availability, preventing leakage, conducting out-of-sample experiments, and performing robustness analyses; ozone forecasting should not be presented as a trading strategy.

## 2. Agreed Principles and Open Decisions

### Agreed Principles

- The forecast horizon is one day: each day, predict the following day using information available at that time.
- Reserve the final test dates and unseen stations in advance; neither may be used for model selection.
- Use Expanding-window Walk-forward Validation during development to select model specifications.
- After final fitting, freeze the prediction function and preprocessing parameters; do not re-estimate coefficients during final evaluation.
- Update historical inputs daily during final testing, but do not update the model.
- Retain the Persistence Baseline and compare it fairly on the same test records.
- Generalization across locations is a core evaluation objective; whether to assess stability across measurement methods depends on the data.

### Open Decisions

- The final response variable and its measurement definition.
- Data inclusion rules, date boundaries, the proportion of held-out stations, and the sampling procedure.
- Candidate lags, feature combinations, model hyperparameters, and validation windows.
- The type of spline and the variables to which it will be applied.
- How Trend Filtering will generate next-day predictions and whether it will enter the formal comparison.
- Whether sufficiently reliable cases of measurement-method changes exist.

## 3. Data Sources and Availability (Data Requirements)

Current file: `daily_44201_2024.csv.pdf`. This is a PDF exported from a table, not an original CSV. Before formal analysis, prioritize obtaining the original machine-readable data and checking that they match the existing file.

A previous inspection of the observation-date field in the PDF indicated a range from 2024-01-01 to 2024-10-31, with 305 distinct dates; record counts vary across months. This range does not imply complete coverage at every station and should be reconfirmed after obtaining the original data.

Data checks must include at least:

- Station identifiers: State Code, County Code, and Site Num; preserve leading zeros in codes. POC distinguishes monitoring instruments or records and must not be treated directly as a separate location.
- Measurement definitions: Sample Duration, Pollutant Standard, and Units of Measure.
- Concentration and related fields: Arithmetic Mean, 1st Max Value, 1st Max Hour, and AQI.
- Coverage and quality: Observation Count, Observation Percent, missing dates, and the length of continuous records.
- Instruments and methods: Method Code, Method Name, and POC.
- Geographic information: Latitude, Longitude, and station and administrative-area names.

The source of multiple records for the same station on the same day must be explained before they are deleted or averaged. Verify the meaning of Observation Percent rather than assuming it must lie between 0 and 100. Do not fill missing concentration values with zero.

## 4. Forecasting Task (Problem Definition)

Provisional target: predict the maximum ozone concentration at station i on day t+1, denoted Y(i,t+1). Confirm that `1st Max Value` is consistent with the selected Sample Duration; if the underlying values are eight-hour rolling averages, do not describe the target as an hourly peak.

Forecast origin: assume that the required summaries for day t are already available. Verify publication delays when reporting the study. If they cannot be verified, state explicitly that this is a forecasting experiment based on historical archived data under the assumption that same-day summaries are available; do not claim that feasibility for real-time deployment has been established.

Candidate inputs:

| Category | Candidate Variables | Processing Requirements |
|---|---|---|
| Historical concentration | Concentrations for the current day and earlier dates | Construct lags by station using actual dates |
| Historical conditions | Rolling means and standard deviations | Windows must end at the forecast origin and exclude future information |
| Date | Day of year, month, etc. | Use appropriate encoding; avoid unjustified complexity |
| Location | Latitude and longitude | Check differences in spatial coverage between training and test data |
| Quality and method | Historical observation counts, Method, etc. | Initially use for quality checks and grouping; separately justify any inclusion in the model |

Do not use information unavailable at the forecast origin, such as the actual concentration, AQI, or actual observation count for day t+1. Unseen stations may provide historical inputs through day t, but may not participate in model development. Initially, do not use one-hot station encoding as the primary source of information for prediction across locations.

## 5. Research Stages and Methods (Research Workflow)

### Part 1 — Problem Definition

Define Y, candidate X variables, the forecast origin, and the research questions. Produce a variable-definition table and a statement of information availability.

### Part 2 — Data Preparation and Sample Design

Check measurement definitions, duplicates, missingness, and coverage; use these checks to determine the development dates, final test dates, development stations, and final held-out stations.

- Split dates according to the prediction target date, ensuring that all targets used for fitting are known at the forecast origin.
- Use the same final test dates for seen and unseen stations to facilitate comparison.
- Do not remove stations or change coverage rules because final test errors are large.
- If the station composition changes during evaluation, prioritize reporting results for the common set of stations and the actual sample sizes.

Produce cleaning rules, a coverage table, a sample-split table, and a record of the data-processing workflow.

### Part 3 — Time-series Exploration and Feature Engineering

Inspect time-series plots, ACF, PACF, concentration distributions, and plots of relationships among predictors using development data only. Analyze each station separately; do not concatenate records from different stations into a single continuous series. Use ACF/PACF to propose candidate lags, not to directly determine the optimal forecasting specification.

Construct a small number of justified candidate feature sets. Specify the validity requirements for lags and rolling windows when dates are missing. Produce a list of candidate X variables and lags, with reasons for their selection.

### Part 4 — Model Development and Selection

Candidate methods:

| Method | Role | Specifications to Determine |
|---|---|---|
| Persistence Baseline | Simple evaluation benchmark | Predict the next day's concentration as equal to the current day's concentration |
| Ridge Regression | Linear reference candidate | X variables, lags, and penalty strength |
| Spline Regression | Main nonlinear candidate | Which X variables receive splines, spline type, knots or smoothing strength, and extrapolation rules |
| Trend Filtering | Conditional candidate | Ordering variable, order, penalty strength, and future-prediction rule |
| OLS | Optional additional reference | Retain if scope permits; it is not a required additional model |

Splines and Ridge can be combined; do not assume they are mutually exclusive. If using Spline Basis + Ridge, clearly distinguish it from Ridge with ordinary linear features.

If Trend Filtering only fits historical trends, it cannot be treated as a completed forecasting model. If its output is used as a feature, it must be computed using only the history available at each development iteration; smoothing the full series before splitting is not permitted. If a clear, reproducible, leakage-free forecasting rule cannot be specified, move it to the exploratory section or remove it.

Use Expanding-window Walk-forward Validation during development to compare a limited set of candidate combinations. In every iteration, the training period must precede the validation target date. When making consecutive next-day forecasts within a validation segment, newly available historical inputs may be used each day.

Fit standardization, imputation, spline knots, and all other learned preprocessing on the training data within each iteration. Select parameters using the prespecified primary metric, MAE, with RMSE and performance across windows as supplementary evidence; prefer simpler specifications when performance is similar. An internal station holdout may be used as a development check of applicability across locations, without accessing the final test stations.

This stage compares validation prediction errors; changes in errors across windows must not be described directly as a statistical bias–variance decomposition.

### Part 5 — Final Fitting and Temporal Assessment

Refit the retained models on the available development data using the specifications selected during development, then freeze the prediction function and all preprocessing parameters.

Make consecutive next-day predictions for the development stations during the final future test period. Report errors by prespecified month or fixed time interval. This is a Frozen-model Temporal Holdout, not an Expanding-window Evaluation with retraining.

Temporal differences may arise from seasonality, concentration ranges, missingness, and station composition. An increase in error over time alone does not establish model aging. The Persistence Baseline provides a reference on the same dates.

### Part 6 — Cross-location Assessment

Apply the same final prediction function to stations that were entirely excluded from development, generating predictions on the common future test dates.

- Construct inputs from the known historical records of unseen stations, but do not use those records to estimate coefficients, knots, standardization parameters, or hyperparameters.
- Report both per-station metrics and an equally weighted summary across stations so that stations with more records do not dominate the results.
- Compare improvements over the baseline separately for seen and unseen stations, avoiding the misinterpretation of differences in station difficulty as a loss of generalization.
- Examine geographic distributions and feature-value ranges; conclusions do not automatically extend to new locations with no monitoring history.

### Part 7 — Interpretation and Limitations

Report model accuracy, value relative to the baseline, and limitations under different conditions. Use the final test only for evaluation, not to reselect lags, variables, parameters, or models. If the design is revised based on test results, the revised version must be treated as exploratory and requires new data for independent confirmation.

## 6. Optional Extension: Stability Across Measurement Methods

Proceed only if sufficiently reliable within-station method-change events exist and coverage before and after the changes is adequate.

1. Verify Method Code / Name, POC, and dates, ruling out synonymous names and multiple instruments operating in parallel.
2. Complete model development, parameter selection, and final fitting before the change.
3. Freeze the model and evaluate it in a pre-change holdout period and a post-change period.
4. Distinguish the initial transition period, when historical windows mix old and new methods, from the subsequent period when all historical inputs come from the new method.
5. Where feasible, use stations without a method change during the same period as a reference.

If the data only show different stations using different methods, descriptive comparisons of errors by group are possible, but must not be described as the effect of a method change, much less used to infer causality directly.

## 7. Evaluation Metrics and Figures (Metrics and Outputs)

- MAE: mean absolute error, the primary accuracy metric.
- RMSE: more sensitive to large prediction errors.
- Mean Prediction Error: defined as predicted minus actual; positive values indicate average overprediction and negative values indicate average underprediction.
- Relative Improvement: for example, MAE improvement rate = 1 − model MAE / baseline MAE; do not calculate this ratio when the baseline error is zero.
- Plots of actual and predicted values during the test period, and error plots by period and station.
- Residual Plots / ACF: inspect remaining patterns; do not use these directly for causal interpretation or for significance claims based on an assumption of independent errors.
- State the number of stations, date range, and effective sample size in every results table; use common records for models and the baseline.

## 8. Deliverables and Acceptance Criteria

Deliverables:

1. Reproducible cleaning and analysis notebooks / scripts with fixed random seeds.
2. A report on field definitions, data coverage, and cleaning.
3. Explicit documentation of sample splits and information availability.
4. Candidate specifications and selection results from development.
5. Final temporal and cross-location test tables, figures, and discussion of limitations.
6. The report and presentation slides required by the course; follow the current Canvas requirements for length and deadlines.

Acceptance criteria:

- Clearly explain when each prediction is generated and what information is used at that time.
- Final test dates and stations are excluded from development decisions.
- All learned preprocessing uses only the permitted training data.
- The model and preprocessing remain frozen during final evaluation; inputs may be updated as new observations become available.
- Report results honestly whether or not the model outperforms the baseline; predictive improvement is not a necessary condition for project success.
- If Spline / Trend Filtering enters the formal comparison, it must have explicit next-day forecasting and extrapolation rules.

## 9. Immediate Priorities

1. Obtain the original CSV and confirm field definitions, the actual date range, and station coverage.
2. Use common coverage to determine the prediction target Y, included stations, and date splits.
3. Clarify the specific forms of Spline and Trend Filtering recommended by the professor, and limit the number of candidate models.
4. Complete exploration of the development data before finalizing candidate features and lags.

## 10. Implementation Configuration and Default Plan

This section translates the research design above into coding rules. The defaults below are starting points for implementation, not conclusions already established from the data. If date boundaries and measurement definitions have not yet been confirmed, the data audit may run, but the program must stop before formal training rather than silently guessing.

| Configuration Item | Default Plan / Requirement |
|---|---|
| `raw_data_path` | Path to the original CSV, which is currently unavailable; do not rename a PDF extension to CSV and use it |
| `target_column` | Provisionally `1st Max Value`, to be confirmed after checking the field's measurement definition |
| `sample_duration`, `pollutant_standard`, `units` | Explicitly specify consistent measurement definitions; audit available values first |
| `forecast_horizon_days` | Fixed at 1 |
| `test_start`, `test_end` | Explicitly specify based on the coverage audit; do not assume October is complete |
| `holdout_station_fraction` | Initial plan: 0.20; if there are too few stations, revise the plan first and document the reason |
| `random_seed` | 42; persist the list of held-out stations and do not resample it later |
| `lag_candidates` | Initial candidates: the most recent 1, 3, or 7 days; may be revised and documented during development after ACF/PACF analysis |
| `rolling_window_days` | Initial candidate: 7 days; this is not the training-window length |
| `initial_train_days` | Initial plan: 60 calendar days; confirm after checking valid target counts and station coverage; do not automatically use 30 days |
| `validation_block_days` | Default: 1 day |
| `validation_refit_frequency` | Default: refit on every validation day |
| `training_window_type` | `expanding`: fixed start and advancing end, retaining older training samples |
| `primary_metric` | Validation-period station-macro MAE (the equally weighted mean of per-station MAEs) |
| `secondary_metrics` | pooled MAE, RMSE, mean prediction error, and results by window |
| `final_refit` | Fit once on all permitted development stations and development target dates |
| `test_refit` | False; updating model or preprocessing parameters during final testing is prohibited |

Determine the station split first using identifiers and development-period coverage rules; do not decide inclusion based on future concentrations or future errors. The presence of observations during the test period may be checked to describe evaluation coverage, but such screening must not be presented as fully prospective station selection. If necessary, report deployment-candidate stations separately from stations that can actually be evaluated.

## 11. Cleaned Data and Modeling Sample Contracts (Data Contracts)

### 11.1 Daily Panel `daily_panel`

The main analysis requires at most one row per `(station_id, date)`:

| Internal Column Name | Meaning |
|---|---|
| `station_id` | Concatenation of normalized state, county, and site codes, for example preserving widths of 2/3/4 digits, subject to confirmation against the field definitions |
| `date` | `Date Local`, parsed as a local calendar date |
| `ozone_max` | Maximum value under a consistent measurement definition |
| `ozone_mean` | Mean value from a record with the same measurement definition; may be missing |
| `latitude`, `longitude` | Geographic coordinates |
| `method_code`, `method_name`, `poc` | Information retained for auditing and optional grouping |
| `observation_count`, `observation_percent` | Original quality fields; do not alter their numerical definitions without justification |

Cleaning sequence: preserve the original file → normalize column names and types → filter by units and statistical definitions → check for primary-key conflicts → establish rules for handling instruments within stations → produce unique daily records.

Only exact duplicate records with identical values in every retained field may be directly deduplicated. Conflicts caused by multiple POCs or record types must be exported to a conflict table; do not default to `drop_duplicates(keep='first')` or averaging. Block the main model from running if conflicts cannot be explained. Explicitly document rules for handling missing latitude or longitude as well.

### 11.2 Supervised Samples `supervised_samples`

Each row represents one prediction task, with primary key `(station_id, target_date)`:

- `origin_date = t`: the date on which the prediction is made.
- `target_date = t + 1 calendar day`.
- `y_true = ozone_max(target_date)`.
- `max_lag_1 = ozone_max(t)`; `max_lag_2 = ozone_max(t-1)`, and so on. Names are defined relative to the target date.
- If `lookback_days = p`, inputs use t, t−1, …, t−p+1; do not include an extra day.
- Seven-day rolling features use t−6 through t, including t and excluding t+1.
- Date features are based on `target_date`; location features are based on the station.

Single-station example with lag=5: days 1–5 form X and day 6 is Y; days 2–6 form the next X and day 7 is the next Y. The training set contains many such complete rows; the first five days themselves are not the training set.

In implementation, complete the calendar index within each station before applying shift / rolling, or use explicit date-based joins. Do not apply row-based shifts directly to records that skip missing dates. The initial version does not impute concentrations: a sample with a missing target or required historical input is not evaluable, and its exclusion reason must be reported. Rolling means and standard deviations require values on all seven days by default; use `ddof=1` for standard deviations consistently in training and prediction.

Default feature sets may be defined as follows:

- F1: maximum concentrations over the most recent p days.
- F2: F1 + the seven-day mean and standard deviation of maximum concentrations.
- F3: F2 + sin/cos of the target date's phase within the year + latitude and longitude.
- Historical mean concentrations may form an optional feature set later in development; do not include every field in the model by default.

Define the annual phase using the actual number of days in the year; 2024 is a leap year. The initial version does not use future AQI, future quality fields, or one-hot station encoding. To avoid exact feature redundancy, if a rolling mean is an exact linear combination of a set of lags, remove the redundancy in OLS or do not enable that combination. Ridge can handle correlated features, but duplicate features must still be checked.

## 12. Sample Splits and Information Boundaries (Split Rules)

First assign fixed labels by station and target date:

| station_role | target_date | sample_role |
|---|---|---|
| development | Before `test_start` | development: used for exploration, training, and internal validation |
| development | Within the test period | temporal_test |
| held_out | Within the test period | spatial_test |
| held_out | Before the test period | history_only: permitted only for constructing prediction inputs for that station |

Split training and test data by target date; historical inputs do not have to belong to the same split. Inputs for the first test day may come from the development period; the second test day may also use the concentration observed on the first test day. This does not trigger refitting.

Use only development stations during validation. No date from an unseen station may be used for parameter selection, standardization, knot determination, or error-driven decisions. Do not use final test performance to tune models.

Validate candidate models on a predetermined common set of validation target rows: determine the common valid rows from the historical windows required by all enabled candidates, preventing unfair comparisons caused by different available dates for different lags. Report coverage losses caused by the common-sample restriction.

## 13. Expanding-window Validation Execution Rules

Distinguish three windows:

1. Lookback window: the number of days included in one X; this may be a candidate p and remains fixed within a specification.
2. Training window: the range of historical samples used to estimate the model; its start stays fixed while it continually expands.
3. Validation window: one day by default; its length does not increase.

For validation target date d:

1. The forecast origin is after the summaries for day d−1 become available.
2. Training samples contain only known targets from development stations with `target_date <= d−1`.
3. Validation samples have `target_date == d`; X may use only observations from d−1 or earlier.
4. Refit each candidate pipeline on this iteration's training data, including standardization, spline knots, and other preprocessing.
5. Save each station's prediction for that day; once the actual value becomes available, use it only for scoring at that iteration, after which it becomes available history for the next iteration.
6. Advance d by one day in the next iteration, retaining all previous training samples.
7. d must not enter the final test period; validation ends on the last target day of the development period.

Example: with lag=5, the first iteration trains on samples whose targets are days 6–30 and validates on day 31; the second trains on targets from days 6–31 and validates on day 32. The 31 days are neither nonoverlapping blocks nor a seasonal cycle.

If the sample size makes daily fitting too expensive, explicitly change the validation update frequency to weekly before the formal run, freezing the model within each segment and predicting daily. Document this as a different execution specification; do not change it after final testing to improve results.

Check the minimum valid target count and whether the design matrix permits model estimation in each development iteration. If requirements are not met, report this and postpone the first validation day. Do not silently remove unfavorable windows merely because a candidate model fails.

## 14. Model Pipelines and Selection Rules

### Required Initial Implementation

1. Persistence: `prediction = max_lag_1`; no training.
2. Linear Ridge: numerical standardization + intercept + Ridge; an initial alpha grid may be 0.01, 0.1, 1, 10, 100. Record the library's objective-function definition; alpha values are not directly comparable across different implementations.
3. Spline + Ridge: apply a cubic B-spline expansion to specified continuous concentration features, then fit using Ridge; date sin/cos and latitude/longitude enter directly by default. For now, do not fit a calendar-time trend that is extrapolated indefinitely into the future.

Initial spline candidates may use 3 or 5 knots and the same alpha grid. Derive knots only from the training data in each iteration, with an explicit linear extrapolation rule beyond the boundaries. The number of knots, interpolation degree, and alpha are distinct parameters. Variables that are constant or have too few unique values must explicitly fall back to linear features, or the candidate must be stopped and the reason recorded; invalid basis functions must not be generated. This specification is a regression spline basis expansion with coefficient regularization and must not be mislabeled as a classical smoothing spline.

OLS may be included as a supplementary reference, but is not required merely to increase the number of models. Until a forecasting rule is confirmed, retain only an interface / research note for Trend Filtering; do not claim it has been implemented or compared in the default run. If the professor explicitly requires it, first add separate specifications for its data inputs, future-prediction rule, parameter tuning, and application across locations.

Keep the number of candidates limited, and save the complete feature and parameter specification for every `candidate_id` in the configuration file. Aggregate each candidate's predictions over the entire validation period. Compute the primary score by calculating each station's MAE and then taking an equally weighted average; do not select models using errors from training sets whose sizes differ across iterations.

For each model family, select the specification with the lowest primary validation score, then use that same score to designate the primary model before final testing. If primary scores are within 1% of the best score in relative terms, a preference for shorter lags and fewer features / knots may be specified in advance. This threshold is a simplifying decision rule, not a test of statistical significance or the one-standard-error rule. Retain the full rankings and results by window.

The final test report must include the primary model and the reference models retained in advance; do not retroactively declare another model the development-stage winner based on test results. Overlapping samples and validation errors are correlated, so the number of validation iterations must not be treated as the number of independent experiments.

## 15. Final Fitting, Prediction, and Evaluation Execution Rules

1. Refit the retained specifications using all permitted development stations and samples with `target_date < test_start`.
2. Save the complete model pipeline, configuration, development dates, and station list.
3. From `test_start` through `test_end`, advance one day at a time, read historical observations known through origin_date, and generate temporal_test and spatial_test predictions separately.
4. Do not call fit / partial_fit or re-estimate means, variances, knots, or hyperparameters.
5. Compute errors using actual targets, but do not use those targets to change the prediction rule.
6. Compare all retained models and the baseline on the same set of valid test primary keys. Also report each model's original prediction coverage so that the common sample does not conceal model failures.
7. If an actual value exists but a candidate model produces NaN or an exception, record the failure and stop or mark the comparison as invalid; do not silently discard the record.

The test prediction table must include at least: `station_id, origin_date, target_date, split, model_id, y_true, y_pred, baseline_pred, error, absolute_error, squared_error`. Here, `error = y_pred - y_true`. Metadata such as Method may be included as additional columns.

Output overall pooled metrics, equally weighted station-level metrics, per-station metrics, and monthly metrics. Calculate relative improvement using model and baseline metrics with the same aggregation method. Indicate counts for groups with small samples and do not claim statistically significant differences based on an assumption of independence. If the test does not span multiple months, report only the period actually covered; do not describe the results as evidence of stability across seasons or years.

The optional Method analysis cannot directly use a main model that has already seen post-change data. If no eligible events exist, report the audit conclusion, “These data do not support this design,” rather than forcing an analysis.

## 16. Code Organization, Execution Stages, and File Outputs

A basic modular Python implementation is recommended, with a notebook presenting the study in research order; do not build a website, service, or complex framework.

Suggested structure:

```text
config.json
src/
  data.py             # Reading, measurement-definition filtering, primary-key and coverage audits
  features.py         # Calendar-based feature and target construction within stations
  splits.py           # Fixed station labels, target-date labels, validation iterations
  models.py           # Baseline, linear ridge, and spline ridge pipelines
  validation.py       # Expanding-window validation and candidate selection
  evaluation.py       # Frozen-model predictions, metrics, and grouped summaries
run.py                # audit / develop / finalize / evaluate
analysis.ipynb        # Presentation of figures, methods, and results
outputs/
  audit/              # Fields, conflicts, coverage, and exclusion reasons
  development/        # Candidate configurations, predictions by iteration, validation rankings
  models/             # Frozen pipelines and training metadata
  test/               # Final predictions and evaluation tables
  figures/
```

Prefer pandas, numpy, scikit-learn, statsmodels, matplotlib, and joblib as dependencies; there is no requirement to invent algorithms or install deep-learning frameworks. Record the specific library versions during implementation.

Execution stages:

- `audit`: inspect structure and coverage; do not fit models or select final dates.
- `develop`: after the configuration is confirmed, use development samples only to produce exploratory and validation results.
- `finalize`: read the selected specifications, fit the models, and save the frozen models.
- `evaluate`: load frozen models only and generate final predictions and results; do not automatically return to tuning.

Core outputs: `data_audit.json`, `station_coverage.csv`, `duplicate_conflicts.csv`, `excluded_rows.csv`, `station_split.csv`, `date_split.json`, `feature_dictionary.csv`, `validation_folds.csv`, `validation_predictions.csv`, `candidate_scores.csv`, `selected_specs.json`, frozen model files, `test_predictions.csv`, `metrics_overall.csv`, `metrics_by_station.csv`, `metrics_by_month.csv`, and `run_metadata.json`.

All names are suggestions, and the file structure may be simplified, but every category of information must remain traceable. Original files are read-only; outputs must not overwrite the raw data.

## 17. Required Checks and Acceptance (Verification)

During implementation, write only the essential checks needed to verify the validity of the research:

1. Use a small, manually verifiable panel to confirm that days 1–5 predict day 6 and that inputs do not cross station boundaries.
2. Remove an intermediate day and confirm that lags do not treat the previous nonadjacent record as yesterday.
3. Check that training target dates in every iteration strictly precede the validation target date and that none fall within the test period.
4. Check that training, validation, and parameter-selection records do not include held_out stations.
5. Modify the concentration on a future date and confirm that features and predictions for earlier dates do not change.
6. Check that frozen model parameters and preprocessing parameters remain unchanged before and after final testing begins.
7. Use a small example to verify MAE, RMSE, error direction, and improvement over the baseline.
8. Check that spline behavior outside the training range follows the specified rule and that knots are not reselected using the test distribution.
9. Check that all model comparisons use the same primary keys, and output coverage and failure counts.
10. Rerun from a clean output directory using the fixed configuration and obtain identical splits and numerically reproducible results.

Completing this requirements document does not mean that the CSV has been obtained, cleaning has been completed, or models have been trained. The first deliverable of formal implementation should be the data audit; only afterward should dates, station inclusion rules, and model candidates be finalized. Any changes to default values must be entered in the configuration and documented before final testing.
