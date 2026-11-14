# Reproduced Ozone Forecasting Results

Four frozen models; 2021–2024 development and 2025 supplementary evaluation.

## Selected configurations

| family | lag | mean_window | std_window | parameter | selection_macro_mae |
| --- | --- | --- | --- | --- | --- |
| Lasso | 7 | 28 | 14 | 0.010000 | 5.554470 |
| OLS | 7 | 28 | 14 | 0.000000 | 5.554675 |
| Ridge | 7 | 28 | 14 | 0.001000 | 5.554722 |
| Spline_Ridge | 7 | 28 | 14 | 0.001000 | 5.556171 |

## 2025 evaluation

| family | group | mae | rmse | relative_mae | improvement_pct |
| --- | --- | --- | --- | --- | --- |
| Lasso | development | 5.614135 | 7.372098 | 0.914535 | 8.546509 |
| Lasso | held_out | 5.575288 | 7.347301 | 0.918015 | 8.198534 |
| Lasso | all | 5.606157 | 7.367012 | 0.915243 | 8.475655 |
| OLS | development | 5.614670 | 7.372960 | 0.914622 | 8.537799 |
| OLS | held_out | 5.575781 | 7.347844 | 0.918096 | 8.190421 |
| OLS | all | 5.606683 | 7.367809 | 0.915329 | 8.467067 |
| Ridge | development | 5.614908 | 7.372934 | 0.914661 | 8.533922 |
| Ridge | held_out | 5.576112 | 7.347918 | 0.918150 | 8.184967 |
| Ridge | all | 5.606941 | 7.367804 | 0.915371 | 8.462868 |
| Spline_Ridge | development | 5.613959 | 7.368975 | 0.914506 | 8.549371 |
| Spline_Ridge | held_out | 5.575559 | 7.344261 | 0.918059 | 8.194073 |
| Spline_Ridge | all | 5.606073 | 7.363907 | 0.915230 | 8.477026 |

Held-out stations provide observed prior ozone histories but never enter model fitting or tuning. Models are frozen throughout 2025. Differences among model families are small. No formal significance claim is made.
