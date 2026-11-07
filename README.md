# Next-Day Ozone Forecasting (Data Mining SDS 4971/5805 Final Project)

Forecast tomorrow's daily maximum 8-hour ozone (MDA8) at EPA monitoring stations from each
station's recent history, and test it on unseen days in every season and on unseen stations.
The full design is in [research_requirements_en.md](research_requirements_en.md).

## Run

```bash
pip install -r requirements.txt
# put daily_44201_2024.csv in the repo root (see data/README.md)
python run.py all          # audit -> split -> explore -> develop -> finalize
pytest tests/              # leakage and correctness checks
python run.py evaluate     # ONE-TIME final test on the frozen models
```

Outputs go to `outputs/` (figures in `outputs/figures/`).
