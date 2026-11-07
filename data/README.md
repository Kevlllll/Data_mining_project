# Data

The raw file is not committed (79 MB). To reproduce the analysis:

1. Download the EPA AQS pre-generated daily ozone file for 2024:
   <https://aqs.epa.gov/aqsweb/airdata/download_files.html#Daily> → "Ozone (44201)", year 2024
   (`daily_44201_2024.zip`). The course also distributed it on Canvas (Week 7, "Daily Ozone Data").
2. Unzip it and place `daily_44201_2024.csv` in the **repository root** (the path is set by
   `raw_data_path` in `config.json`).

The copy used for this project covers 2024-01-01 to 2024-10-31 (239,414 rows, 1,189 stations).
A later download may contain more dates; the study window in `config.json` (Jan 1 – Sep 30)
keeps the analysis fixed either way.

Source: U.S. Environmental Protection Agency, Air Quality System (AQS), pre-generated data files.
