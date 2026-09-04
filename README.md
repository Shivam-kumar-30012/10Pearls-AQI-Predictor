# Pearls AQI Predictor

A three-day air quality forecast for **Ghotki, Sindh**, updated hourly and
retrained nightly. Built for the 10Pearls Shine internship, Data Science track.

**[Live dashboard →](https://10pearls-aqi-predictor-52wav2lhgysbpemqnqu63e.streamlit.app/)**

Three models cover the horizon — one per day, each predicting all 24 hours of
its block. Data collection, retraining, model registration and dashboard
refresh all run unattended through GitHub Actions.

| Horizon | R² | RMSE | MAE | vs. persistence |
|---|---|---|---|---|
| Day +1 (hours 1–24) | 0.932 | 16.5 | 8.3 | +0.024 |
| Day +2 (hours 25–48) | 0.716 | 33.9 | 22.6 | +0.034 |
| Day +3 (hours 49–72) | 0.613 | 39.5 | 26.1 | +0.069 |

Scores are on a held-out chronological test period (July 2025 onward) and are
always reported against a persistence baseline — predicting that the AQI stays
exactly as it is now. AQI is strongly autocorrelated, so that is a demanding
benchmark rather than a trivial one.

The full methodology, including a data-quality bug that had capped
performance, is in **[PROJECT_REPORT.pdf](PROJECT_REPORT.pdf)**.

---

## How it works

```
OpenWeather ──┐
              ├──► merge ──► features ──► Hopsworks ──► train ──► registry
Open-Meteo ───┘                          feature store           │
                                                                  ▼
                                                              dashboard
```

| Stage | What happens |
|---|---|
| **Collect** | Hourly pollutant and weather readings for Ghotki |
| **Engineer** | 71 features: EPA-compliant AQI, lags, rolling stats, weather derivatives |
| **Store** | Hopsworks feature group, 48,710 rows from December 2020 |
| **Train** | Ridge, Random Forest and XGBoost compared at every horizon |
| **Register** | Best model per horizon, versioned in the Hopsworks Model Registry |
| **Serve** | Streamlit dashboard reading a nightly snapshot |

---

## Running it yourself

### Prerequisites

- Python 3.11 (Hopsworks does not yet support 3.12+)
- A free [OpenWeather](https://openweathermap.org/api) API key
- A free [Hopsworks](https://www.hopsworks.ai/) account

### Setup

```bash
git clone https://github.com/Shivam-kumar-30012/10Pearls-AQI-Predictor.git
cd 10Pearls-AQI-Predictor

py -3.11 -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
OPENWEATHER_API_KEY=your_key_here
HOPSWORKS_API_KEY=your_key_here
```

Verify it loads:

```bash
python config.py
```

Both keys should report `True`.

### Building from scratch

Run in order. Steps 01 and 02 take a few minutes each; the rest are quick.

```bash
python pipeline/01_backfill_pollutants.py    # ~49,000 hourly readings
python pipeline/02_backfill_weather.py       # matching weather
python pipeline/03_merge_raw.py              # join on timestamp
python pipeline/04_build_features.py         # AQI + 71 features
python pipeline/05_upload_to_store.py        # → Hopsworks
python pipeline/06_train.py                  # days 2 and 3
python pipeline/06b_train_delta.py           # day 1
python pipeline/07_register_models.py        # → Model Registry
python pipeline/10_build_snapshot.py         # dashboard data
```

Then:

```bash
streamlit run app/dashboard.py
```

### Day to day

```bash
python pipeline/08_hourly_collector.py       # fetch new hours
python pipeline/09_predict.py                # 72-hour forecast
python pipeline/11_explain.py                # SHAP charts
python notebooks/eda.py                      # exploratory figures
```

Most scripts accept `--local` to skip Hopsworks and use the local CSV, which is
useful when the free tier is slow.

---

## Repository layout

```
config.py                       All settings; secrets read from .env
requirements.txt                Full environment, used by the workflows
src/aqi_calculator.py           EPA AQI, with regression tests

pipeline/
  01_backfill_pollutants.py     OpenWeather history, 2020 onward
  02_backfill_weather.py        Open-Meteo ERA5, matching period
  03_merge_raw.py               Join both sources on hourly UTC
  04_build_features.py          AQI + 71 features
  05_upload_to_store.py         → Hopsworks feature group v3
  06_train.py                   Absolute models (days 2 and 3)
  06b_train_delta.py            Delta model (day 1)
  07_register_models.py         → Hopsworks Model Registry
  08_hourly_collector.py        Incremental collection
  09_predict.py                 72-hour forecast
  10_build_snapshot.py          Dashboard data
  11_explain.py                 SHAP explanations

app/
  dashboard.py                  Streamlit application
  requirements.txt              Deployment dependencies only
  assets/                       10Pearls logo, light and dark

notebooks/
  eda.py                        Exploratory analysis
  figures/                      8 EDA charts + 6 SHAP charts

data/
  dashboard_snapshot.json       Committed; what the dashboard reads
  shap_summary.json             Committed; SHAP values for the dashboard
  raw/                          Downloaded API data (gitignored)
  processed/                    Merged and engineered data (gitignored)

models/
  metrics.json                  Scores for the absolute models
  metrics_delta.json            Scores for the delta model
  feature_cols.json             Feature order, absolute models
  feature_cols_delta.json       Feature order, delta model
  *.pkl                         Trained models (gitignored)

.github/workflows/
  hourly_collect.yaml           Every hour
  daily_train.yaml              02:00 UTC

archive/                        Earlier experiments, kept as a record
```

**Two `requirements.txt` files, deliberately.** The root one is the full
environment the pipeline needs. `app/requirements.txt` lists only the four
packages the dashboard uses — `hopsworks` depends on `confluent-kafka`, which
has no wheel for newer Python versions and broke the first deployment.

**What is and isn't committed.** The CSVs are gitignored: they total around
70 MB and are regenerable from steps 01–04. The `.pkl` models are gitignored
too — the day+1 XGBoost model alone is 9 MB, and `09_predict.py` downloads the
current version from the registry. The two JSON files in `data/` *are*
committed, because the deployed dashboard reads them directly.

**`archive/`** holds the earlier modelling scripts and EDA charts. They are not
part of the pipeline and are kept as a record of what was tried.

## Automation

Two workflows, both also runnable by hand from the Actions tab.

**Hourly** (`hourly_collect.yaml`) fetches new readings, builds features, and
writes to both the local CSV and the feature store.

**Nightly** (`daily_train.yaml`, 02:00 UTC) collects, retrains all four models,
registers them, rebuilds the dashboard snapshot and commits it back.

The collector fetches a *range* rather than a single hour — it asks each
destination for its newest timestamp and retrieves everything since. One code
path handles both the normal case and recovery after an outage, so a missed run
needs no special handling.

To use them in your own fork, add `OPENWEATHER_API_KEY` and `HOPSWORKS_API_KEY`
under **Settings → Secrets and variables → Actions**.

> **Note on scheduling.** GitHub's cron is best-effort on free accounts and
> typically fires three or four times a day rather than 24. Because the
> collector fetches a range, data completeness is unaffected — only latency.

---

## Deployment

The dashboard runs on Streamlit Community Cloud with `app/dashboard.py` as the
entry point and **Python 3.11** selected in the app settings. That version
matters: `hopsworks` fails to build on 3.12+, which is what broke the first
deployment.

Add both API keys under the app's **Secrets**.

The dashboard reads `data/dashboard_snapshot.json`, committed to the repo and
regenerated nightly. It does not query the feature store at page load — the
snapshot is rebuilt from the same models in the same run, so a live read would
return the same forecast several seconds slower, and during a Hopsworks outage
it froze the page for minutes.

---

## Known limitations

- **Day +3 accuracy.** At three days the mean absolute error reaches 26 AQI
  points, wide enough to cross a category boundary. Seven approaches were tried
  and all converged, which suggests the constraint is the information available
  rather than the method.
- **Observed rather than forecast weather.** Using forecast weather for the
  target day would be more correct, but Open-Meteo's archive returned only 55%
  coverage for wind and pressure at this location.
- **Hopsworks free tier** intermittently times out on reads. Every consumer
  retries and falls back to a local CSV.

Details and the reasoning behind each are in the project report.

---

## Built with

Python 3.11 · pandas · scikit-learn · XGBoost · SHAP · Hopsworks · Streamlit ·
Plotly · GitHub Actions

Data from [OpenWeather](https://openweathermap.org/) and
[Open-Meteo](https://open-meteo.com/). AQI computed to the US EPA standard
(EPA-454/B-24-002).
