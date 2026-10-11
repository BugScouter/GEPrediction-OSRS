# Modernized OSRS Grand Exchange Price Predictor & Flipping Screener

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](notebooks/ge_predictor_colab.ipynb)

A modernized machine learning pipeline for predicting Old School RuneScape (OSRS) Grand Exchange prices and identifying optimal 2% GE tax-adjusted flipping opportunities using official OSRS Wiki Real-time Prices API and LightGBM.

## Key Features

- **Official OSRS Wiki API Integration**: Uses real-time high/low snapshots and 5m/1h historical price/volume aggregates.
- **Accurate 2% GE Tax & Trading Rules**:
  - 2% tax rate on seller's gross price rounded down (`math.floor(sell_price * 0.02)`).
  - 0 tax on items selling below 50 GP or tax-exempt items (e.g., Old School Bonds).
  - Capped at 5,000,000 GP max tax per item (reached at 250,000,000 GP).
  - Calculates net profit margins and 4-hour buy limit constraints.
- **Stationary Feature Engineering**: Log returns, EMA ratios (EMA-5 / EMA-20), rolling volatility, relative high-low spreads, volume momentum, and cyclical hour/day sine-cosine encodings.
- **LightGBM Time-Series Forecasting**: Walk-forward cross-validation (`TimeSeriesSplit`) to evaluate RMSE and directional accuracy without lookahead bias.
- **Turnkey Google Colab Notebook**: Self-contained cloud training and interactive Plotly visualizer.
- **FastAPI REST Service**: Light REST endpoints for item search, predictions, and flip screening.

---

## Quickstart

### Option 1: Open in Google Colab
Click the badge above or navigate to [`notebooks/ge_predictor_colab.ipynb`](notebooks/ge_predictor_colab.ipynb) to execute the notebook in Google Colab.

### Option 2: Local Installation & Execution

#### 1. Install Package
Requires Python 3.11+:

```bash
pip install -e .[test]
```

#### 2. Run Tests

```bash
pytest
```

#### 3. Start REST API

```bash
uvicorn ge_predictor.api:app --reload --port 8000
```

Access interactive API docs at `http://localhost:8000/docs`.

---

## API Endpoints

- `GET /health` : API status and version.
- `GET /items?query={name}` : Search item metadata catalog.
- `GET /predict/{item_id}` : Get real-time prices, LightGBM price forecast, tax, margin, and recommendation (BUY/SELL/HOLD).
- `GET /screener?budget={gp}&min_roi={pct}` : Find top tax-adjusted flips matching budget and liquidity parameters.

---

## Project Architecture

```text
ge_predictor/
├── __init__.py
├── client.py          # OSRS Wiki Real-Time Prices API client
├── features.py        # Stationary feature engineering & time features
├── models.py          # LightGBM time-series regressor with walk-forward CV
├── market_rules.py    # 2% GE tax, 4-hour buy limits, net margins
├── scoring.py         # Flipping screener & cash-stack optimizer
└── api.py             # FastAPI REST endpoints
notebooks/
└── ge_predictor_colab.ipynb  # Turnkey Google Colab notebook
tests/
├── test_client.py
├── test_features.py
└── test_market_rules.py
pyproject.toml
README.md
```
