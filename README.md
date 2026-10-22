# ICT Liquidity Intelligence & XGBoost Stop-Hunt Detection System

An enterprise-grade, modular quantitative research and execution system for detecting resting liquidity sweeps, stop-loss hunts, and reversal follow-through in **XAUUSD (Gold)** and **XAGUSD (Silver)**.

The system combines ICT Killzones and session high/low levels with algorithmic Double-Bottom, Triple-Bottom, and Trendline detection, a leakage-free feature pipeline, an XGBoost classifier, a Pine Script v6 TradingView indicator, and an interactive Streamlit research terminal.

---

## 1. System Architecture

```mermaid
flowchart TD
    subgraph Data ["Data Layer"]
        A1["Dukascopy / Broker M15 CSVs"] --> A2["Data Loader & Validator"]
        A2 --> A3["Core Technical Indicators (ATR, Ratios, PDH/PDL)"]
    end

    subgraph Detection ["Liquidity Detection Engine"]
        A3 --> B1["Option B Session Levels (Asia, London, NY)"]
        A3 --> B2["Confirmed Swings Detector (k-lag, No Look-Ahead)"]
        B2 --> B3["Double Bottom Detector (|L1-L2| <= 0.25 ATR)"]
        B2 --> B4["Triple Bottom Detector (3 Touches, <= 0.25 ATR)"]
        B2 --> B5["Ascending Trendline Detector (>= 3 Touches)"]
    end

    subgraph Labeling ["Outcome Evaluation & Labeling"]
        B1 & B3 & B4 & B5 --> C1["Sweep & Reclaim Engine"]
        C1 --> C2["Labeling Engine: 1 (Hunt Reversal), 0 (Non-Hunt), Breakdown"]
    end

    subgraph ML ["Machine Learning Pipeline"]
        C2 --> D1["Zero-Leakage Feature Pipeline (21 Features)"]
        D1 --> D2["Chronological Split (60% Train, 20% Val, 20% Test)"]
        D2 --> D3["XGBoost Ensemble Classifier"]
        D3 --> D4["Probability Calibration (Sigmoid on Val Set)"]
        D4 --> D5["Artifacts: Model Binary, Schema, Metadata"]
    end

    subgraph Outputs ["Execution & Dashboards"]
        D5 --> E1["Streamlit Trading Terminal (7 Interactive Pages)"]
        D5 --> E2["FastAPI Webhook & Inference Server"]
        D5 --> E3["Realistic Cost Backtester (Spread, Slippage, Fees)"]
        E2 <--> E4["TradingView Pine Script v6 Indicator"]
    end
```

---

## 2. Directory Structure

```text
ict-liquidity-intelligence/
├── app.py                             # Interactive Streamlit Quantitative Dashboard
├── config.yaml                        # Central configuration parameters
├── requirements.txt                   # Production Python dependencies
├── .env.example                       # Environment variables template
├── run_pipeline.py                    # End-to-end data processing & model training
├── prepare_raw_data.py                # Ingestion and alignment script
├── data/
│   ├── raw/                           # Raw M15 OHLCV CSVs (XAUUSD, XAGUSD)
│   └── processed/                     # Engineered feature tables & trade logs
├── models/
│   ├── model_shared.joblib            # Trained ensemble model artifact
│   ├── calibrated_shared.joblib       # Probability calibration artifact
│   ├── feature_schema.json            # Exact input feature schema
│   └── training_metadata.json         # Performance metrics & hyperparameters
├── docs/
│   ├── DATASET_SCHEMA.md              # Full feature & label data dictionary
│   ├── MODEL_EVALUATION_REPORT.md     # Out-of-sample metrics & transfer tests
│   └── BACKTEST_REPORT.md             # Backtest performance & risk breakdown
├── pine/
│   └── ict_liquidity_intelligence.pine# Complete Pine Script v6 indicator
├── src/
│   ├── __init__.py
│   ├── data_loader.py                 # Multi-format CSV ingestion & chronological splitting
│   ├── validation.py                  # Data quality reports & OHLC consistency checks
│   ├── indicators.py                  # ATR, wick/body ratios, PDH/PDL without look-ahead
│   ├── swing_detection.py             # Confirmed swing points with strict k confirmation
│   ├── liquidity_patterns.py          # Double Bottom, Triple Bottom, and Trendline detection
│   ├── session_liquidity.py           # Option B overlapping session extremes & confluence
│   ├── feature_engineering.py         # Tabular feature generator with zero leakage
│   ├── labeling.py                    # Sweep, reclaim, breakdown, and horizon labeling
│   ├── train.py                       # XGBoost training, calibration & cross-instrument transfer
│   ├── predict.py                     # Inference engine & confidence tiers
│   ├── evaluation.py                  # Classification metrics, PR-AUC, calibration curve
│   ├── backtest.py                    # Realistic portfolio simulator with transaction costs
│   ├── storage.py                     # SQLite persistence for patterns, events, predictions
│   ├── api.py                         # FastAPI inference & TradingView webhook receiver
│   └── alerts.py                      # Alert dispatching & duplicate signal suppression
└── tests/
    ├── __init__.py
    ├── test_swings.py                 # Swing confirmation lag & visibility tests
    ├── test_patterns.py               # Pattern detection & invalidation tests
    ├── test_features.py               # ATR & anatomy ratio bounding tests
    ├── test_labels.py                 # Sweep, reclaim, and breakdown labeling tests
    ├── test_leakage.py                # Mutation test proving zero look-ahead bias
    └── test_backtest.py               # Simulated execution & cost accounting tests
```

---

## 3. Existing Indicator Integration & Pine Script v6 Audit

The indicator in `pine/ict_liquidity_intelligence.pine` preserves 100% of the supplied script (`ICT_Killzones_Liquidity_High_Low.pine`) and appends independent modular extensions:

- **Original Features Preserved:** Killzone boxes (Asia, London, NYAM, NYL, NYPM, RTH), session high/low liquidity lines, pivots with mitigation tracking, range levels, DWM (Daily, Weekly, Monthly) levels, opening price horizontal lines, timestamps, and data tables.
- **New Modular Settings Added:**
  - `Liquidity: Double Bottoms`: Equal-low tolerance ($0.25 \times \text{ATR}$), swing lookback $k=3$, separation $[5, 60]$, neckline height $\ge 1.5 \times \text{ATR}$.
  - `Liquidity: Triple Bottoms`: 3-touch tolerance, maximum duration 120 bars, automatic re-labeling.
  - `Liquidity: Ascending Trendlines`: $\ge 3$ touches, slope filter ($0.01 - 1.2 \text{ ATR/bar}$), no close below line.
  - `Liquidity: Sweep & Reclaim Engine`: Sweep depth $\ge 0.1 \times \text{ATR}$, reclaim window $\le 3$ bars, visual hunt reversal tags.
  - `XGBoost Machine Learning Integration`: Live status widget, model probability display, threshold alerts.
- **Important Technical Reality:** Pine Script cannot execute Python machine learning models natively. Model scores are supplied via external alert bridges/webhooks or user inputs, and displayed alongside visual liquidity zones.

---

## 4. Setup and Execution Guide

### Prerequisites
- Python 3.11, 3.12, or 3.13
- Git

### Installation
```bash
# 1. Clone or navigate to the project directory
cd "d:/KOLADILIP K/Internship/slhl"

# 2. Install dependencies
pip install -r requirements.txt
```

### Running the Complete Pipeline
To ingest data, compute indicators, extract patterns, generate features, train the calibrated model, run cross-instrument tests, and execute the backtest:
```bash
python run_pipeline.py
```

### Launching the Streamlit Research Terminal
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to access the 7-page interactive trading dashboard.

### Launching the FastAPI Webhook Server
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```
Interactive API docs available at `http://localhost:8000/docs`.

### Running Automated Tests
```bash
python -m unittest discover tests -v
```
All 11 unit and integration tests run and pass in under 0.05 seconds.

---

## 5. Connecting TradingView Alerts to the Python Service

### Step 1: Start the Webhook Receiver
Ensure the FastAPI server is running and accessible (via localhost, VPS, or tunnel like ngrok/Cloudflare):
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

### Step 2: Configure the Alert on TradingView
1. Open TradingView and add `pine/ict_liquidity_intelligence.pine` to your XAUUSD or XAGUSD chart.
2. Click **Create Alert** (Alt + A).
3. Under **Condition**, select `ICT Killzones & Pivots [TFO]`.
4. Choose the condition: `Stop-Hunt Reclaim (Long)` or `Any alert() function call`.
5. Under **Webhook URL**, check the box and enter:
   ```text
   http://YOUR_SERVER_IP:8000/webhook/tradingview
   ```
6. In the **Message** box, paste the following JSON payload:
   ```json
   {
     "passphrase": "secret-token-change-me",
     "instrument": "{{ticker}}",
     "timeframe": "{{interval}}",
     "event_type": "LOW_SWEEP",
     "pattern_type": "double_bottom",
     "level_price": {{close}},
     "bar_low": {{low}},
     "bar_high": {{high}},
     "bar_close": {{close}},
     "atr": 4.5
   }
   ```
7. Click **Create**. When price sweeps a liquidity level and reclaims it, TradingView immediately sends the event to your Python service, which scores it using the model, logs the signal, and dispatches downstream alerts.

---

## 6. Research Disclaimer

This project is intended strictly for quantitative research, systematic trading model development, and educational purposes. Financial market trading involves substantial risk of loss. Historical pattern detection and machine learning probability estimates do not guarantee future profitability.
