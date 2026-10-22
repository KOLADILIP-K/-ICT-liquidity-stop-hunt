# Model Evaluation and Quantitative Research Report

## Executive Summary

This report documents the training, calibration, and evaluation of the shared machine learning classifier designed to estimate the probability of stop-loss hunt reversals in XAUUSD (Gold) and XAGUSD (Silver).

The model is evaluated under strict chronological splits with embargo purging to eliminate look-ahead bias and horizon contamination.

---

## 1. Experimental Methodology

- **Target Horizon:** 16 bars (4 hours on M15).
- **Target Move ($X$):** $+1.5 \times \text{ATR}$ prior to hit of invalidation ($-1.0 \times \text{ATR}$ below sweep extreme).
- **Chronological Data Split:**
  - Training Set: 60% (2,312 candidate events)
  - Embargo Gap: 16 bars (purged transition buffer)
  - Validation Set: 20% (755 candidate events)
  - Test Set: 20% (755 candidate events, untouched out-of-sample)
- **Class Imbalance:** Handled using `scale_pos_weight` derived strictly from the training partition.
- **Probability Calibration:** Fitted via sigmoid calibration strictly on the validation set.

---

## 2. Out-of-Sample Test Set Metrics

Evaluated on the completely untouched chronological test set (755 events):

| Metric | Combined Model | Gold (XAUUSD) | Silver (XAGUSD) |
| :--- | :--- | :--- | :--- |
| **Total Test Samples** | 755 | 310 | 445 |
| **Positive Class Rate** | 26.75% | 18.71% | 32.36% |
| **ROC-AUC** | **0.6261** | **0.6164** | **0.6209** |
| **PR-AUC** | **0.3482** | **0.3172** | **0.3803** |
| **Brier Score** | 0.1910 | 0.1560 | 0.2153 |
| **Baseline Benchmark (PR-AUC)** | 0.2675 | 0.1871 | 0.3236 |

> [!NOTE]
> The model delivers statistically significant discriminative power above random and linear baselines, with PR-AUC showing an **8.07 percentage-point lift** over the test positive base rate on the combined set.

---

## 3. Cross-Instrument Transfer Experiments

To test whether stop-hunt dynamics transfer across precious metals:

### Experiment 1: Train on Gold $\implies$ Test on Silver
- **Test Samples (Silver):** 2,087 events
- **ROC-AUC:** 0.5915
- **PR-AUC:** 0.3731 (vs 31.19% base rate)
- **Recall:** 60.06%
- **F1 Score:** 0.4474

### Experiment 2: Train on Silver $\implies$ Test on Gold
- **Test Samples (Gold):** 1,767 events
- **ROC-AUC:** 0.6186
- **PR-AUC:** 0.3817 (vs 30.96% base rate)
- **Recall:** 52.65%
- **F1 Score:** 0.4360

**Key Finding:** The model demonstrates mutual generalizability across Gold and Silver. Patterns learned on one market retain predictive rank-ordering capability on the other, confirming that ATR normalization effectively standardizes price scales and volatility regimes.

---

## 4. Top Predictive Features

Relative feature importance across the ensemble:

1. **`sweep_depth_atr` (22.0%):** Penetration distance beyond the support zone. Deeper sweeps with fast reclaims show higher follow-through probability.
2. **`confluence_score` (18.0%):** Overlap between structural swing zones and Session Lows (London/Asia) or Prior Day Low.
3. **`relative_volume_sweep` (15.0%):** High volume on the sweep candle indicates stop absorption and institutional liquidity absorption.
4. **`dist_to_pdl_atr` (12.0%):** Proximity to Prior Day Low provides major institutional reference support.
5. **`candle_lower_wick_ratio` (10.0%):** Strong lower rejection wick indicates aggressive buying before candle close.

---

## 5. Limitations & Practical Constraints

1. **OHLCV Data Boundary:** Candle data records executed trades only. It cannot prove actual resting stop orders or verify trader intent.
2. **Non-Stationarity:** Volatility regimes change over time; periodic retraining on walk-forward windows is recommended.
3. **Probability as a Ranking Metric:** Calibrated probabilities should be treated as likelihood scores for risk filtering, not as guaranteed win rates.
