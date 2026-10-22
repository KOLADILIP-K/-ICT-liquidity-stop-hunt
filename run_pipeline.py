"""
End-to-End Pipeline Execution Script
Runs data ingestion, indicator calculations, pattern detection, labeling,
feature engineering, XGBoost training, cross-instrument evaluation, and backtesting.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd

from src.data_loader import DataLoader
from src.indicators import add_core_indicators
from src.swing_detection import SwingDetector
from src.liquidity_patterns import LiquidityPatternDetector
from src.session_liquidity import SessionLiquidityManager
from src.labeling import LabelingEngine
from src.feature_engineering import FeatureEngineer
from src.train import ModelTrainer
from src.backtest import BacktestEngine
from src.storage import DatabaseManager


def run_full_pipeline():
    print("=" * 80)
    print("RUNNING ICT LIQUIDITY INTELLIGENCE & XGBOOST STOP-HUNT PIPELINE")
    print("=" * 80)

    # 1. Initialize modules
    dl = DataLoader()
    db = DatabaseManager()
    session_mgr = SessionLiquidityManager()
    swing_detector = SwingDetector(k=3)
    pattern_detector = LiquidityPatternDetector(
        tol_atr=0.25,
        min_separation=5,
        max_separation_db=60,
        max_duration_tb=120,
        min_neckline_atr=1.5
    )
    labeling_engine = LabelingEngine(
        sweep_threshold_atr=0.1,
        max_reclaim_bars=3,
        target_atr_x=1.5,
        invalidation_atr=1.0,
        horizon_bars_m=16
    )
    feature_engineer = FeatureEngineer(session_manager=session_mgr)
    trainer = ModelTrainer()
    backtester = BacktestEngine(
        initial_capital=100000.0,
        risk_per_trade_pct=1.0,
        target_atr_x=1.5,
        max_holding_bars=32
    )

    proc_dir = Path("data/processed")
    proc_dir.mkdir(parents=True, exist_ok=True)

    instruments = ["XAUUSD", "XAGUSD"]
    all_feature_dfs = []
    all_patterns = []
    all_events = []
    enriched_dfs = {}

    # Sample for pipeline if large: we can use a representative slice (e.g. 15,000 bars per instrument)
    # for super fast, high-quality development, or all bars. Let's use 20,000 bars per instrument (~1.5 years of 15m)
    # which gives hundreds of clean patterns and trains in seconds!
    SAMPLE_BARS = 20000

    for inst in instruments:
        print(f"\nProcessing {inst}...")
        df_raw, report = dl.load_instrument(inst)
        if len(df_raw) > SAMPLE_BARS:
            print(f"Selecting latest {SAMPLE_BARS} candles for focused research evaluation.")
            df_raw = df_raw.iloc[-SAMPLE_BARS:].reset_index(drop=True)

        print(f"- Adding core technical indicators (ATR, anatomy, returns, PDH/PDL)...")
        df_ind = add_core_indicators(df_raw, atr_period=14)

        print(f"- Computing Option B session high/low levels...")
        df_sess, sess_levels = session_mgr.compute_session_levels(df_ind)
        enriched_dfs[inst] = df_sess

        print(f"- Detecting confirmed swing highs and lows (k=3)...")
        swings = swing_detector.detect_swings(df_sess)
        print(f"  Found {len(swings)} confirmed swings.")

        print(f"- Detecting Double Bottoms, Triple Bottoms, and Trendlines...")
        patterns = pattern_detector.detect_all_patterns(df_sess, swings, instrument=inst)
        print(f"  Found {len(patterns)} valid structural liquidity patterns.")
        all_patterns.extend(patterns)

        print(f"- Evaluating sweeps, reclaims, and target follow-through labels...")
        events = labeling_engine.generate_events_and_labels(df_sess, patterns)
        print(f"  Generated {len(events)} candidate sweep events.")
        all_events.extend(events)

        print(f"- Engineering pre-event features without leakage...")
        feat_df = feature_engineer.create_feature_dataset(df_sess, patterns, events)
        all_feature_dfs.append(feat_df)
        feat_df.to_csv(proc_dir / f"features_{inst}.csv", index=False)

    # Combine datasets
    combined_features = pd.concat(all_feature_dfs, ignore_index=True)
    combined_features.to_csv(proc_dir / "features_combined.csv", index=False)
    print(f"\nTotal combined feature rows: {len(combined_features)}")

    # Class balance summary
    label_counts = combined_features["label"].value_counts().to_dict()
    print(f"Label distribution: {label_counts}")

    # 2. Train shared XGBoost model
    print("\nTraining Shared XGBoost Classifier...")
    xgb_model, calibrator, metadata = trainer.train_shared_model(combined_features)
    print("Test Set Performance Summary:")
    print(json.dumps(metadata.get("test_metrics", {}), indent=2))

    # 3. Cross-instrument transfer experiments
    print("\nRunning Cross-Instrument Transfer Experiments...")
    cross_metrics = trainer.run_cross_instrument_experiments(combined_features)
    print(json.dumps(cross_metrics, indent=2))
    with open(proc_dir / "cross_instrument_results.json", "w") as f:
        json.dump(cross_metrics, f, indent=2)

    # 4. Predict probabilities on feature dataset
    print("\nGenerating model predictions for backtesting...")
    from src.predict import StopHuntPredictor
    predictor = StopHuntPredictor()
    pred_features = predictor.predict_batch(combined_features)
    pred_features.to_csv(proc_dir / "predictions_combined.csv", index=False)

    # 5. Run Backtests
    print("\nRunning Backtest on XAUUSD...")
    gold_features = pred_features[pred_features["instrument"] == "XAUUSD"]
    trades_gold, equity_gold, metrics_gold = backtester.run_backtest(
        df=enriched_dfs["XAUUSD"],
        signals_df=gold_features,
        probability_threshold=0.30
    )
    print(f"Gold Backtest: {metrics_gold.get('total_trades')} trades, Net Profit: ${metrics_gold.get('net_profit')}, Win Rate: {metrics_gold.get('win_rate')}%")

    print("\nRunning Backtest on XAGUSD...")
    silver_features = pred_features[pred_features["instrument"] == "XAGUSD"]
    trades_silver, equity_silver, metrics_silver = backtester.run_backtest(
        df=enriched_dfs["XAGUSD"],
        signals_df=silver_features,
        probability_threshold=0.30
    )
    print(f"Silver Backtest: {metrics_silver.get('total_trades')} trades, Net Profit: ${metrics_silver.get('net_profit')}, Win Rate: {metrics_silver.get('win_rate')}%")

    # Combine backtest records
    all_trades = pd.concat([trades_gold, trades_silver], ignore_index=True) if not trades_gold.empty and not trades_silver.empty else (trades_gold if not trades_gold.empty else trades_silver)
    all_trades.to_csv(proc_dir / "backtest_trades.csv", index=False)
    equity_gold.to_csv(proc_dir / "backtest_equity_XAUUSD.csv", index=False)
    equity_silver.to_csv(proc_dir / "backtest_equity_XAGUSD.csv", index=False)

    backtest_summary = {
        "XAUUSD": metrics_gold,
        "XAGUSD": metrics_silver
    }
    with open(proc_dir / "backtest_summary.json", "w") as f:
        json.dump(backtest_summary, f, indent=2)

    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION COMPLETE - ALL ARTIFACTS AND DATASETS SAVED!")
    print("=" * 80)


if __name__ == "__main__":
    run_full_pipeline()
