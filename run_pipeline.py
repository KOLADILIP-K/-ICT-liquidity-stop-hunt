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

    instruments = ["XAUUSD", "XAGUSD", "BTCUSD", "ETHUSD"]
    all_feature_dfs = []
    all_patterns = []
    all_events = []
    enriched_dfs = {}
    checklist_summaries = {}

    from src.liquidity_points import LiquidityPointsEngine
    lp_engine = LiquidityPointsEngine()

    SAMPLE_BARS = 25000

    for inst in instruments:
        print(f"\nProcessing {inst}...")
        df_raw, report = dl.load_instrument(inst)
        if len(df_raw) > SAMPLE_BARS:
            print(f"Selecting latest {SAMPLE_BARS} candles for focused research evaluation.")
            df_raw = df_raw.iloc[-SAMPLE_BARS:].reset_index(drop=True)

        print(f"- Adding core technical indicators (ATR, anatomy, returns, PDH/PDL, PWH/PWL, PMH/PML)...")
        df_ind = add_core_indicators(df_raw, atr_period=14)

        print(f"- Computing Option B session high/low levels...")
        df_sess, sess_levels = session_mgr.compute_session_levels(df_ind)
        df_sess["instrument"] = inst
        enriched_dfs[inst] = df_sess

        print(f"- Detecting confirmed swing highs and lows (k=3)...")
        swings = swing_detector.detect_swings(df_sess)
        print(f"  Found {len(swings)} confirmed swings.")

        print(f"- Detecting Double Bottoms, Triple Bottoms, and Trendlines...")
        patterns = pattern_detector.detect_all_patterns(df_sess, swings, instrument=inst)
        print(f"  Found {len(patterns)} valid structural liquidity patterns.")
        all_patterns.extend(patterns)

        print(f"- Evaluating sweeps, reclaims, HTF liquidity points, and follow-through labels...")
        events = labeling_engine.generate_events_and_labels(df_sess, patterns)
        print(f"  Generated {len(events)} candidate sweep events.")
        all_events.extend(events)

        print(f"- Engineering pre-event features without leakage...")
        feat_df = feature_engineer.create_feature_dataset(df_sess, patterns, events)
        all_feature_dfs.append(feat_df)
        feat_df.to_csv(proc_dir / f"features_{inst}.csv", index=False)

        # Run Section 6 checklist on the latest bar for terminal inspection
        latest_chk = lp_engine.evaluate_bar_checklist(len(df_sess) - 1, df_sess)
        checklist_summaries[inst] = {
            "timestamp": str(latest_chk.timestamp),
            "current_price": float(latest_chk.current_price),
            "atr": float(latest_chk.atr),
            "levels_up_to_date": bool(latest_chk.levels_up_to_date),
            "all_levels": {k: float(v) for k, v in latest_chk.all_levels.items() if not np.isnan(v)},
            "nearest_above": {"level": latest_chk.nearest_above[0], "price": float(latest_chk.nearest_above[1]), "dist_atr": float(latest_chk.nearest_above[2])} if latest_chk.nearest_above else None,
            "nearest_below": {"level": latest_chk.nearest_below[0], "price": float(latest_chk.nearest_below[1]), "dist_atr": float(latest_chk.nearest_below[2])} if latest_chk.nearest_below else None,
            "active_clusters": [{"center": float(c.center_price), "levels": c.levels, "combined_weight": float(c.combined_weight), "type": c.zone_type} for c in latest_chk.confluence_clusters],
            "active_behavior": str(latest_chk.active_behavior),
            "interacting_level": str(latest_chk.interacting_level),
            "htf_bias": str(latest_chk.htf_bias),
            "behavior_fits_bias": bool(latest_chk.behavior_fits_bias)
        }

    # Save checklist summary
    with open(proc_dir / "liquidity_checklist_summary.json", "w") as f:
        json.dump(checklist_summaries, f, indent=2)

    # Combine datasets
    combined_features = pd.concat(all_feature_dfs, ignore_index=True)
    combined_features.to_csv(proc_dir / "features_combined.csv", index=False)
    print(f"\nTotal combined feature rows: {len(combined_features)}")

    # Class balance summary
    label_counts = combined_features["label"].value_counts().to_dict()
    print(f"Label distribution: {label_counts}")

    # 2. Train shared XGBoost model on 80% chronological train data
    print("\nTraining Shared XGBoost Classifier (80% Chronological Training Data)...")
    xgb_model, calibrator, metadata = trainer.train_shared_model(combined_features)
    print("Test Set Performance Summary:")
    print(json.dumps(metadata.get("test_metrics", {}), indent=2))

    # 3. Cross-instrument transfer experiments
    print("\nRunning Cross-Instrument Transfer Experiments across Metals and Crypto...")
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

    # 5. Run Backtests across all 4 instruments
    backtest_summary = {}
    trade_list = []

    for inst in instruments:
        print(f"\nRunning Backtest on {inst}...")
        inst_features = pred_features[pred_features["instrument"] == inst]
        trades, equity, metrics = backtester.run_backtest(
            df=enriched_dfs[inst],
            signals_df=inst_features,
            probability_threshold=0.30
        )
        print(f"{inst} Backtest: {metrics.get('total_trades')} trades, Net Profit: ${metrics.get('net_profit')}, Win Rate: {metrics.get('win_rate')}%, Max DD: {metrics.get('max_drawdown_pct')}%")
        backtest_summary[inst] = metrics
        if not trades.empty:
            trade_list.append(trades)
        if not equity.empty:
            equity.to_csv(proc_dir / f"backtest_equity_{inst}.csv", index=False)

    all_trades = pd.concat(trade_list, ignore_index=True) if trade_list else pd.DataFrame()
    all_trades.to_csv(proc_dir / "backtest_trades.csv", index=False)

    with open(proc_dir / "backtest_summary.json", "w") as f:
        json.dump(backtest_summary, f, indent=2)

    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION COMPLETE - ALL ARTIFACTS AND DATASETS SAVED!")
    print("=" * 80)


if __name__ == "__main__":
    run_full_pipeline()
