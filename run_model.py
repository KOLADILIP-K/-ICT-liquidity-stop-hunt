"""
Model Inference Runner
Executes trained stop-hunt model on the latest market data for XAUUSD and XAGUSD,
printing the latest detected liquidity zones, sweep events, and calibrated probabilities.
"""

import sys
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

from src.data_loader import DataLoader
from src.indicators import add_core_indicators
from src.swing_detection import SwingDetector
from src.liquidity_patterns import LiquidityPatternDetector
from src.session_liquidity import SessionLiquidityManager
from src.labeling import LabelingEngine
from src.feature_engineering import FeatureEngineer
from src.predict import StopHuntPredictor


def run_live_inference():
    print("=" * 80)
    print("EXECUTING STOP-HUNT INFERENCE ENGINE (XAUUSD & XAGUSD)")
    print("=" * 80)

    predictor = StopHuntPredictor()
    if not predictor.is_loaded:
        print("[ERROR] Model artifacts not found in models/. Running pipeline first...")
        import subprocess
        subprocess.run([sys.executable, "run_pipeline.py"], check=True)
        predictor = StopHuntPredictor()

    print(f"Model Status:  {predictor.metadata.get('model_type', 'Boosted Trees')}")
    print(f"Model Version: {predictor.metadata.get('model_version', 'v1.0.0')}")
    print(f"Engine:        {predictor.metadata.get('engine', 'scikit-learn / xgboost')}")
    print(f"Trained On:    {predictor.metadata.get('train_samples', 'N/A')} samples")
    print("-" * 80)

    dl = DataLoader()
    session_mgr = SessionLiquidityManager()
    swing_detector = SwingDetector(k=3)
    pattern_detector = LiquidityPatternDetector()
    labeling_engine = LabelingEngine()
    feature_engineer = FeatureEngineer(session_manager=session_mgr)

    instruments = ["XAUUSD", "XAGUSD"]

    for inst in instruments:
        print(f"\nScanning {inst} (Latest Candles)...")
        df_raw, report = dl.load_instrument(inst)
        
        # Take latest 500 candles (~5 days of 15m)
        df_recent = df_raw.iloc[-500:].reset_index(drop=True)
        latest_candle = df_recent.iloc[-1]
        latest_time = latest_candle["timestamp"]
        latest_close = latest_candle["close"]

        print(f"  Latest Candle Timestamp: {latest_time}")
        print(f"  Latest Closing Price:    {latest_close:.3f}")

        # Compute indicators
        df_ind = add_core_indicators(df_recent, atr_period=14)
        df_sess, _ = session_mgr.compute_session_levels(df_ind)
        latest_atr = df_ind["atr"].iloc[-1]
        print(f"  Current ATR(14):         {latest_atr:.3f}")

        # Prior Day & Session Levels
        pdh = df_ind["pdh"].iloc[-1]
        pdl = df_ind["pdl"].iloc[-1]
        print(f"  Prior Day High (PDH):    {pdh:.3f}")
        print(f"  Prior Day Low  (PDL):    {pdl:.3f}")

        # Swings and patterns
        swings = swing_detector.detect_swings(df_sess)
        patterns = pattern_detector.detect_all_patterns(df_sess, swings, instrument=inst)
        print(f"  Active Patterns Found:   {len(patterns)}")

        # Candidate sweeps
        events = labeling_engine.generate_events_and_labels(df_sess, patterns)
        print(f"  Candidate Sweeps Found:  {len(events)}")

        if events:
            feat_df = feature_engineer.create_feature_dataset(df_sess, patterns, events)
            pred_df = predictor.predict_batch(feat_df)

            print(f"\n  === Latest 5 Stop-Hunt Scored Events for {inst} ===")
            print("-" * 75)
            print(f"{'Timestamp':<24} | {'Pattern':<14} | {'Level':<9} | {'Prob %':<7} | {'Tier':<6} | {'Outcome'}")
            print("-" * 75)
            
            recent_preds = pred_df.tail(5)
            for _, r in recent_preds.iterrows():
                ts_str = str(r['timestamp'])[:19]
                pat_str = str(r['pattern_type'])[:13]
                lvl_str = f"{r['level_price']:.2f}"
                prob_str = f"{r['hunt_probability']*100:.1f}%"
                tier_str = str(r.get('confidence_tier', 'LOW'))
                lbl_str = str(r.get('label', 'Pending'))
                
                # Highlight high probability
                flag = "[*] " if r['hunt_probability'] >= 0.30 else "    "
                print(f"{flag}{ts_str:<20} | {pat_str:<14} | {lvl_str:<9} | {prob_str:<7} | {tier_str:<6} | {lbl_str}")
        else:
            print("  No active sweep events detected in the latest window.")

    print("\n" + "=" * 80)
    print("INFERENCE RUN COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_live_inference()
