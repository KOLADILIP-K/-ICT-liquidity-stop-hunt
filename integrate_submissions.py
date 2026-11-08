import os
import shutil
from pathlib import Path
import json
import pandas as pd

def integrate_submission_artifacts():
    base_dir = Path("D:/KOLADILIP K/Internship")
    slhl_dir = base_dir / "slhl"
    models_dir = slhl_dir / "models"
    processed_dir = slhl_dir / "data" / "processed"
    
    models_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    btc_sub = base_dir / "BTCUSD_Submission" / "BTCUSD_Submission"
    eth_sub = base_dir / "ETHUSD_Submission" / "ETHUSD_Submission"
    
    # 1. Copy crypto-specific ML models and reports
    crypto_models = {
        "BTCUSD": btc_sub / "13_Machine_Learning",
        "ETHUSD": eth_sub / "13_Machine_Learning"
    }
    
    for inst, ml_dir in crypto_models.items():
        if ml_dir.exists():
            for m_name in ["lightgbm.joblib", "random_forest.joblib", "logistic_regression.joblib"]:
                src = ml_dir / m_name
                dst = models_dir / f"{m_name.replace('.joblib', f'_{inst}.joblib')}"
                if src.exists():
                    shutil.copy2(src, dst)
                    print(f"Copied {src.name} -> {dst.name}")
            
            # Copy report & feature importance
            report_src = ml_dir / "ml_report.json"
            if report_src.exists():
                shutil.copy2(report_src, models_dir / f"crypto_ml_report_{inst}.json")
            
            fi_src = ml_dir / "ml_feature_importance.csv"
            if fi_src.exists():
                shutil.copy2(fi_src, processed_dir / f"feature_importance_{inst}_lgb.csv")
                
            metrics_src = ml_dir / "ml_metrics.csv"
            if metrics_src.exists():
                shutil.copy2(metrics_src, processed_dir / f"crypto_metrics_{inst}.csv")

    # 2. Copy canonical event detection files
    for inst, sub in [("BTCUSD", btc_sub), ("ETHUSD", eth_sub)]:
        ev_dir = sub / "04_Event_Detection"
        if ev_dir.exists():
            for f in ["events.csv", "events.parquet", "period_touch_summary.csv", "h1_atr.csv"]:
                src = ev_dir / f
                if src.exists():
                    dst_name = f.replace(".", f"_{inst}.")
                    shutil.copy2(src, processed_dir / dst_name)
                    print(f"Copied {src.name} -> {dst_name}")

    # 3. Copy Weekly and Monthly Open level files
    for inst, sub in [("BTCUSD", btc_sub), ("ETHUSD", eth_sub)]:
        lvl_dir = sub / "03_WO_MO_Levels"
        if lvl_dir.exists():
            for f in ["weekly_open.csv", "monthly_open.csv", "weekly_open.parquet", "monthly_open.parquet"]:
                src = lvl_dir / f
                if src.exists():
                    dst_name = f.replace(".", f"_{inst}.")
                    shutil.copy2(src, processed_dir / dst_name)
                    print(f"Copied {src.name} -> {dst_name}")

    # 4. Copy uploaded Dukascopy XAUUSD and XAGUSD event files from user upload
    u1 = Path("C:/Users/kolab/.gemini/antigravity/brain/fe6271f7-4d79-4eb5-a337-8a59f6a39980/.user_uploaded/media_1794128505577_35ae0ff4.csv")
    u2 = Path("C:/Users/kolab/.gemini/antigravity/brain/fe6271f7-4d79-4eb5-a337-8a59f6a39980/.user_uploaded/media_1794128505586_de2449cf.csv")
    if u1.exists():
        shutil.copy2(u1, processed_dir / "events_XAGUSD_dukascopy.csv")
        print("Copied user XAGUSD event file to processed_dir.")
    if u2.exists():
        shutil.copy2(u2, processed_dir / "events_XAUUSD_dukascopy.csv")
        print("Copied user XAUUSD event file to processed_dir.")

    print("\nArtifact integration complete!")

if __name__ == "__main__":
    integrate_submission_artifacts()
