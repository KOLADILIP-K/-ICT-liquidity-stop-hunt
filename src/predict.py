"""
Prediction Engine Module
Loads trained model artifacts (XGBoost or Scikit-Learn fallback) and computes calibrated stop-hunt probabilities and confidence tiers.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import joblib
import numpy as np
import pandas as pd


class StopHuntPredictor:
    """Production predictor with artifact versioning, schema verification, and confidence tiers."""

    def __init__(self, models_dir: str = "models/"):
        self.models_dir = Path(models_dir)
        self.model = None
        self.calibrator = None
        self.features: List[str] = []
        self.metadata: Dict[str, Any] = {}
        self.is_loaded = False
        self.crypto_models: Dict[str, Any] = {}
        self.crypto_features: List[str] = [
            "distance_before_touch",
            "sigma_event",
            "approach_direction",
            "is_gap_through",
            "confluence_flag",
            "confluence_distance_atr",
            "touch_sequence",
            "hour_of_day",
            "day_of_week",
            "session_asia",
            "session_london",
            "session_overlap",
            "session_ny",
            "vol_low",
            "vol_high",
            "is_monthly_open"
        ]
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        model_xgb_path = self.models_dir / "xgboost_shared.json"
        model_joblib_path = self.models_dir / "model_shared.joblib"
        cal_path = self.models_dir / "calibrated_shared.joblib"
        schema_path = self.models_dir / "feature_schema.json"
        meta_path = self.models_dir / "training_metadata.json"

        if schema_path.exists():
            with open(schema_path, "r", encoding="utf-8") as f:
                self.features = json.load(f).get("features", [])

        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

        if cal_path.exists():
            try:
                self.calibrator = joblib.load(cal_path)
                self.is_loaded = True
            except Exception as e:
                print(f"Error loading calibrator: {e}")

        if model_joblib_path.exists():
            try:
                self.model = joblib.load(model_joblib_path)
                self.is_loaded = True
            except Exception:
                pass
        elif model_xgb_path.exists():
            try:
                from xgboost import XGBClassifier
                self.model = XGBClassifier()
                self.model.load_model(str(model_xgb_path))
                self.is_loaded = True
            except Exception:
                pass

        # Load integrated crypto-specific LightGBM models from submissions
        for inst in ["BTCUSD", "ETHUSD"]:
            crypto_lgb_path = self.models_dir / f"lightgbm_{inst}.joblib"
            if crypto_lgb_path.exists():
                try:
                    self.crypto_models[inst] = joblib.load(crypto_lgb_path)
                except Exception as e:
                    print(f"Notice: Could not load crypto model for {inst}: {e}")

    def predict_event(self, feature_row: Dict[str, Any]) -> Dict[str, Any]:
        """
        Computes probability and classification for a single pre-event feature dictionary.
        """
        if not self.is_loaded or (self.calibrator is None and self.model is None):
            return {
                "hunt_probability": None,
                "confidence_tier": "UNAVAILABLE",
                "model_status": "MODEL_NOT_LOADED",
                "model_version": "N/A"
            }

        X_vec = []
        for feat in self.features:
            val = feature_row.get(feat, 0.0)
            X_vec.append(float(val) if val is not None and not np.isnan(val) else 0.0)

        X = np.array([X_vec])

        if self.calibrator is not None:
            prob = float(self.calibrator.predict_proba(X)[0, 1])
        elif self.model is not None:
            prob = float(self.model.predict_proba(X)[0, 1])
        else:
            prob = 0.5

        if prob >= 0.75:
            tier = "HIGH"
        elif prob >= 0.55:
            tier = "MEDIUM"
        else:
            tier = "LOW"

        return {
            "hunt_probability": round(prob, 4),
            "confidence_tier": tier,
            "instrument": feature_row.get("instrument", "UNKNOWN"),
            "pattern_type": feature_row.get("pattern_type", "UNKNOWN"),
            "event_id": feature_row.get("event_id", "N/A"),
            "timestamp": str(feature_row.get("timestamp", "")),
            "model_version": self.metadata.get("model_version", "v1.0.0"),
            "model_status": "ACTIVE"
        }

    def predict_batch(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """
        Batch prediction for a dataframe of features.
        """
        if feature_df.empty or not self.is_loaded:
            return feature_df

        X = feature_df[self.features].fillna(0.0).values
        if self.calibrator is not None:
            probs = self.calibrator.predict_proba(X)[:, 1]
        elif self.model is not None:
            probs = self.model.predict_proba(X)[:, 1]
        else:
            probs = np.full(len(feature_df), 0.5)

        out_df = feature_df.copy()
        out_df["hunt_probability"] = np.round(probs, 4)
        out_df["confidence_tier"] = pd.cut(
            out_df["hunt_probability"],
            bins=[-0.1, 0.55, 0.75, 1.01],
            labels=["LOW", "MEDIUM", "HIGH"]
        )
        return out_df

    def predict_crypto_outcome(
        self,
        event_dict: Dict[str, Any],
        instrument: str = "BTCUSD"
    ) -> Dict[str, Any]:
        """
        Predicts 3-class structural outcome (0: REJECTION, 1: CONTINUATION, 2: SWEEP_RECLAIM)
        for Weekly/Monthly Open price interaction events using integrated LightGBM models.
        """
        inst_clean = "ETHUSD" if "ETH" in str(instrument).upper() else "BTCUSD"
        model = self.crypto_models.get(inst_clean)
        
        if model is None:
            return {
                "instrument": inst_clean,
                "status": "CRYPTO_MODEL_NOT_AVAILABLE",
                "predicted_outcome": "UNKNOWN",
                "probabilities": {}
            }

        vec = []
        for feat in self.crypto_features:
            val = event_dict.get(feat, 0.0)
            vec.append(float(val) if val is not None and not np.isnan(val) else 0.0)

        X = np.array([vec], dtype=np.float32)
        probs = model.predict_proba(X)[0]
        class_idx = int(np.argmax(probs))
        
        class_names = {
            0: "REJECTION",
            1: "CONTINUATION",
            2: "SWEEP_RECLAIM"
        }

        return {
            "instrument": inst_clean,
            "status": "SUCCESS",
            "model_type": "LightGBM_Crypto_Dedicated",
            "predicted_class_id": class_idx,
            "predicted_outcome": class_names.get(class_idx, "UNKNOWN"),
            "confidence": round(float(probs[class_idx]), 4),
            "probabilities": {
                "rejection": round(float(probs[0]), 4),
                "continuation": round(float(probs[1]), 4),
                "sweep_reclaim": round(float(probs[2]), 4)
            }
        }
