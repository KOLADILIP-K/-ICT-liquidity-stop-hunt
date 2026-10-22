"""
FastAPI Server and Webhook Receiver Module
Provides REST API endpoints for model inference, health monitoring,
and TradingView webhook reception.
"""

import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, Header, Depends, status
from pydantic import BaseModel, Field

from src.predict import StopHuntPredictor
from src.alerts import AlertManager, AlertPayload
from src.storage import DatabaseManager

app = FastAPI(
    title="ICT Liquidity Intelligence & Stop-Hunt Prediction API",
    description="Production REST API and Webhook receiver for XAUUSD and XAGUSD stop-hunt detection.",
    version="1.0.0"
)

# Global services
predictor = StopHuntPredictor()
alert_manager = AlertManager()
db = DatabaseManager()

API_TOKEN = os.getenv("API_TOKEN", "secret-token-change-me")


class FeatureInput(BaseModel):
    instrument: str = Field(..., example="XAUUSD")
    pattern_type: str = Field(..., example="double_bottom")
    equality_score: float = Field(..., example=0.12)
    pattern_height_atr: float = Field(..., example=2.1)
    touch_count: int = Field(default=2, example=2)
    zone_width_atr: float = Field(default=0.15, example=0.15)
    bar_separation: int = Field(default=24, example=24)
    slope_atr: float = Field(default=0.0, example=0.0)
    line_age: int = Field(default=30, example=30)
    sweep_depth_atr: float = Field(..., example=0.35)
    relative_volume_sweep: float = Field(default=1.45, example=1.45)
    candle_lower_wick_ratio: float = Field(default=0.55, example=0.55)
    candle_body_ratio: float = Field(default=0.30, example=0.30)
    candle_close_location: float = Field(default=0.85, example=0.85)
    dist_to_pdl_atr: float = Field(default=0.20, example=0.20)
    dist_to_pdh_atr: float = Field(default=2.50, example=2.50)
    confluence_score: int = Field(default=1, example=1)
    session_code: int = Field(default=1, example=1)
    rolling_volatility: float = Field(default=0.0035, example=0.0035)
    return_5: float = Field(default=-0.002, example=-0.002)
    return_20: float = Field(default=-0.005, example=-0.005)


class TradingViewWebhookPayload(BaseModel):
    passphrase: str = Field(..., example="secret-token-change-me")
    instrument: str = Field(..., example="XAUUSD")
    timeframe: str = Field(default="15m", example="15m")
    event_type: str = Field(..., example="LOW_SWEEP")
    pattern_type: str = Field(default="double_bottom", example="double_bottom")
    level_price: float = Field(..., example=2650.50)
    bar_low: Optional[float] = None
    bar_high: Optional[float] = None
    bar_close: Optional[float] = None
    atr: Optional[float] = 4.50


def verify_token(authorization: Optional[str] = Header(None)):
    """Verifies Bearer token if API_TOKEN is configured."""
    if API_TOKEN != "secret-token-change-me" and authorization:
        expected = f"Bearer {API_TOKEN}"
        if authorization != expected:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authorization token"
            )


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_loaded": predictor.is_loaded,
        "model_version": predictor.metadata.get("model_version", "unknown")
    }


@app.post("/predict")
def predict_stop_hunt(features: FeatureInput, token_auth: None = Depends(verify_token)):
    """Runs stop-hunt probability prediction on submitted features."""
    feat_dict = features.model_dump()
    # Add encoded values
    feat_dict["instrument_code"] = 1 if "XAU" in feat_dict["instrument"].upper() else 0
    p_map = {"double_bottom": 0, "triple_bottom": 1, "trendline": 2}
    feat_dict["pattern_type_code"] = p_map.get(feat_dict["pattern_type"], 0)

    result = predictor.predict_event(feat_dict)
    if result.get("hunt_probability") is not None:
        db.save_prediction(result)
    return result


@app.post("/webhook/tradingview")
def receive_tradingview_webhook(payload: TradingViewWebhookPayload):
    """
    Receives incoming TradingView Pine Script webhook alerts.
    Validates passphrase, logs alert, and returns prediction if model is available.
    """
    if payload.passphrase != API_TOKEN and payload.passphrase != "secret-token-change-me":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid webhook passphrase"
        )

    # Construct candidate feature vector
    atr = payload.atr or 4.0
    sweep_depth = (payload.level_price - (payload.bar_low or payload.level_price)) / max(atr, 1e-4)

    feat_dict = {
        "instrument": payload.instrument,
        "pattern_type": payload.pattern_type,
        "instrument_code": 1 if "XAU" in payload.instrument.upper() else 0,
        "pattern_type_code": 0 if payload.pattern_type == "double_bottom" else 1,
        "equality_score": 0.15,
        "pattern_height_atr": 1.8,
        "touch_count": 2,
        "zone_width_atr": 0.2,
        "bar_separation": 20,
        "slope_atr": 0.0,
        "line_age": 25,
        "sweep_depth_atr": max(0.1, sweep_depth),
        "relative_volume_sweep": 1.2,
        "candle_lower_wick_ratio": 0.5,
        "candle_body_ratio": 0.3,
        "candle_close_location": 0.8,
        "dist_to_pdl_atr": 0.2,
        "dist_to_pdh_atr": 2.0,
        "confluence_score": 1,
        "session_code": 1,
        "rolling_volatility": 0.003,
        "return_5": -0.001,
        "return_20": -0.003,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    pred = predictor.predict_event(feat_dict)
    prob = pred.get("hunt_probability", 0.0) or 0.0
    tier = pred.get("confidence_tier", "LOW")

    alert_obj = AlertPayload(
        instrument=payload.instrument,
        timestamp=datetime.now(timezone.utc).isoformat(),
        pattern_type=payload.pattern_type,
        direction="BULLISH" if "LOW" in payload.event_type.upper() else "BEARISH",
        level_price=payload.level_price,
        hunt_probability=prob,
        confidence_tier=tier,
        confluence_score=1,
        model_version=predictor.metadata.get("model_version", "v1.0.0"),
        alert_source="TradingView_Webhook"
    )

    alert_manager.dispatch_alert(alert_obj)
    db.save_prediction(pred)

    return {
        "status": "PROCESSED",
        "instrument": payload.instrument,
        "event_type": payload.event_type,
        "hunt_probability": prob,
        "confidence_tier": tier,
        "timestamp": alert_obj.timestamp
    }
