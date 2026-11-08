"""
Unit tests for BTCUSD and ETHUSD submission integration and crypto model inference.
"""

import pytest
from pathlib import Path
from src.predict import StopHuntPredictor


def test_crypto_models_loaded():
    predictor = StopHuntPredictor()
    assert predictor.is_loaded is True
    assert "BTCUSD" in predictor.crypto_models
    assert "ETHUSD" in predictor.crypto_models


def test_crypto_inference_btc():
    predictor = StopHuntPredictor()
    dummy_event = {
        "distance_before_touch": 0.5,
        "sigma_event": 65.0,
        "approach_direction": 1.0,
        "is_gap_through": 0.0,
        "confluence_flag": 1.0,
        "confluence_distance_atr": 0.15,
        "touch_sequence": 1,
        "hour_of_day": 10.0,
        "day_of_week": 1.0,
        "session_asia": 0.0,
        "session_london": 1.0,
        "session_overlap": 0.0,
        "session_ny": 0.0,
        "vol_low": 0.0,
        "vol_high": 1.0,
        "is_monthly_open": 0.0,
    }
    res = predictor.predict_crypto_outcome(dummy_event, instrument="BTCUSD")
    assert res["status"] == "SUCCESS"
    assert res["predicted_outcome"] in ["REJECTION", "CONTINUATION", "SWEEP_RECLAIM"]
    assert "probabilities" in res
    assert 0.0 <= res["confidence"] <= 1.0


def test_crypto_inference_eth():
    predictor = StopHuntPredictor()
    dummy_event = {
        "distance_before_touch": 0.3,
        "sigma_event": 4.5,
        "approach_direction": -1.0,
        "is_gap_through": 0.0,
        "confluence_flag": 0.0,
        "confluence_distance_atr": 0.0,
        "touch_sequence": 2,
        "hour_of_day": 15.0,
        "day_of_week": 3.0,
        "session_asia": 0.0,
        "session_london": 0.0,
        "session_overlap": 1.0,
        "session_ny": 0.0,
        "vol_low": 1.0,
        "vol_high": 0.0,
        "is_monthly_open": 1.0,
    }
    res = predictor.predict_crypto_outcome(dummy_event, instrument="ETHUSD")
    assert res["status"] == "SUCCESS"
    assert res["predicted_outcome"] in ["REJECTION", "CONTINUATION", "SWEEP_RECLAIM"]
    assert "probabilities" in res
    assert 0.0 <= res["confidence"] <= 1.0
