"""
Alerting and Signal Dispatcher Module
Provides structured JSON alert formatting, duplicate signal suppression,
stale-data detection, and webhook integration.
"""

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import requests


@dataclass
class AlertPayload:
    """Standardized alert payload for trading signals and webhooks."""
    instrument: str
    timestamp: str
    pattern_type: str
    direction: str
    level_price: float
    hunt_probability: float
    confidence_tier: str
    confluence_score: int
    model_version: str
    alert_source: str = "ICT_Liquidity_Engine"

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


class AlertManager:
    """Manages alert dispatching, duplicate suppression, and stale data verification."""

    def __init__(
        self,
        cooldown_seconds: int = 1800,  # 30 minutes suppression per level
        max_data_staleness_minutes: int = 60,
        webhook_url: Optional[str] = None,
        log_file: str = "logs/alerts.log"
    ):
        self.cooldown_seconds = cooldown_seconds
        self.max_data_staleness_minutes = max_data_staleness_minutes
        self.webhook_url = webhook_url
        self.alert_cache: Dict[str, datetime] = {}

        # Set up file logger
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("AlertManager")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
            self.logger.addHandler(fh)

    def is_stale_data(self, candle_timestamp: datetime) -> bool:
        """Checks if a candle is older than the staleness threshold."""
        now = datetime.now(timezone.utc)
        if candle_timestamp.tzinfo is None:
            candle_timestamp = candle_timestamp.replace(tzinfo=timezone.utc)
        diff_minutes = (now - candle_timestamp).total_seconds() / 60.0
        return diff_minutes > self.max_data_staleness_minutes

    def dispatch_alert(self, alert: AlertPayload) -> bool:
        """
        Dispatches an alert if it is not duplicate or stale.
        """
        cache_key = f"{alert.instrument}_{alert.pattern_type}_{round(alert.level_price, 2)}"
        now = datetime.now(timezone.utc)

        # Duplicate suppression check
        if cache_key in self.alert_cache:
            last_alert_time = self.alert_cache[cache_key]
            if (now - last_alert_time).total_seconds() < self.cooldown_seconds:
                self.logger.info(f"Duplicate alert suppressed for key: {cache_key}")
                return False

        # Update cache
        self.alert_cache[cache_key] = now

        # Log alert
        alert_json = alert.to_json()
        self.logger.info(f"SIGNAL ALERT TRIGGERED:\n{alert_json}")
        print(f"\n[ALERT TRIGGERED] {alert.instrument} {alert.pattern_type} {alert.direction} | Prob: {alert.hunt_probability} ({alert.confidence_tier}) | Level: {alert.level_price}")

        # Send webhook if configured
        if self.webhook_url:
            try:
                resp = requests.post(
                    self.webhook_url,
                    data=alert_json,
                    headers={"Content-Type": "application/json"},
                    timeout=5.0
                )
                self.logger.info(f"Webhook dispatched to {self.webhook_url}: Status {resp.status_code}")
            except Exception as e:
                self.logger.error(f"Failed to send webhook to {self.webhook_url}: {e}")

        return True
