"""
Storage and Persistence Module
Manages SQLite database storage for patterns, sweep events, ML predictions, and backtest results.
"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd


class DatabaseManager:
    """Production SQLite database interface for liquidity intelligence system."""

    def __init__(self, db_path: str = "data/liquidity_system.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def _init_db(self) -> None:
        """Creates necessary database tables if they do not exist."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Patterns table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS patterns (
                pattern_id TEXT PRIMARY KEY,
                pattern_type TEXT,
                instrument TEXT,
                direction TEXT,
                creation_timestamp TEXT,
                confirmation_timestamp TEXT,
                zone_low REAL,
                zone_high REAL,
                neckline_price REAL,
                equality_score REAL,
                pattern_height_atr REAL,
                touch_count INTEGER,
                is_active INTEGER DEFAULT 1
            )
            """)

            # Sweep Events table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS sweep_events (
                event_id TEXT PRIMARY KEY,
                pattern_id TEXT,
                instrument TEXT,
                sweep_timestamp TEXT,
                level_price REAL,
                sweep_depth_atr REAL,
                reclaimed INTEGER,
                reclaim_speed_bars INTEGER,
                label TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)

            # Predictions table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                instrument TEXT,
                pattern_type TEXT,
                event_id TEXT,
                hunt_probability REAL,
                confidence_tier TEXT,
                model_version TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)

            # Backtest Trades table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS backtest_trades (
                trade_id TEXT PRIMARY KEY,
                instrument TEXT,
                entry_time TEXT,
                exit_time TEXT,
                direction TEXT,
                entry_price REAL,
                exit_price REAL,
                position_size REAL,
                pnl_net REAL,
                exit_reason TEXT,
                hunt_probability REAL
            )
            """)

            conn.commit()

    def save_patterns(self, patterns_df: pd.DataFrame) -> None:
        """Stores detected patterns to database."""
        if patterns_df.empty:
            return
        with self.get_connection() as conn:
            patterns_df.to_sql("patterns", conn, if_exists="append", index=False)

    def save_events(self, events_df: pd.DataFrame) -> None:
        """Stores sweep events to database."""
        if events_df.empty:
            return
        with self.get_connection() as conn:
            events_df.to_sql("sweep_events", conn, if_exists="append", index=False)

    def save_prediction(self, pred: Dict[str, Any]) -> None:
        """Logs real-time / batch prediction."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO predictions (timestamp, instrument, pattern_type, event_id, hunt_probability, confidence_tier, model_version)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                pred.get("timestamp"),
                pred.get("instrument"),
                pred.get("pattern_type"),
                pred.get("event_id"),
                pred.get("hunt_probability"),
                pred.get("confidence_tier"),
                pred.get("model_version")
            ))
            conn.commit()

    def get_recent_predictions(self, limit: int = 100) -> pd.DataFrame:
        """Retrieves recent predictions."""
        with self.get_connection() as conn:
            return pd.read_sql(f"SELECT * FROM predictions ORDER BY prediction_id DESC LIMIT {limit}", conn)

    def get_all_patterns(self, instrument: Optional[str] = None) -> pd.DataFrame:
        """Retrieves patterns, optionally filtered by instrument."""
        query = "SELECT * FROM patterns"
        params = []
        if instrument:
            query += " WHERE instrument = ?"
            params.append(instrument)
        with self.get_connection() as conn:
            return pd.read_sql(query, conn, params=params)
