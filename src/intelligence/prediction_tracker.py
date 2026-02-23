"""
PredictionTracker — Records model predictions and resolves outcomes.

Every cycle, each model makes a prediction for each pair. We record:
- What each model predicted (action + confidence)
- Market context (price, volatility, regime)
- What ACTUALLY happened 6 hours later (to match training label lookahead)
- Whether a trade was executed and its P&L

This data feeds the EnsembleEvaluator for dynamic weight adjustment
and the AdaptiveTriggerEngine for retrain decisions.
"""

import sqlite3
import logging
import os
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class PredictionTracker:

    def __init__(self, db_path: str = "data/trades.db"):
        self.db_path = db_path
        self._outcomes_since_last_update = 0
        self._create_tables()

    def _create_tables(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_predictions (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                pair TEXT NOT NULL,

                -- Per-model predictions
                xgb_action INTEGER,
                xgb_conf REAL,
                rl_action INTEGER,
                rl_conf REAL,
                lstm_action INTEGER,
                lstm_conf REAL,

                -- Ensemble output
                ensemble_action INTEGER,
                ensemble_conf REAL,

                -- Market context at decision time
                price_at_decision REAL,
                volatility REAL,
                regime TEXT,
                trend_strength REAL,

                -- Actual outcomes (filled in after lookahead period)
                price_after_6h REAL,
                actual_return_6h REAL,
                actual_label INTEGER,
                outcome_resolved INTEGER DEFAULT 0,
                resolved_at TEXT,

                -- Per-model correctness (filled after outcome)
                xgb_correct INTEGER,
                rl_correct INTEGER,
                lstm_correct INTEGER,
                ensemble_correct INTEGER,

                -- Trade execution info (if a trade was made)
                trade_executed INTEGER DEFAULT 0,
                trade_id TEXT,
                trade_pnl REAL,
                trade_pnl_pct REAL,
                trade_closed_at TEXT
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_predictions_timestamp
            ON model_predictions(timestamp)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_predictions_unresolved
            ON model_predictions(outcome_resolved, timestamp)
        """)

        conn.commit()
        conn.close()
        logger.info("PredictionTracker initialized")

    def record_prediction(
        self,
        pair: str,
        xgb_action: int, xgb_conf: float,
        rl_action: int, rl_conf: float,
        lstm_action: int, lstm_conf: float,
        ensemble_action: int, ensemble_conf: float,
        price: float,
        volatility: float = 0.0,
        regime: str = "unknown",
        trend_strength: float = 0.0,
    ) -> str:
        """Record a prediction from all models. Returns prediction ID."""
        pred_id = str(uuid.uuid4())[:12]
        now = datetime.now().isoformat()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO model_predictions
            (id, timestamp, pair,
             xgb_action, xgb_conf, rl_action, rl_conf, lstm_action, lstm_conf,
             ensemble_action, ensemble_conf,
             price_at_decision, volatility, regime, trend_strength)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            pred_id, now, pair,
            xgb_action, xgb_conf, rl_action, rl_conf, lstm_action, lstm_conf,
            ensemble_action, ensemble_conf,
            price, volatility, regime, trend_strength,
        ))
        conn.commit()
        conn.close()
        return pred_id

    def record_trade_execution(self, prediction_id: str, trade_id: str):
        """Link a prediction to an executed trade."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE model_predictions
            SET trade_executed = 1, trade_id = ?
            WHERE id = ?
        """, (trade_id, prediction_id))
        conn.commit()
        conn.close()

    def record_trade_outcome(self, trade_id: str, pnl: float, pnl_pct: float):
        """Record P&L when a trade is closed."""
        now = datetime.now().isoformat()
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE model_predictions
            SET trade_pnl = ?, trade_pnl_pct = ?, trade_closed_at = ?
            WHERE trade_id = ?
        """, (pnl, pnl_pct, now, trade_id))
        conn.commit()
        conn.close()

    def resolve_pending_outcomes(self, current_prices: Dict[str, float], lookahead_hours: int = 6):
        """
        Fill in actual outcomes for predictions older than lookahead_hours.
        Called every cycle.
        """
        cutoff = (datetime.now() - timedelta(hours=lookahead_hours)).isoformat()

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Get unresolved predictions older than lookahead
        cursor.execute("""
            SELECT id, pair, price_at_decision, ensemble_action,
                   xgb_action, rl_action, lstm_action
            FROM model_predictions
            WHERE outcome_resolved = 0 AND timestamp < ?
        """, (cutoff,))

        rows = cursor.fetchall()
        resolved_count = 0

        for row in rows:
            pair = row["pair"]
            price_now = current_prices.get(pair)
            if price_now is None:
                continue

            price_at = row["price_at_decision"]
            if price_at <= 0:
                continue

            # Calculate actual return
            actual_return = (price_now - price_at) / price_at

            # Determine what the "correct" action was
            # Using same logic as training labels: >0.3% = BUY, <-0.3% = SELL
            if actual_return > 0.003:
                actual_label = 1   # Should have bought
            elif actual_return < -0.003:
                actual_label = -1  # Should have sold
            else:
                actual_label = 0   # HOLD was correct

            # Score each model
            xgb_correct = 1 if row["xgb_action"] == actual_label else 0
            rl_correct = 1 if row["rl_action"] == actual_label else 0
            lstm_correct = 1 if row["lstm_action"] == actual_label else 0
            ensemble_correct = 1 if row["ensemble_action"] == actual_label else 0

            # For directional trades, also count "not wrong" as partial credit
            # E.g., if actual = BUY and model said HOLD, that's better than SELL
            # But for weight scoring, we keep it binary for clarity

            now = datetime.now().isoformat()
            cursor.execute("""
                UPDATE model_predictions
                SET price_after_6h = ?, actual_return_6h = ?, actual_label = ?,
                    outcome_resolved = 1, resolved_at = ?,
                    xgb_correct = ?, rl_correct = ?, lstm_correct = ?,
                    ensemble_correct = ?
                WHERE id = ?
            """, (
                price_now, actual_return, actual_label, now,
                xgb_correct, rl_correct, lstm_correct, ensemble_correct,
                row["id"],
            ))
            resolved_count += 1

        conn.commit()
        conn.close()

        if resolved_count > 0:
            self._outcomes_since_last_update += resolved_count
            logger.info(f"Resolved {resolved_count} prediction outcomes")

        return resolved_count

    def has_new_outcomes(self, min_count: int = 5) -> bool:
        """Check if enough new outcomes have been resolved since last weight update."""
        return self._outcomes_since_last_update >= min_count

    def reset_outcome_counter(self):
        """Called after ensemble weights are updated."""
        self._outcomes_since_last_update = 0

    def get_recent_outcomes(self, window: int = 30) -> List[Dict]:
        """Get last N predictions with known outcomes for the evaluator."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM model_predictions
            WHERE outcome_resolved = 1
              AND ensemble_action != 0
            ORDER BY timestamp DESC
            LIMIT ?
        """, (window,))

        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return rows

    def get_all_recent_outcomes(self, window: int = 30) -> List[Dict]:
        """Get last N predictions including HOLD decisions."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM model_predictions
            WHERE outcome_resolved = 1
            ORDER BY timestamp DESC
            LIMIT ?
        """, (window,))

        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return rows

    def get_performance_metrics(self) -> Dict:
        """Aggregate performance metrics for the trigger engine."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Last 30 resolved non-HOLD predictions
        cursor.execute("""
            SELECT xgb_correct, rl_correct, lstm_correct, ensemble_correct,
                   ensemble_conf, actual_return_6h, trade_pnl, trade_executed
            FROM model_predictions
            WHERE outcome_resolved = 1 AND ensemble_action != 0
            ORDER BY timestamp DESC
            LIMIT 30
        """)
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return {
                "total_predictions": 0,
                "xgb_accuracy": 0.5,
                "rl_accuracy": 0.5,
                "lstm_accuracy": 0.5,
                "ensemble_accuracy": 0.5,
                "avg_confidence": 0.5,
                "rolling_pnl": 0.0,
                "trade_count": 0,
                "data_ready": False,
            }

        n = len(rows)
        xgb_acc = sum(r["xgb_correct"] or 0 for r in rows) / n
        rl_acc = sum(r["rl_correct"] or 0 for r in rows) / n
        lstm_acc = sum(r["lstm_correct"] or 0 for r in rows) / n
        ens_acc = sum(r["ensemble_correct"] or 0 for r in rows) / n
        avg_conf = sum(r["ensemble_conf"] or 0 for r in rows) / n
        rolling_pnl = sum(r["trade_pnl"] or 0 for r in rows if r["trade_executed"])
        trade_count = sum(1 for r in rows if r["trade_executed"])

        return {
            "total_predictions": n,
            "xgb_accuracy": round(xgb_acc, 3),
            "rl_accuracy": round(rl_acc, 3),
            "lstm_accuracy": round(lstm_acc, 3),
            "ensemble_accuracy": round(ens_acc, 3),
            "avg_confidence": round(avg_conf, 3),
            "rolling_pnl": round(rolling_pnl, 4),
            "trade_count": trade_count,
            "data_ready": n >= 15,
        }

    def get_prediction_count(self) -> int:
        """Total number of recorded predictions."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM model_predictions")
        count = cursor.fetchone()[0]
        conn.close()
        return count

    def cleanup_old_records(self, keep_days: int = 90):
        """Remove records older than keep_days to prevent DB bloat."""
        cutoff = (datetime.now() - timedelta(days=keep_days)).isoformat()
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM model_predictions WHERE timestamp < ?",
            (cutoff,)
        )
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        if deleted:
            logger.info(f"Cleaned up {deleted} old prediction records")
        return deleted
