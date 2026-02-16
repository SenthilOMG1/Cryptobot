"""
Watchdog - Self-Healing System
==============================
Monitors bot health and auto-recovers from failures.
"""

import os
import sys
import time
import logging
import traceback
from datetime import datetime, timedelta
from typing import Optional, Callable, Any
from functools import wraps

logger = logging.getLogger(__name__)


class Watchdog:
    """
    Self-healing watchdog system.

    Features:
    - Monitors system health
    - Auto-restarts on critical failures
    - Recovers state after crash
    - Handles API outages gracefully
    """

    def __init__(self, max_retries: int = 5, retry_delay: int = 60):
        """
        Initialize watchdog.

        Args:
            max_retries: Maximum consecutive failures before giving up
            retry_delay: Seconds to wait between retries
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.consecutive_failures = 0
        self.last_error: Optional[str] = None
        self.last_error_time: Optional[datetime] = None
        self.last_successful_run: Optional[datetime] = None
        self.total_errors = 0
        self.recovered_count = 0

    def monitor(self, func: Callable) -> Callable:
        """
        Decorator to monitor a function and auto-recover on failure.

        Usage:
            @watchdog.monitor
            def main_loop():
                ...
        """
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            while True:
                try:
                    result = func(*args, **kwargs)
                    self._on_success()
                    return result

                except KeyboardInterrupt:
                    logger.info("Received shutdown signal")
                    raise

                except Exception as e:
                    should_continue = self._on_error(e)
                    if not should_continue:
                        raise

        return wrapper

    def _on_success(self):
        """Called when monitored function succeeds."""
        self.consecutive_failures = 0
        self.last_successful_run = datetime.now()

    def _on_error(self, error: Exception) -> bool:
        """
        Handle an error.

        Args:
            error: The exception that occurred

        Returns:
            True if should retry, False if should give up
        """
        self.consecutive_failures += 1
        self.total_errors += 1
        self.last_error = str(error)
        self.last_error_time = datetime.now()

        logger.error(f"Error #{self.consecutive_failures}: {error}")
        logger.error(traceback.format_exc())

        # Check if we've exceeded max retries
        if self.consecutive_failures >= self.max_retries:
            logger.critical(f"Max retries ({self.max_retries}) exceeded. Giving up.")
            return False

        # Calculate backoff delay (exponential)
        delay = self.retry_delay * (2 ** (self.consecutive_failures - 1))
        delay = min(delay, 300)  # Max 5 minutes

        logger.warning(f"Retrying in {delay} seconds... (attempt {self.consecutive_failures + 1}/{self.max_retries})")
        time.sleep(delay)

        self.recovered_count += 1
        return True

    def handle_api_outage(self, error: Exception) -> bool:
        """
        Handle API outage specifically.

        Returns:
            True if should retry
        """
        error_str = str(error).lower()

        # Detect API-related errors
        api_errors = ["connection", "timeout", "rate limit", "502", "503", "504"]
        is_api_error = any(e in error_str for e in api_errors)

        if is_api_error:
            logger.warning(f"API outage detected: {error}")
            logger.warning("Waiting 30 seconds before retry...")
            time.sleep(30)
            return True

        return self._on_error(error)

    def recover_state(self, components: dict) -> bool:
        """
        Recover system state after restart.

        Args:
            components: Dict of component instances to recover

        Returns:
            True if recovery successful
        """
        logger.info("Starting state recovery...")

        try:
            # Recover positions from exchange
            if "position_tracker" in components:
                components["position_tracker"].sync_from_exchange()
                logger.info("Positions recovered from exchange")

            # Reload ML models
            if "xgb_model" in components:
                components["xgb_model"].load_model()
                logger.info("XGBoost model reloaded")

            if "rl_agent" in components:
                components["rl_agent"].load_model()
                logger.info("RL agent reloaded")

            logger.info("State recovery complete")
            return True

        except Exception as e:
            logger.error(f"State recovery failed: {e}")
            return False

    def get_status(self) -> dict:
        """Get watchdog status."""
        return {
            "consecutive_failures": self.consecutive_failures,
            "total_errors": self.total_errors,
            "recovered_count": self.recovered_count,
            "last_error": self.last_error,
            "last_error_time": self.last_error_time.isoformat() if self.last_error_time else None,
            "last_successful_run": self.last_successful_run.isoformat() if self.last_successful_run else None,
            "is_healthy": self.consecutive_failures == 0
        }


def with_retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Decorator for automatic retry with exponential backoff.

    Usage:
        @with_retry(max_attempts=3, delay=1.0)
        def api_call():
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < max_attempts - 1:
                        wait_time = delay * (backoff ** attempt)
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_attempts}): {e}. "
                            f"Retrying in {wait_time:.1f}s..."
                        )
                        time.sleep(wait_time)
            raise last_error
        return wrapper
    return decorator
