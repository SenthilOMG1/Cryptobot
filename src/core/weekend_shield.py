"""
Weekend Shield
==============
Halves max leverage during weekend hours to protect against
low-liquidity volatility and Sunday dumps.

Active: Friday 20:00 UTC → Monday 04:00 UTC
"""

import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class WeekendShield:
    """Dynamically caps leverage during weekend hours."""

    def __init__(self, base_leverage: int):
        self._base = base_leverage

    @property
    def base_leverage(self) -> int:
        return self._base

    @staticmethod
    def is_active() -> bool:
        """Check if we're in the weekend shield window."""
        utc_now = datetime.now(tz=timezone.utc)
        return (
            (utc_now.weekday() == 4 and utc_now.hour >= 20)  # Friday after 20:00
            or utc_now.weekday() == 5                         # Saturday
            or utc_now.weekday() == 6                         # Sunday
            or (utc_now.weekday() == 0 and utc_now.hour < 4)  # Monday before 04:00
        )

    def get_leverage(self) -> int:
        """Return the effective max leverage for this moment."""
        if self.is_active():
            return max(2, self._base // 2)
        return self._base

    def apply(self, config) -> None:
        """Update config.futures.leverage in-place. Logs transitions."""
        target = self.get_leverage()
        if config.futures.leverage == target:
            return
        if self.is_active():
            logger.info(f"WEEKEND SHIELD ACTIVE: max leverage capped {self._base}x → {target}x")
        else:
            logger.info(f"WEEKEND SHIELD OFF: max leverage restored to {self._base}x")
        config.futures.leverage = target
