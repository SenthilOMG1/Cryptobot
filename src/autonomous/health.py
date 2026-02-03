"""
Health Monitor
==============
Monitors system health and provides status reports.
"""

import os
import psutil
import logging
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class HealthMonitor:
    """
    System health monitoring.

    Features:
    - CPU/Memory usage tracking
    - Component health checks
    - Uptime tracking
    - Status reporting
    """

    def __init__(self):
        """Initialize health monitor."""
        self.start_time = datetime.now()
        self.component_status: Dict[str, bool] = {}
        self.last_check: Optional[datetime] = None
        self.error_count = 0
        self.warning_count = 0

    def check_system_health(self) -> Dict[str, Any]:
        """
        Check overall system health.

        Returns:
            Dict with health metrics
        """
        self.last_check = datetime.now()

        # System resources
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        # Uptime
        uptime = datetime.now() - self.start_time
        uptime_hours = uptime.total_seconds() / 3600

        health = {
            "timestamp": datetime.now().isoformat(),
            "uptime_hours": round(uptime_hours, 2),
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_mb": memory.available / (1024 * 1024),
                "disk_percent": disk.percent,
            },
            "components": self.component_status,
            "errors": self.error_count,
            "warnings": self.warning_count,
            "is_healthy": self._is_healthy(cpu_percent, memory.percent)
        }

        return health

    def _is_healthy(self, cpu: float, memory: float) -> bool:
        """Determine if system is healthy based on resources."""
        # Unhealthy if resources are critically high
        if cpu > 95 or memory > 95:
            return False

        # Unhealthy if any component is down
        if not all(self.component_status.values()):
            return False

        return True

    def register_component(self, name: str, healthy: bool = True):
        """Register a component for monitoring."""
        self.component_status[name] = healthy
        logger.debug(f"Component registered: {name} (healthy: {healthy})")

    def update_component(self, name: str, healthy: bool):
        """Update component health status."""
        old_status = self.component_status.get(name)
        self.component_status[name] = healthy

        if old_status != healthy:
            if healthy:
                logger.info(f"Component recovered: {name}")
            else:
                logger.warning(f"Component unhealthy: {name}")
                self.warning_count += 1

    def check_component(self, name: str, check_func) -> bool:
        """
        Check a component's health using provided function.

        Args:
            name: Component name
            check_func: Function that returns True if healthy

        Returns:
            True if component is healthy
        """
        try:
            healthy = check_func()
            self.update_component(name, healthy)
            return healthy
        except Exception as e:
            logger.error(f"Health check failed for {name}: {e}")
            self.update_component(name, False)
            self.error_count += 1
            return False

    def record_error(self):
        """Record an error occurrence."""
        self.error_count += 1

    def record_warning(self):
        """Record a warning occurrence."""
        self.warning_count += 1

    def get_uptime(self) -> str:
        """Get formatted uptime string."""
        uptime = datetime.now() - self.start_time
        days = uptime.days
        hours, remainder = divmod(uptime.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        if days > 0:
            return f"{days}d {hours}h {minutes}m"
        elif hours > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{minutes}m {seconds}s"

    def get_summary(self) -> str:
        """Get health summary as string."""
        health = self.check_system_health()

        status = "HEALTHY" if health["is_healthy"] else "UNHEALTHY"
        uptime = self.get_uptime()

        components_ok = sum(1 for v in self.component_status.values() if v)
        components_total = len(self.component_status)

        return (
            f"Status: {status} | "
            f"Uptime: {uptime} | "
            f"CPU: {health['system']['cpu_percent']:.1f}% | "
            f"Memory: {health['system']['memory_percent']:.1f}% | "
            f"Components: {components_ok}/{components_total}"
        )
