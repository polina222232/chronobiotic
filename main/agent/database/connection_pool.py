"""
Connection Pool Manager for Database Operations.

Manages database connections efficiently for high-performance agent operations.
"""

import logging
import threading
import time
from typing import Optional, List, Dict, Any
from django.db import connections, connection
from django.db.utils import OperationalError, DatabaseError

logger = logging.getLogger(__name__)


class ConnectionPoolManager:
    """Manages a pool of database connections."""

    def __init__(self, max_connections: int = 10):
        self.max_connections = max_connections
        self.active_connections = 0
        self.connection_lock = threading.Lock()
        self.connection_stats = {
            "total_requests": 0,
            "successful_connections": 0,
            "failed_connections": 0,
            "connection_wait_time": 0,
            "avg_connection_time": 0,
        }
        self._initialize_connections()

    def _initialize_connections(self):
        """Initialize database connections."""
        try:
            # Test initial connection
            connection.ensure_connection()
            logger.info("Initial database connection established")
            self.active_connections = 1
            self.connection_stats["successful_connections"] = 1
            self.connection_stats["total_requests"] = 1
        except Exception as e:
            logger.error(f"Failed to initialize database connection: {e}")
            self.connection_stats["failed_connections"] = 1

    def get_connection(self, timeout: float = 5.0) -> Optional[Any]:
        """
        Get a database connection from the pool.

        Args:
            timeout: Maximum time to wait for a connection (seconds)

        Returns:
            Database connection or None if unavailable
        """
        start_time = time.time()

        with self.connection_lock:
            self.connection_stats["total_requests"] += 1

            if self.active_connections >= self.max_connections:
                wait_time = time.time() - start_time
                if wait_time > timeout:
                    logger.warning(f"Connection pool timeout after {wait_time:.2f}s")
                    self.connection_stats["failed_connections"] += 1
                    return None

            try:
                # Ensure connection is alive
                connection.ensure_connection()
                self.active_connections += 1

                connection_time = time.time() - start_time
                self.connection_stats["connection_wait_time"] += connection_time
                self.connection_stats["successful_connections"] += 1

                # Update average
                total_success = self.connection_stats["successful_connections"]
                total_wait = self.connection_stats["connection_wait_time"]
                self.connection_stats["avg_connection_time"] = (
                    total_wait / total_success
                )

                logger.debug(
                    f"Connection acquired in {connection_time:.3f}s. Active: {self.active_connections}"
                )
                return connection

            except (OperationalError, DatabaseError) as e:
                logger.error(f"Database connection error: {e}")
                self.connection_stats["failed_connections"] += 1
                return None
            except Exception as e:
                logger.error(f"Unexpected connection error: {e}")
                self.connection_stats["failed_connections"] += 1
                return None

    def release_connection(self):
        """Release a connection back to the pool."""
        with self.connection_lock:
            if self.active_connections > 0:
                self.active_connections -= 1
                logger.debug(f"Connection released. Active: {self.active_connections}")
            else:
                logger.warning("Attempted to release connection when none were active")

    def close_all_connections(self):
        """Close all database connections."""
        with self.connection_lock:
            try:
                for conn in connections.all():
                    conn.close_if_unusable_or_obsolete()
                self.active_connections = 0
                logger.info("All database connections closed")
            except Exception as e:
                logger.error(f"Error closing connections: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        stats = self.connection_stats.copy()
        stats.update(
            {
                "active_connections": self.active_connections,
                "max_connections": self.max_connections,
                "available_connections": self.max_connections - self.active_connections,
                "utilization_percentage": (
                    self.active_connections / self.max_connections
                )
                * 100,
            }
        )
        return stats

    def health_check(self) -> bool:
        """Perform health check on database connections."""
        try:
            with self.connection_lock:
                # Test connection
                connection.ensure_connection()

                # Test simple query
                with connection.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    result = cursor.fetchone()

                if result and result[0] == 1:
                    logger.debug("Database health check passed")
                    return True
                else:
                    logger.warning(
                        "Database health check failed - unexpected query result"
                    )
                    return False

        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False

    def optimize_pool(self, target_utilization: float = 0.7):
        """
        Optimize connection pool size based on utilization.

        Args:
            target_utilization: Target utilization percentage (0-1)
        """
        with self.connection_lock:
            current_utilization = self.active_connections / self.max_connections

            if current_utilization > target_utilization * 1.2:  # Overutilized
                new_size = min(self.max_connections * 2, 50)  # Max 50 connections
                logger.info(
                    f"Increasing connection pool from {self.max_connections} to {new_size}"
                )
                self.max_connections = new_size

            elif current_utilization < target_utilization * 0.5:  # Underutilized
                new_size = max(
                    int(self.max_connections * 0.8), 1
                )  # Minimum 1 connection
                logger.info(
                    f"Decreasing connection pool from {self.max_connections} to {new_size}"
                )
                self.max_connections = new_size

    def reset_stats(self):
        """Reset connection statistics."""
        with self.connection_lock:
            self.connection_stats = {
                "total_requests": 0,
                "successful_connections": 0,
                "failed_connections": 0,
                "connection_wait_time": 0,
                "avg_connection_time": 0,
            }
            logger.info("Connection pool statistics reset")


# Global connection pool instance
connection_pool = ConnectionPoolManager()


def get_connection_pool() -> ConnectionPoolManager:
    """Get the global connection pool instance."""
    return connection_pool
