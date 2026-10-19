"""
Database Connector Module using SQLAlchemy.
Provides connection pooling, safe query execution, and graceful degradation
when no remote database is configured.
"""

from typing import Optional, Dict, Any
import logging
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from app.config import DATABASE_URL

logger = logging.getLogger(__name__)


class DatabaseConnector:
    """Manages PostgreSQL / Supabase connection pooling and query execution."""

    def __init__(self, db_url: Optional[str] = None):
        self.db_url = db_url or DATABASE_URL
        self.engine: Optional[Engine] = None
        self._init_engine()

    def _init_engine(self) -> None:
        """Initializes SQLAlchemy Engine if valid DB URL is present."""
        if not self.db_url:
            logger.info("DATABASE_URL is not set. Database connector in offline mode.")
            return

        url = self.db_url.strip()
        # Ensure psycopg2 dialect is used if raw postgresql:// is provided
        if url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+psycopg2://", 1)

        try:
            self.engine = create_engine(
                url,
                pool_pre_ping=True,
                pool_size=5,
                max_overflow=10,
                connect_args={
                    "connect_timeout": 10,
                    "keepalives": 1,
                    "keepalives_idle": 30,
                    "keepalives_interval": 10,
                    "keepalives_count": 5,
                }
            )
        except Exception as e:
            logger.error(f"Failed to initialize database engine: {e}")
            self.engine = None

    @property
    def is_configured(self) -> bool:
        """Returns True if engine is initialized."""
        return self.engine is not None

    def test_connection(self) -> bool:
        """Tests whether database is reachable."""
        if not self.engine:
            return False
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                return True
        except Exception as e:
            logger.warning(f"Database connection test failed: {e}")
            return False

    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Executes a SQL query and returns results as a Pandas DataFrame."""
        if not self.engine:
            raise ConnectionError("Database is not configured. Set DATABASE_URL in .env.")

        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query), params or {})
                rows = result.fetchall()
                columns = result.keys()
                return pd.DataFrame(rows, columns=columns)
        except Exception as e:
            logger.error(f"Error executing database query: {e}")
            raise