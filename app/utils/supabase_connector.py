"""
Supabase REST Client Wrapper.
Provides secure and fault-tolerant operations with Supabase.
"""

from typing import Optional, Any
import logging
from app.config import SUPABASE_URL, SUPABASE_KEY

logger = logging.getLogger(__name__)


class SupabaseConnector:
    """Safely connects to Supabase REST API."""

    def __init__(self, url: Optional[str] = None, key: Optional[str] = None):
        self.url = (url or SUPABASE_URL or "").strip()
        self.key = (key or SUPABASE_KEY or "").strip()
        self.client = None
        self._init_client()

    def _init_client(self) -> None:
        if not self.url or not self.key:
            logger.info("SUPABASE_URL or SUPABASE_KEY not configured.")
            return

        try:
            from supabase import create_client
            self.client = create_client(self.url, self.key)
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {e}")
            self.client = None

    @property
    def is_configured(self) -> bool:
        return self.client is not None

    def fetch_table(self, table_name: str, select: str = "*", limit: Optional[int] = None) -> Any:
        """Fetches rows from a Supabase table."""
        if not self.client:
            raise ConnectionError("Supabase client is not configured. Set SUPABASE_URL and SUPABASE_KEY.")

        query = self.client.table(table_name).select(select)
        if limit:
            query = query.limit(limit)
        return query.execute()