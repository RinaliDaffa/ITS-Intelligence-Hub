"""
ITS Intelligence Hub v2 - Database Module
Supabase interface for research data persistence.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from dotenv import load_dotenv
from supabase import create_client, Client

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class DatabaseClient:
    """Supabase database client for research data operations."""
    
    TABLE_NAME = "researches"
    
    def __init__(
        self,
        url: Optional[str] = None,
        key: Optional[str] = None,
        batch_size: int = 50
    ) -> None:
        """
        Initialize Supabase client.
        
        Args:
            url: Supabase project URL (or from SUPABASE_URL env)
            key: Supabase API key (or from SUPABASE_KEY env)
            batch_size: Records per batch for bulk operations
        """
        self.url = url or os.getenv("SUPABASE_URL")
        self.key = key or os.getenv("SUPABASE_KEY")
        self.batch_size = batch_size
        
        if not self.url or not self.key:
            raise ValueError(
                "Supabase credentials required. Set SUPABASE_URL and SUPABASE_KEY "
                "environment variables or pass them to constructor."
            )
        
        self._client: Optional[Client] = None
        self._stats = {
            "inserted": 0,
            "updated": 0,
            "failed": 0,
            "skipped": 0
        }
    
    @property
    def client(self) -> Client:
        """Lazy-initialize Supabase client."""
        if self._client is None:
            try:
                self._client = create_client(self.url, self.key)
                logger.info("Successfully connected to Supabase")
            except Exception as e:
                logger.error(f"Failed to connect to Supabase: {e}")
                raise
        return self._client
    
    @property
    def stats(self) -> Dict[str, int]:
        """Get current operation statistics."""
        return self._stats.copy()
    
    def reset_stats(self) -> None:
        """Reset operation statistics."""
        self._stats = {"inserted": 0, "updated": 0, "failed": 0, "skipped": 0}
    
    def upsert(self, record: Dict[str, Any]) -> bool:
        """
        Upsert a single research record.
        Uses URL as the conflict key for deduplication.
        
        Args:
            record: Cleaned research record
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Validate required fields
            if not record.get("url"):
                logger.warning("Skipping record without URL")
                self._stats["skipped"] += 1
                return False
            
            if not record.get("title"):
                logger.warning(f"Skipping record without title: {record.get('url')}")
                self._stats["skipped"] += 1
                return False
            
            # Prepare record for insertion
            db_record = self._prepare_record(record)
            
            # Perform upsert (on_conflict for URL)
            response = (
                self.client.table(self.TABLE_NAME)
                .upsert(db_record, on_conflict="url")
                .execute()
            )
            
            if response.data:
                logger.debug(f"Upserted record: {record.get('title', '')[:50]}")
                self._stats["inserted"] += 1
                return True
            else:
                logger.warning(f"No data returned for upsert: {record.get('url')}")
                self._stats["failed"] += 1
                return False
                
        except Exception as e:
            logger.error(f"Failed to upsert record {record.get('url')}: {e}")
            self._stats["failed"] += 1
            return False
    
    def upsert_batch(self, records: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Upsert multiple records in batches.
        
        Args:
            records: List of cleaned research records
            
        Returns:
            Statistics dict with inserted/failed counts
        """
        total = len(records)
        batch_stats = {"inserted": 0, "failed": 0, "skipped": 0}
        
        logger.info(f"Starting batch upsert of {total} records (batch size: {self.batch_size})")
        
        for i in range(0, total, self.batch_size):
            batch = records[i:i + self.batch_size]
            
            # Filter valid records
            valid_records = []
            for record in batch:
                if record.get("url") and record.get("title"):
                    valid_records.append(self._prepare_record(record))
                else:
                    batch_stats["skipped"] += 1
                    logger.warning(f"Skipping invalid record (missing url/title)")
            
            if not valid_records:
                continue
            
            try:
                response = (
                    self.client.table(self.TABLE_NAME)
                    .upsert(valid_records, on_conflict="url")
                    .execute()
                )
                
                if response.data:
                    inserted_count = len(response.data)
                    batch_stats["inserted"] += inserted_count
                    self._stats["inserted"] += inserted_count
                    logger.info(f"Batch {i // self.batch_size + 1}: inserted {inserted_count} records")
                    
            except Exception as e:
                batch_stats["failed"] += len(valid_records)
                self._stats["failed"] += len(valid_records)
                logger.error(f"Batch {i // self.batch_size + 1} failed: {e}")
        
        return batch_stats
    
    def get_existing_urls(self, urls: List[str]) -> set:
        """
        Check which URLs already exist in database.
        
        Args:
            urls: List of URLs to check
            
        Returns:
            Set of URLs that exist in database
        """
        try:
            existing = set()
            
            # Query in batches to avoid URL length limits
            for i in range(0, len(urls), 100):
                batch = urls[i:i + 100]
                response = (
                    self.client.table(self.TABLE_NAME)
                    .select("url")
                    .in_("url", batch)
                    .execute()
                )
                
                if response.data:
                    existing.update(r["url"] for r in response.data)
            
            return existing
            
        except Exception as e:
            logger.error(f"Failed to check existing URLs: {e}")
            return set()
    
    def get_record_count(self, degree: Optional[str] = None) -> int:
        """
        Get total record count, optionally filtered by degree.
        
        Args:
            degree: Filter by degree (S1, S2, S3)
            
        Returns:
            Total record count
        """
        try:
            query = self.client.table(self.TABLE_NAME).select("id", count="exact")
            
            if degree:
                query = query.eq("degree", degree.upper())
            
            response = query.execute()
            return response.count or 0
            
        except Exception as e:
            logger.error(f"Failed to get record count: {e}")
            return 0
    
    def _prepare_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare record for database insertion.
        
        Args:
            record: Raw record dict
            
        Returns:
            Formatted record for Supabase
        """
        # Handle supervisors array
        supervisors = record.get("supervisors")
        if isinstance(supervisors, list):
            supervisors = [s for s in supervisors if s]  # Remove empty strings
        else:
            supervisors = None
        
        # Handle metadata as JSONB
        metadata = record.get("metadata", {})
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {"raw": metadata}
        
        return {
            "title": record.get("title", "")[:2000],  # Truncate if needed
            "abstract": record.get("abstract"),
            "author": record.get("author", "Unknown")[:500],
            "supervisors": supervisors,
            "degree": record.get("degree", "")[:50],
            "year": record.get("year"),
            "dept": record.get("dept", "")[:200] if record.get("dept") else None,
            "url": record.get("url"),
            "data_confidence": record.get("data_confidence"),
            "metadata": metadata
        }


class MockDatabaseClient:
    """Mock database client for dry-run testing."""
    
    def __init__(self, *args, **kwargs) -> None:
        self._records: List[Dict[str, Any]] = []
        self._stats = {"inserted": 0, "updated": 0, "failed": 0, "skipped": 0}
    
    @property
    def stats(self) -> Dict[str, int]:
        return self._stats.copy()
    
    def reset_stats(self) -> None:
        self._stats = {"inserted": 0, "updated": 0, "failed": 0, "skipped": 0}
    
    def upsert(self, record: Dict[str, Any]) -> bool:
        if record.get("url") and record.get("title"):
            self._records.append(record)
            self._stats["inserted"] += 1
            return True
        self._stats["skipped"] += 1
        return False
    
    def upsert_batch(self, records: List[Dict[str, Any]]) -> Dict[str, int]:
        batch_stats = {"inserted": 0, "failed": 0, "skipped": 0}
        for record in records:
            if self.upsert(record):
                batch_stats["inserted"] += 1
            else:
                batch_stats["skipped"] += 1
        return batch_stats
    
    def get_existing_urls(self, urls: List[str]) -> set:
        return {r["url"] for r in self._records if r.get("url") in urls}
    
    def get_record_count(self, degree: Optional[str] = None) -> int:
        if degree:
            return sum(1 for r in self._records if r.get("degree") == degree)
        return len(self._records)
    
    def get_all_records(self) -> List[Dict[str, Any]]:
        """Get all stored records (for testing)."""
        return self._records.copy()


def get_database_client(dry_run: bool = False, **kwargs) -> DatabaseClient:
    """
    Factory function to get appropriate database client.
    
    Args:
        dry_run: If True, return mock client
        **kwargs: Passed to DatabaseClient
        
    Returns:
        DatabaseClient or MockDatabaseClient instance
    """
    if dry_run:
        logger.info("Using mock database client (dry-run mode)")
        return MockDatabaseClient(**kwargs)
    return DatabaseClient(**kwargs)
