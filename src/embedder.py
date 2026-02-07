"""
ITS Intelligence Hub v2 - Embedding Module
Generates vector embeddings using Google Gemini text-embedding-004.
"""

import os
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json

import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)
from tqdm import tqdm
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()
logger = logging.getLogger(__name__)

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-004")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "50"))
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "768"))
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")


class EmbeddingError(Exception):
    """Base exception for embedding errors."""
    pass


class RateLimitError(EmbeddingError):
    """Rate limit hit - should retry with backoff."""
    pass


class QuotaExhaustedError(EmbeddingError):
    """Daily quota exhausted - should stop."""
    pass


@dataclass
class EmbeddingProgress:
    """Track embedding progress."""
    total_pending: int = 0
    total_processed: int = 0
    total_success: int = 0
    total_failed: int = 0
    failed_ids: List[str] = None
    
    def __post_init__(self):
        if self.failed_ids is None:
            self.failed_ids = []


class GeminiEmbedder:
    """
    Generates embeddings using Google Gemini API.
    Features batching, rate limiting, and resumable processing.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = EMBEDDING_MODEL,
        batch_size: int = EMBEDDING_BATCH_SIZE
    ) -> None:
        """
        Initialize Gemini embedder.
        
        Args:
            api_key: Gemini API key (or from GEMINI_API_KEY env)
            model: Embedding model name
            batch_size: Records per API batch
        """
        self.api_key = api_key or GEMINI_API_KEY
        self.model = model
        self.batch_size = batch_size
        self.progress = EmbeddingProgress()
        
        if not self.api_key:
            raise ValueError("Gemini API key required. Set GEMINI_API_KEY environment variable.")
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        
        # Initialize Supabase client
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError("Supabase credentials required.")
        self.supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        logger.info(f"Initialized Gemini embedder with model: {model}")
    
    def _build_embedding_input(self, title: str, abstract: Optional[str]) -> str:
        """
        Build input text for embedding.
        
        Uses hybrid format: "Title: {title}. Abstract: {abstract}"
        Falls back to title only if abstract missing.
        
        Args:
            title: Research title
            abstract: Research abstract (optional)
            
        Returns:
            Formatted input string
        """
        if abstract and len(abstract.strip()) > 50:
            # Truncate abstract if too long (Gemini has token limits)
            abstract_text = abstract[:8000] if len(abstract) > 8000 else abstract
            return f"Title: {title}. Abstract: {abstract_text}"
        return f"Title: {title}"
    
    @retry(
        retry=retry_if_exception_type(RateLimitError),
        wait=wait_exponential(multiplier=2, min=10, max=300),
        stop=stop_after_attempt(10),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
            
        Raises:
            RateLimitError: On 429 (triggers retry)
            QuotaExhaustedError: On quota exhaustion (stops)
        """
        try:
            # Use embed_content for batch embedding
            result = genai.embed_content(
                model=f"models/{self.model}",
                content=texts,
                task_type="retrieval_document"
            )
            
            # Handle single vs batch response
            if isinstance(result['embedding'][0], float):
                # Single embedding returned
                return [result['embedding']]
            return result['embedding']
            
        except google_exceptions.ResourceExhausted as e:
            error_msg = str(e).lower()
            if "quota" in error_msg or "exhausted" in error_msg:
                logger.error("Daily quota exhausted. Please try again tomorrow.")
                raise QuotaExhaustedError(str(e))
            logger.warning(f"Rate limit hit: {e}")
            raise RateLimitError(str(e))
            
        except google_exceptions.TooManyRequests as e:
            logger.warning(f"Too many requests: {e}")
            raise RateLimitError(str(e))
            
        except Exception as e:
            logger.error(f"Embedding API error: {e}")
            raise EmbeddingError(str(e))
    
    def embed_single(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector (768 dimensions)
        """
        embeddings = self._embed_batch([text])
        return embeddings[0]
    
    def fetch_pending_records(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Fetch records that need embeddings.
        
        Args:
            limit: Maximum records to fetch
            
        Returns:
            List of research records
        """
        try:
            response = (
                self.supabase.table("researches")
                .select("id, title, abstract")
                .is_("embedding", "null")
                .limit(limit)
                .execute()
            )
            return response.data or []
        except Exception as e:
            logger.error(f"Failed to fetch pending records: {e}")
            return []
    
    def update_embedding(self, record_id: str, embedding: List[float]) -> bool:
        """
        Update a single record's embedding in the database.
        
        Args:
            record_id: UUID of the research record
            embedding: Vector embedding
            
        Returns:
            True if successful
        """
        try:
            # Convert to string format for pgvector
            embedding_str = f"[{','.join(map(str, embedding))}]"
            
            self.supabase.table("researches").update({
                "embedding": embedding_str
            }).eq("id", record_id).execute()
            
            return True
        except Exception as e:
            logger.error(f"Failed to update embedding for {record_id}: {e}")
            return False
    
    def process_all(
        self,
        max_records: Optional[int] = None,
        show_progress: bool = True
    ) -> EmbeddingProgress:
        """
        Process all pending records and generate embeddings.
        
        Args:
            max_records: Maximum records to process (None for all)
            show_progress: Show tqdm progress bar
            
        Returns:
            EmbeddingProgress with statistics
        """
        # Fetch pending records
        limit = max_records or 10000
        records = self.fetch_pending_records(limit)
        
        self.progress.total_pending = len(records)
        logger.info(f"Found {len(records)} records needing embeddings")
        
        if not records:
            logger.info("No pending records to process")
            return self.progress
        
        # Process in batches
        pbar = tqdm(
            total=len(records),
            desc="Generating embeddings",
            disable=not show_progress
        )
        
        for i in range(0, len(records), self.batch_size):
            batch = records[i:i + self.batch_size]
            
            # Build input texts
            texts = []
            for record in batch:
                text = self._build_embedding_input(
                    record.get("title", ""),
                    record.get("abstract")
                )
                texts.append(text)
            
            try:
                # Generate embeddings
                embeddings = self._embed_batch(texts)
                
                # Update database
                for j, record in enumerate(batch):
                    if j < len(embeddings):
                        success = self.update_embedding(
                            record["id"],
                            embeddings[j]
                        )
                        if success:
                            self.progress.total_success += 1
                        else:
                            self.progress.total_failed += 1
                            self.progress.failed_ids.append(record["id"])
                    
                    self.progress.total_processed += 1
                    pbar.update(1)
                
                # Small delay between batches to avoid rate limits
                time.sleep(0.5)
                
            except QuotaExhaustedError:
                logger.error("Stopping due to quota exhaustion")
                break
                
            except RateLimitError as e:
                # Tenacity should have handled retries, but if we're here, give up on this batch
                logger.error(f"Batch failed after retries: {e}")
                for record in batch:
                    self.progress.total_failed += 1
                    self.progress.failed_ids.append(record["id"])
                    pbar.update(1)
                
            except EmbeddingError as e:
                logger.error(f"Embedding error: {e}")
                for record in batch:
                    self.progress.total_failed += 1
                    self.progress.failed_ids.append(record["id"])
                    pbar.update(1)
        
        pbar.close()
        
        logger.info(f"Embedding complete: {self.progress.total_success} succeeded, "
                   f"{self.progress.total_failed} failed")
        
        return self.progress
    
    def get_stats(self) -> Dict[str, int]:
        """Get embedding statistics from database."""
        try:
            response = self.supabase.rpc("get_embedding_stats").execute()
            if response.data:
                return response.data[0]
        except Exception as e:
            logger.warning(f"Could not fetch stats via RPC: {e}")
            # Fallback to direct queries
            try:
                total = self.supabase.table("researches").select("id", count="exact").execute()
                embedded = self.supabase.table("researches").select("id", count="exact").not_.is_("embedding", "null").execute()
                return {
                    "total_researches": total.count or 0,
                    "embedded_count": embedded.count or 0,
                    "pending_count": (total.count or 0) - (embedded.count or 0)
                }
            except:
                pass
        
        return {"total_researches": 0, "embedded_count": 0, "pending_count": 0}


def create_embedder(**kwargs) -> GeminiEmbedder:
    """Factory function to create embedder."""
    return GeminiEmbedder(**kwargs)


def main():
    """CLI entry point for embedding."""
    import argparse
    import colorlog
    
    # Setup logging
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s [%(levelname)s]%(reset)s %(message)s",
        datefmt="%H:%M:%S"
    ))
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)
    
    parser = argparse.ArgumentParser(description="Generate embeddings for research data")
    parser.add_argument("--limit", type=int, help="Maximum records to process")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size")
    parser.add_argument("--stats", action="store_true", help="Show stats only")
    
    args = parser.parse_args()
    
    embedder = GeminiEmbedder(batch_size=args.batch_size)
    
    if args.stats:
        stats = embedder.get_stats()
        print("\n📊 Embedding Statistics:")
        print(f"   Total researches: {stats.get('total_researches', 0)}")
        print(f"   Embedded: {stats.get('embedded_count', 0)}")
        print(f"   Pending: {stats.get('pending_count', 0)}")
        return
    
    print("\n🚀 Starting embedding pipeline...\n")
    progress = embedder.process_all(max_records=args.limit)
    
    print(f"\n✅ Complete!")
    print(f"   Processed: {progress.total_processed}")
    print(f"   Succeeded: {progress.total_success}")
    print(f"   Failed: {progress.total_failed}")


if __name__ == "__main__":
    main()
