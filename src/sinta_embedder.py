"""
ITS Intelligence Hub v2 - SINTA Embedder Module
Generates expertise embeddings from SINTA publication titles.

Features:
- Concatenates publication titles into expertise string
- Generates 768-dimensional vectors via Gemini text-embedding-004
- Upserts vectors to lecturers.expertise_vector column
- Resumable processing with progress tracking
"""

import os
import time
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from google import genai
from google.genai import types
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

# ============================================================================
# Configuration
# ============================================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-004")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "768"))
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "50"))
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Expertise string configuration
MIN_PUBLICATIONS = 1  # Lowered to 1 to capture new lecturers
MAX_PUBLICATIONS = 20  # Maximum publications to use for embedding


# ============================================================================
# Exceptions
# ============================================================================

class EmbeddingError(Exception):
    """Base exception for embedding errors."""
    pass


class RateLimitError(EmbeddingError):
    """Rate limit hit - should retry with backoff."""
    pass


class QuotaExhaustedError(EmbeddingError):
    """Daily quota exhausted - should stop."""
    pass


# ============================================================================
# Progress Tracking
# ============================================================================

@dataclass
class EmbeddingProgress:
    """Track embedding progress."""
    total_pending: int = 0
    total_processed: int = 0
    total_success: int = 0
    total_failed: int = 0
    total_skipped: int = 0  # Lecturers with too few publications
    failed_ids: List[str] = None
    
    def __post_init__(self):
        if self.failed_ids is None:
            self.failed_ids = []


# ============================================================================
# SINTA Embedder Class
# ============================================================================

class SINTAEmbedder:
    """
    Generates expertise embeddings for lecturers based on their SINTA publications.
    
    Expertise DNA = Embedding of concatenated publication titles
    This captures the semantic essence of a lecturer's research focus.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = EMBEDDING_MODEL,
        batch_size: int = EMBEDDING_BATCH_SIZE,
        min_publications: int = MIN_PUBLICATIONS,
        max_publications: int = MAX_PUBLICATIONS
    ) -> None:
        """
        Initialize SINTA embedder.
        
        Args:
            api_key: Gemini API key (or from GEMINI_API_KEY env)
            model: Embedding model name
            batch_size: Records per API batch
            min_publications: Minimum publications required for embedding
            max_publications: Maximum publications to use
        """
        self.api_key = api_key or GEMINI_API_KEY
        self.model = model
        self.batch_size = batch_size
        self.min_publications = min_publications
        self.max_publications = max_publications
        self.progress = EmbeddingProgress()
        
        if not self.api_key:
            raise ValueError("Gemini API key required. Set GEMINI_API_KEY environment variable.")
        
        # Configure new Gemini client
        self.client = genai.Client(api_key=self.api_key)
        
        # Initialize Supabase client
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError("Supabase credentials required.")
        self.supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        logger.info(f"Initialized SINTA embedder with model: {model}")
    
    def _build_expertise_string(self, name: str, dept: str, publications: List[Dict[str, Any]]) -> str:
        """
        Build expertise string from publication titles.
        
        Format: "RETRIEVAL_DOCUMENT: Academic Expertise Profile of {Name}, Department of {Dept}. Research Focus: {Pub1}; {Pub2}..."
        
        Args:
            name: Lecturer name
            dept: Department name
            publications: List of publication dicts with 'title' key
            
        Returns:
            Formatted expertise string
        """
        if not publications:
            return ""
        
        # Get titles, limited to max_publications
        titles = []
        for pub in publications[:self.max_publications]:
            title = pub.get("title", "").strip()
            if title and len(title) > 10:
                titles.append(title)
        
        if not titles:
            return ""
        
        # Format: "RETRIEVAL_DOCUMENT: Academic Expertise Profile of [Name], Department of [Dept]. Research Focus: [Pub 1]; [Pub 2]..."
        dept_str = dept if dept else "Unknown Department"
        expertise_text = f"RETRIEVAL_DOCUMENT: Academic Expertise Profile of {name}, Department of {dept_str}. Research Focus: {'; '.join(titles)}"
        
        # Truncate if too long (Gemini has token limits)
        if len(expertise_text) > 8000:
            expertise_text = expertise_text[:8000]
        
        return expertise_text
    
    @retry(
        retry=retry_if_exception_type(RateLimitError),
        wait=wait_exponential(multiplier=2, min=10, max=300),
        stop=stop_after_attempt(10),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    def _embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector (768 dimensions)
            
        Raises:
            RateLimitError: On 429 (triggers retry)
            QuotaExhaustedError: On quota exhaustion (stops)
        """
        try:
            result = self.client.models.embed_content(
                model=self.model,
                contents=text,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=EMBEDDING_DIMENSION
                )
            )
            
            # Get embedding from response
            embedding = result.embeddings[0].values
            
            return list(embedding)
            
        except Exception as e:
            error_msg = str(e).lower()
            if "429" in str(e) or "rate" in error_msg or "quota" in error_msg:
                if "quota" in error_msg or "exhausted" in error_msg:
                    logger.error("Daily quota exhausted. Please try again tomorrow.")
                    raise QuotaExhaustedError(str(e))
                logger.warning(f"Rate limit hit: {e}")
                raise RateLimitError(str(e))
            
            logger.error(f"Embedding API error: {e}")
            raise EmbeddingError(str(e))
    
    def fetch_pending_lecturers(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Fetch lecturers that need expertise embeddings.
        
        Criteria:
        - Has SINTA publications (sinta_publications != '[]')
        - No expertise vector yet (expertise_vector IS NULL)
        
        Args:
            limit: Maximum records to fetch
            
        Returns:
            List of lecturer records
        """
        try:
            # Fetch lecturers with publications but no expertise vector
            response = (
                self.supabase.table("lecturers")
                .select("id, name, dept, sinta_publications")
                .is_("expertise_vector", "null")
                .neq("sinta_publications", "[]")
                .limit(limit)
                .execute()
            )
            
            # Filter to ensure sufficient publications
            lecturers = []
            for record in response.data or []:
                pubs = record.get("sinta_publications", [])
                if isinstance(pubs, list) and len(pubs) >= self.min_publications:
                    lecturers.append(record)
            
            return lecturers
            
        except Exception as e:
            logger.error(f"Failed to fetch pending lecturers: {e}")
            return []
    
    def update_expertise_vector(
        self,
        lecturer_id: str,
        embedding: List[float]
    ) -> bool:
        """
        Update a lecturer's expertise vector in the database.
        
        Args:
            lecturer_id: UUID of the lecturer record
            embedding: Expertise embedding vector
            
        Returns:
            True if successful
        """
        try:
            # Convert to string format for pgvector
            embedding_str = f"[{','.join(map(str, embedding))}]"
            
            self.supabase.table("lecturers").update({
                "expertise_vector": embedding_str,
                "last_updated": "now()"
            }).eq("id", lecturer_id).execute()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update expertise vector for {lecturer_id}: {e}")
            return False
    
    def process_all(
        self,
        max_records: Optional[int] = None,
        show_progress: bool = True
    ) -> EmbeddingProgress:
        """
        Process all pending lecturers and generate expertise embeddings.
        
        Args:
            max_records: Maximum records to process (None for all)
            show_progress: Show tqdm progress bar
            
        Returns:
            EmbeddingProgress with statistics
        """
        # Fetch pending lecturers
        limit = max_records or 10000
        lecturers = self.fetch_pending_lecturers(limit)
        
        self.progress.total_pending = len(lecturers)
        logger.info(f"Found {len(lecturers)} lecturers needing expertise embeddings")
        
        if not lecturers:
            logger.info("No pending lecturers to process")
            return self.progress
        
        # Process each lecturer
        pbar = tqdm(
            lecturers,
            desc="Generating expertise embeddings",
            disable=not show_progress
        )
        
        for lecturer in pbar:
            lecturer_id = lecturer["id"]
            name = lecturer.get("name", "Unknown")
            dept = lecturer.get("dept", "Unknown Department")
            publications = lecturer.get("sinta_publications", [])
            
            pbar.set_postfix({"current": name[:25]})
            
            # Check minimum publications
            if len(publications) < self.min_publications:
                logger.debug(f"Skipping {name}: only {len(publications)} publications")
                self.progress.total_skipped += 1
                self.progress.total_processed += 1
                continue
            
            try:
                # Build expertise string
                expertise_text = self._build_expertise_string(name, dept, publications)
                
                if not expertise_text:
                    logger.warning(f"Empty expertise string for {name}")
                    self.progress.total_skipped += 1
                    self.progress.total_processed += 1
                    continue
                
                # Generate embedding
                embedding = self._embed_text(expertise_text)
                
                # Update database
                success = self.update_expertise_vector(lecturer_id, embedding)
                
                if success:
                    self.progress.total_success += 1
                else:
                    self.progress.total_failed += 1
                    self.progress.failed_ids.append(lecturer_id)
                
                self.progress.total_processed += 1
                
                # Small delay to avoid rate limits
                time.sleep(0.3)
                
            except QuotaExhaustedError:
                logger.error("Stopping due to quota exhaustion")
                break
                
            except RateLimitError as e:
                logger.error(f"Rate limit error for {name}: {e}")
                self.progress.total_failed += 1
                self.progress.failed_ids.append(lecturer_id)
                self.progress.total_processed += 1
                
            except EmbeddingError as e:
                logger.error(f"Embedding error for {name}: {e}")
                self.progress.total_failed += 1
                self.progress.failed_ids.append(lecturer_id)
                self.progress.total_processed += 1
        
        pbar.close()
        
        logger.info(
            f"Embedding complete: {self.progress.total_success} succeeded, "
            f"{self.progress.total_failed} failed, "
            f"{self.progress.total_skipped} skipped"
        )
        
        return self.progress
    
    def get_stats(self) -> Dict[str, int]:
        """Get SINTA embedding statistics from database."""
        try:
            response = self.supabase.rpc("get_sinta_stats").execute()
            if response.data:
                return response.data[0] if isinstance(response.data, list) else response.data
        except Exception as e:
            logger.warning(f"Could not fetch stats via RPC: {e}")
            # Fallback to direct queries
            try:
                total = self.supabase.table("lecturers").select("id", count="exact").execute()
                with_sinta = self.supabase.table("lecturers").select("id", count="exact").neq("sinta_publications", "[]").execute()
                with_vector = self.supabase.table("lecturers").select("id", count="exact").not_.is_("expertise_vector", "null").execute()
                
                return {
                    "total_lecturers": total.count or 0,
                    "with_publications": with_sinta.count or 0,
                    "with_expertise_vector": with_vector.count or 0,
                    "pending_vectorization": (with_sinta.count or 0) - (with_vector.count or 0)
                }
            except Exception:
                pass
        
        return {
            "total_lecturers": 0,
            "with_publications": 0,
            "with_expertise_vector": 0,
            "pending_vectorization": 0
        }
    
    def find_similar_lecturers(
        self,
        query_text: str,
        match_count: int = 10,
        match_threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Find lecturers with expertise matching query text.
        
        Args:
            query_text: Research topic or query
            match_count: Maximum results to return
            match_threshold: Minimum similarity score (0-1)
            
        Returns:
            List of matching lecturers with similarity scores
        """
        try:
            # Generate query embedding
            query_embedding = self._embed_text(query_text)
            
            # Convert to string format
            embedding_str = f"[{','.join(map(str, query_embedding))}]"
            
            # Call match_lecturers RPC
            response = self.supabase.rpc(
                "match_lecturers",
                {
                    "query_embedding": embedding_str,
                    "match_threshold": match_threshold,
                    "match_count": match_count
                }
            ).execute()
            
            return response.data or []
            
        except Exception as e:
            logger.error(f"Failed to find similar lecturers: {e}")
            return []


# ============================================================================
# Factory Function
# ============================================================================

def create_sinta_embedder(**kwargs) -> SINTAEmbedder:
    """Factory function to create SINTA embedder."""
    return SINTAEmbedder(**kwargs)


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    """CLI entry point for SINTA embedding."""
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
    
    parser = argparse.ArgumentParser(description="Generate expertise embeddings for SINTA lecturers")
    parser.add_argument("--limit", type=int, help="Maximum lecturers to process")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size")
    parser.add_argument("--stats", action="store_true", help="Show statistics only")
    parser.add_argument("--search", type=str, help="Find experts matching topic")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("\n🧬 SINTA Embedder - ITS Intelligence Hub v2\n")
    
    embedder = SINTAEmbedder(batch_size=args.batch_size)
    
    if args.stats:
        stats = embedder.get_stats()
        print("📊 SINTA Embedding Statistics:\n")
        print(f"   Total lecturers: {stats.get('total_lecturers', 0)}")
        print(f"   With SINTA publications: {stats.get('with_publications', 0)}")
        print(f"   With expertise vector: {stats.get('with_expertise_vector', 0)}")
        print(f"   Pending vectorization: {stats.get('pending_vectorization', 0)}")
        return
    
    if args.search:
        print(f"🔍 Searching for experts in: '{args.search}'...\n")
        experts = embedder.find_similar_lecturers(args.search, match_count=10)
        
        if experts:
            for i, exp in enumerate(experts, 1):
                print(f"   {i}. {exp['name']}")
                print(f"      Dept: {exp.get('dept', 'N/A')}")
                print(f"      H-Index: Scopus={exp.get('h_index_scopus', 0)}, GScholar={exp.get('h_index_gscholar', 0)}")
                print(f"      Similarity: {exp.get('similarity', 0):.3f}")
                
                # Fetch publications for this expert
                try:
                    lecturer_id = exp.get('id')
                    if lecturer_id:
                        pub_response = embedder.supabase.table("lecturers").select("sinta_publications").eq("id", lecturer_id).execute()
                        if pub_response.data and pub_response.data[0].get('sinta_publications'):
                            pubs = pub_response.data[0]['sinta_publications']
                            print(f"      Publications ({len(pubs)}):")
                            for pub in pubs[:5]:  # Show top 5
                                title = pub.get('title', 'N/A')[:70]
                                year = pub.get('year', 'N/A')
                                print(f"         • {title}... ({year})")
                except Exception:
                    pass
                print()
        else:
            print("   No matching experts found.")
        return
    
    print("🚀 Starting expertise embedding pipeline...\n")
    progress = embedder.process_all(max_records=args.limit)
    
    print(f"\n✅ Embedding Complete!")
    print(f"   Processed: {progress.total_processed}")
    print(f"   Succeeded: {progress.total_success}")
    print(f"   Failed: {progress.total_failed}")
    print(f"   Skipped (insufficient pubs): {progress.total_skipped}")


if __name__ == "__main__":
    main()
