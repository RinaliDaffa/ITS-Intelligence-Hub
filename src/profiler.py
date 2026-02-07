"""
ITS Intelligence Hub v2 - Expert Profiler Module
Calculates Expertise DNA (centroid vectors) for lecturers/supervisors.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import numpy as np

from tqdm import tqdm
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()
logger = logging.getLogger(__name__)

# Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "768"))


@dataclass
class ProfilerProgress:
    """Track profiling progress."""
    total_supervisors: int = 0
    total_processed: int = 0
    total_success: int = 0
    total_failed: int = 0
    total_research_linked: int = 0


class ExpertProfiler:
    """
    Calculates Expertise DNA for supervisors.
    
    Expertise Vector = Centroid (average) of all supervised research embeddings:
    v_lecturer = (1/N) * Σ v_research,i
    """
    
    def __init__(self) -> None:
        """Initialize profiler with Supabase connection."""
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError("Supabase credentials required.")
        
        self.supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        self.progress = ProfilerProgress()
        
        logger.info("Initialized Expert Profiler")
    
    def fetch_supervisor_embeddings(self) -> Dict[str, List[Tuple[str, List[float]]]]:
        """
        Fetch all supervisors with their research embeddings.
        
        Returns:
            Dict mapping supervisor name -> list of (research_id, embedding, dept) tuples
        """
        supervisor_data: Dict[str, List[Tuple[str, List[float]]]] = defaultdict(list)
        
        try:
            # Fetch researches with embeddings and supervisors
            # Process in batches to handle large datasets
            offset = 0
            batch_size = 500
            records_with_supervisors = 0
            records_checked = 0
            
            while True:
                # Note: We fetch all records with embeddings, then filter for non-empty supervisors
                # because Supabase doesn't have a direct array_length filter via REST API
                response = (
                    self.supabase.table("researches")
                    .select("id, supervisors, embedding, dept")
                    .not_.is_("embedding", "null")
                    .range(offset, offset + batch_size - 1)
                    .execute()
                )
                
                if not response.data:
                    break
                
                for record in response.data:
                    records_checked += 1
                    supervisors = record.get("supervisors", [])
                    embedding = record.get("embedding")
                    research_id = record.get("id")
                    dept = record.get("dept", "")
                    
                    # Skip if supervisors is NULL, empty, or empty array
                    if not supervisors or not isinstance(supervisors, list) or len(supervisors) == 0:
                        continue
                    
                    if not embedding:
                        continue
                    
                    records_with_supervisors += 1
                    
                    # Parse embedding if string
                    if isinstance(embedding, str):
                        # Remove brackets and split
                        embedding = [float(x) for x in embedding.strip("[]").split(",")]
                    
                    # Link each supervisor to this research
                    for supervisor in supervisors:
                        if supervisor and isinstance(supervisor, str) and len(supervisor.strip()) > 2:
                            supervisor_name = self._normalize_name(supervisor)
                            supervisor_data[supervisor_name].append(
                                (research_id, embedding, dept)
                            )
                
                offset += batch_size
                
                if len(response.data) < batch_size:
                    break
            
            logger.info(f"Checked {records_checked} records, {records_with_supervisors} have supervisors")
            logger.info(f"Found {len(supervisor_data)} unique supervisors")
            return supervisor_data
            
        except Exception as e:
            logger.error(f"Failed to fetch supervisor embeddings: {e}")
            return {}
    
    def _normalize_name(self, name: str) -> str:
        """
        Normalize supervisor name for deduplication.
        
        Args:
            name: Raw supervisor name
            
        Returns:
            Normalized name
        """
        # Clean whitespace
        name = " ".join(name.split())
        
        # Remove common suffixes/prefixes that vary
        name = name.strip("., ")
        
        return name
    
    def _calculate_centroid(self, embeddings: List[List[float]]) -> List[float]:
        """
        Calculate centroid (mean) of embedding vectors.
        
        Args:
            embeddings: List of embedding vectors
            
        Returns:
            Centroid vector
        """
        if not embeddings:
            return [0.0] * EMBEDDING_DIMENSION
        
        # Convert to numpy for efficient calculation
        arr = np.array(embeddings, dtype=np.float32)
        centroid = np.mean(arr, axis=0)
        
        # Normalize to unit vector for cosine similarity
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        
        return centroid.tolist()
    
    def _get_most_common_dept(self, entries: List[Tuple]) -> str:
        """Get most common department from research entries."""
        dept_counts: Dict[str, int] = defaultdict(int)
        for entry in entries:
            if len(entry) >= 3 and entry[2]:
                dept_counts[entry[2]] += 1
        
        if dept_counts:
            return max(dept_counts.keys(), key=lambda k: dept_counts[k])
        return ""
    
    def upsert_lecturer(
        self,
        name: str,
        dept: str,
        research_count: int,
        expertise_vector: List[float]
    ) -> bool:
        """
        Upsert a lecturer record.
        
        Args:
            name: Lecturer name (unique)
            dept: Department
            research_count: Number of supervised researches
            expertise_vector: Centroid expertise embedding
            
        Returns:
            True if successful
        """
        try:
            # Convert vector to string format for pgvector
            vector_str = f"[{','.join(map(str, expertise_vector))}]"
            
            self.supabase.table("lecturers").upsert({
                "name": name,
                "dept": dept,
                "research_count": research_count,
                "expertise_vector": vector_str,
                "last_updated": "now()"
            }, on_conflict="name").execute()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to upsert lecturer {name}: {e}")
            return False
    
    def process_all(self, show_progress: bool = True) -> ProfilerProgress:
        """
        Calculate expertise vectors for all supervisors.
        
        Args:
            show_progress: Show tqdm progress bar
            
        Returns:
            ProfilerProgress with statistics
        """
        logger.info("Starting expertise profiling...")
        
        # Fetch all supervisor -> research mappings
        supervisor_data = self.fetch_supervisor_embeddings()
        
        if not supervisor_data:
            logger.warning("No supervisor data found")
            return self.progress
        
        self.progress.total_supervisors = len(supervisor_data)
        
        # Process each supervisor
        pbar = tqdm(
            supervisor_data.items(),
            desc="Building expertise profiles",
            disable=not show_progress
        )
        
        for supervisor_name, entries in pbar:
            self.progress.total_processed += 1
            
            # Extract embeddings
            embeddings = [entry[1] for entry in entries if entry[1]]
            
            if not embeddings:
                self.progress.total_failed += 1
                continue
            
            # Calculate centroid
            expertise_vector = self._calculate_centroid(embeddings)
            
            # Get most common department
            dept = self._get_most_common_dept(entries)
            
            # Upsert to database
            success = self.upsert_lecturer(
                name=supervisor_name,
                dept=dept,
                research_count=len(entries),
                expertise_vector=expertise_vector
            )
            
            if success:
                self.progress.total_success += 1
                self.progress.total_research_linked += len(entries)
            else:
                self.progress.total_failed += 1
            
            pbar.set_postfix({
                "success": self.progress.total_success,
                "researches": self.progress.total_research_linked
            })
        
        pbar.close()
        
        logger.info(
            f"Profiling complete: {self.progress.total_success} lecturers, "
            f"{self.progress.total_research_linked} research links"
        )
        
        return self.progress
    
    def get_top_lecturers(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get top lecturers by research count.
        
        Args:
            limit: Number of lecturers to return
            
        Returns:
            List of lecturer records
        """
        try:
            response = (
                self.supabase.table("lecturers")
                .select("name, dept, research_count")
                .order("research_count", desc=True)
                .limit(limit)
                .execute()
            )
            return response.data or []
        except Exception as e:
            logger.error(f"Failed to fetch top lecturers: {e}")
            return []
    
    def find_similar_experts(
        self,
        query_text: str,
        match_count: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find lecturers with expertise matching query.
        
        Requires embedder to generate query embedding first.
        
        Args:
            query_text: Research topic/query
            match_count: Number of matches to return
            
        Returns:
            List of matching lecturers with similarity scores
        """
        from .embedder import GeminiEmbedder
        
        try:
            # Generate query embedding
            embedder = GeminiEmbedder()
            query_embedding = embedder.embed_single(query_text)
            
            # Convert to string format
            embedding_str = f"[{','.join(map(str, query_embedding))}]"
            
            # Call match_lecturers RPC
            response = self.supabase.rpc(
                "match_lecturers",
                {
                    "query_embedding": embedding_str,
                    "match_threshold": 0.3,
                    "match_count": match_count
                }
            ).execute()
            
            return response.data or []
            
        except Exception as e:
            logger.error(f"Failed to find similar experts: {e}")
            return []


def create_profiler() -> ExpertProfiler:
    """Factory function to create profiler."""
    return ExpertProfiler()


def main():
    """CLI entry point for profiling."""
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
    
    parser = argparse.ArgumentParser(description="Build lecturer expertise profiles")
    parser.add_argument("--top", type=int, help="Show top N lecturers by research count")
    parser.add_argument("--search", type=str, help="Find experts matching topic")
    
    args = parser.parse_args()
    
    profiler = ExpertProfiler()
    
    if args.top:
        lecturers = profiler.get_top_lecturers(args.top)
        print(f"\n🎓 Top {args.top} Lecturers by Research Count:\n")
        for i, lec in enumerate(lecturers, 1):
            print(f"   {i}. {lec['name']} ({lec['dept']}) - {lec['research_count']} papers")
        return
    
    if args.search:
        print(f"\n🔍 Searching for experts in: '{args.search}'...\n")
        experts = profiler.find_similar_experts(args.search, match_count=5)
        for i, exp in enumerate(experts, 1):
            print(f"   {i}. {exp['name']} ({exp['dept']}) - similarity: {exp['similarity']:.3f}")
        return
    
    print("\n🧬 Starting expertise profiling pipeline...\n")
    progress = profiler.process_all()
    
    print(f"\n✅ Profiling Complete!")
    print(f"   Lecturers profiled: {progress.total_success}")
    print(f"   Research links: {progress.total_research_linked}")
    print(f"   Failed: {progress.total_failed}")


if __name__ == "__main__":
    main()
