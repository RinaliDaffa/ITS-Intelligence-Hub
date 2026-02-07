"""
ITS Intelligence Hub v2 - Supervisor Recovery Script
Re-extracts supervisors from abstracts using aggressive patterns.
"""

import os
import re
import logging
from typing import List, Dict, Any, Optional
from collections import defaultdict

from tqdm import tqdm
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()
logger = logging.getLogger(__name__)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Aggressive supervisor extraction patterns
# These search the ENTIRE abstract, not just first 500 chars
SUPERVISOR_PATTERNS = [
    # Indonesian patterns - more flexible
    r"(?i)(?:dosen\s+)?pembimbing\s*(?:1|I|utama|pertama)?\s*[:\-]\s*([A-Z][^,\n\r]{5,60})",
    r"(?i)(?:dosen\s+)?pembimbing\s*(?:2|II|kedua|pendamping)?\s*[:\-]\s*([A-Z][^,\n\r]{5,60})",
    r"(?i)dibimbing\s+oleh\s*[:\-]?\s*([A-Z][^,\n\r]{5,60})",
    
    # English patterns
    r"(?i)supervisor\s*(?:1|I)?\s*[:\-]\s*([A-Z][^,\n\r]{5,60})",
    r"(?i)supervisor\s*(?:2|II)?\s*[:\-]\s*([A-Z][^,\n\r]{5,60})",
    r"(?i)(?:thesis\s+)?advisor\s*(?:1|I)?\s*[:\-]\s*([A-Z][^,\n\r]{5,60})",
    r"(?i)co-?(?:advisor|supervisor)\s*[:\-]\s*([A-Z][^,\n\r]{5,60})",
    r"(?i)supervised\s+by\s*[:\-]?\s*([A-Z][^,\n\r]{5,60})",
    r"(?i)under\s+(?:the\s+)?(?:supervision|guidance)\s+of\s*[:\-]?\s*([A-Z][^,\n\r]{5,60})",
    
    # Academic title patterns - look for Dr., Prof., Ir. followed by name
    r"(?i)(?:pembimbing|supervisor|advisor)[^:]*:\s*((?:Dr|Prof|Ir)\.\s*[A-Z][a-zA-Z\s,\.]{5,50})",
]

# Patterns to detect academic names (Indonesian style)
NAME_VALIDATORS = [
    r"(?:Dr|Prof|Ir|S\.T|M\.T|M\.Sc|Ph\.D|S\.Kom|M\.Kom)[\.,]?\s*[A-Z]",  # Has title
    r"^[A-Z][a-z]+\s+[A-Z][a-z]+",  # Two capitalized words
]


def clean_supervisor_name(name: str) -> str:
    """Clean and validate supervisor name."""
    # Basic cleaning
    name = " ".join(name.split())
    
    # Remove common noise
    name = re.sub(r'\(?(?:NIP|NRP|NIK)\s*:?\s*[\d\s.\-]+\)?', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\b(?:pembimbing|supervisor|advisor|dosen)\b', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\b(?:utama|pertama|kedua|pendamping)\b', '', name, flags=re.IGNORECASE)
    
    # Remove trailing numbers, punctuation
    name = re.sub(r'^[\s\-:.,;]+|[\s\-:.,;0-9]+$', '', name)
    
    # Must have reasonable length
    name = name.strip()
    if len(name) < 5 or len(name) > 80:
        return ""
    
    # Must have at least one uppercase letter
    if not any(c.isupper() for c in name):
        return ""
    
    return name


def extract_supervisors_aggressive(text: str) -> List[str]:
    """
    Aggressively extract supervisor names from full text.
    
    Args:
        text: Full abstract/text to search
        
    Returns:
        List of supervisor names (max 2)
    """
    if not text:
        return []
    
    supervisors = []
    
    for pattern in SUPERVISOR_PATTERNS:
        try:
            matches = re.findall(pattern, text)
            for match in matches:
                cleaned = clean_supervisor_name(match)
                if cleaned and cleaned not in supervisors:
                    supervisors.append(cleaned)
        except re.error:
            continue
    
    # Deduplicate by normalized name
    seen = set()
    unique = []
    for s in supervisors:
        normalized = s.lower().strip()
        if normalized not in seen:
            seen.add(normalized)
            unique.append(s)
    
    return unique[:2]  # Max 2 supervisors


def run_recovery(limit: Optional[int] = None, dry_run: bool = False):
    """
    Run supervisor recovery on all records.
    
    Args:
        limit: Max records to process
        dry_run: If True, don't update database
    """
    client = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    print("\n🔧 Supervisor Recovery Script\n")
    
    # Fetch records with abstracts
    batch_size = 100
    offset = 0
    total_updated = 0
    total_found = 0
    
    while True:
        query = client.table("researches").select("id, title, abstract, supervisors")
        
        if limit:
            query = query.limit(min(batch_size, limit - offset))
        else:
            query = query.range(offset, offset + batch_size - 1)
        
        response = query.execute()
        
        if not response.data:
            break
        
        for record in tqdm(response.data, desc=f"Batch {offset // batch_size + 1}"):
            abstract = record.get("abstract", "")
            title = record.get("title", "")
            current_supervisors = record.get("supervisors", [])
            
            # Skip if already has supervisors
            if current_supervisors and len(current_supervisors) > 0:
                continue
            
            # Try to extract from abstract
            combined_text = f"{title}\n{abstract}"
            supervisors = extract_supervisors_aggressive(combined_text)
            
            if supervisors:
                total_found += 1
                
                if not dry_run:
                    try:
                        client.table("researches").update({
                            "supervisors": supervisors
                        }).eq("id", record["id"]).execute()
                        total_updated += 1
                    except Exception as e:
                        print(f"Error updating {record['id']}: {e}")
                else:
                    print(f"  Would update: {record['title'][:50]}...")
                    print(f"    Supervisors: {supervisors}")
        
        offset += batch_size
        
        if limit and offset >= limit:
            break
        
        if len(response.data) < batch_size:
            break
    
    print(f"\n✅ Recovery Complete!")
    print(f"   Found supervisors in: {total_found} records")
    print(f"   Updated: {total_updated} records")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Recover supervisor data from abstracts")
    parser.add_argument("--limit", type=int, help="Max records to process")
    parser.add_argument("--dry-run", action="store_true", help="Don't update database")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    run_recovery(limit=args.limit, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
