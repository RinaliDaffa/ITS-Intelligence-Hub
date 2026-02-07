"""
ITS Intelligence Hub v2 - Metadata Enricher
Recovers missing supervisor data from PDF metadata and BibTeX exports.
"""

import os
import re
import io
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from urllib.parse import urlparse
import tempfile

import requests
from tqdm import tqdm
from dotenv import load_dotenv
from supabase import create_client, Client

# PDF extraction
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    logging.warning("PyMuPDF not installed. PDF extraction disabled.")

# BibTeX parsing
try:
    import bibtexparser
    HAS_BIBTEX = True
except ImportError:
    HAS_BIBTEX = False
    logging.warning("bibtexparser not installed. BibTeX parsing disabled.")

load_dotenv()
logger = logging.getLogger(__name__)

# Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
REQUEST_DELAY = float(os.getenv("REQUEST_DELAY_MIN", "2"))

# Repository URL patterns
BASE_URL = "https://repository.its.ac.id"

# Supervisor extraction patterns for PDF text
PDF_SUPERVISOR_PATTERNS = [
    # Indonesian patterns - look for "Dosen Pembimbing" followed by name
    r"(?i)dosen\s+pembimbing\s*(?:1|I|utama)?\s*[:\n\r]\s*([A-Z][^\n\r]{10,60})",
    r"(?i)dosen\s+pembimbing\s*(?:2|II|kedua)?\s*[:\n\r]\s*([A-Z][^\n\r]{10,60})",
    r"(?i)pembimbing\s*(?:1|I|utama)?\s*[:\n\r]\s*([A-Z][^\n\r]{10,60})",
    r"(?i)pembimbing\s*(?:2|II|kedua)?\s*[:\n\r]\s*([A-Z][^\n\r]{10,60})",
    
    # Look for NIP pattern (Indonesian civil servant ID) which often precedes supervisor
    r"(?i)((?:Dr|Prof|Ir)\.[^\n\r]{5,50})\s*NIP[:\.]?\s*\d{10,}",
    r"(?i)NIP[:\.]?\s*\d{10,}\s*[)\n\r]\s*([A-Z][^\n\r]{5,50})",
    
    # Look for approval signatures
    r"(?i)(?:Disetujui|Approved)\s+oleh[:\n\r]\s*([A-Z][^\n\r]{10,60})",
    r"(?i)(?:Mengetahui|Menyetujui)[:\n\r]\s*(?:Dosen\s+Pembimbing)?[:\n\r]\s*([A-Z][^\n\r]{10,60})",
    
    # English patterns
    r"(?i)supervisor[:\n\r]\s*([A-Z][^\n\r]{10,60})",
    r"(?i)thesis\s+advisor[:\n\r]\s*([A-Z][^\n\r]{10,60})",
    r"(?i)approved\s+by[:\n\r]\s*([A-Z][^\n\r]{10,60})",
]


@dataclass  
class EnricherProgress:
    """Track enrichment progress."""
    total_records: int = 0
    total_checked: int = 0
    bibtex_success: int = 0
    pdf_success: int = 0
    total_failed: int = 0
    already_has_supervisors: int = 0


def extract_eprint_id(url: str) -> Optional[str]:
    """Extract eprint ID from repository URL."""
    # Pattern: repository.its.ac.id/123456/ or http://repository.its.ac.id/123456/
    match = re.search(r"repository\.its\.ac\.id/(\d+)/?", url)
    if match:
        return match.group(1)
    return None


def clean_supervisor_name(name: str) -> str:
    """Clean and validate supervisor name."""
    # Basic cleaning
    name = " ".join(name.split())
    
    # Remove NIP and other ID patterns
    name = re.sub(r'\(?NIP[:\.]?\s*[\d\s.\-]+\)?', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\(?NIK[:\.]?\s*[\d\s.\-]+\)?', '', name, flags=re.IGNORECASE)
    
    # Remove role keywords
    name = re.sub(r'\b(?:pembimbing|supervisor|advisor|dosen|ketua|anggota)\b', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\b(?:utama|pertama|kedua|pendamping|I|II)\b', '', name, flags=re.IGNORECASE)
    
    # Remove trailing numbers, punctuation
    name = re.sub(r'^[\s\-:.,;()]+|[\s\-:.,;()0-9]+$', '', name)
    
    # Must have reasonable length
    name = name.strip()
    if len(name) < 5 or len(name) > 80:
        return ""
    
    # Must start with uppercase or academic title
    if not (name[0].isupper() or name.startswith(('Dr', 'Prof', 'Ir'))):
        return ""
    
    return name


class MetadataEnricher:
    """
    Enriches research records with missing supervisor data.
    Uses BibTeX exports and PDF extraction as fallback.
    """
    
    def __init__(self) -> None:
        """Initialize enricher."""
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError("Supabase credentials required.")
        
        self.supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0"
        })
        self.progress = EnricherProgress()
        
        logger.info(f"Initialized MetadataEnricher (PyMuPDF: {HAS_PYMUPDF}, bibtexparser: {HAS_BIBTEX})")
    
    def fetch_records_needing_enrichment(self, limit: int = 500) -> List[Dict[str, Any]]:
        """
        Fetch records with empty or NULL supervisors.
        
        Args:
            limit: Maximum records to fetch
            
        Returns:
            List of research records
        """
        try:
            # Fetch records - we'll filter for empty arrays in Python
            response = (
                self.supabase.table("researches")
                .select("id, url, title, supervisors, metadata")
                .limit(limit)
                .execute()
            )
            
            # Filter to only records with empty or null supervisors
            records = []
            for r in response.data or []:
                supervisors = r.get("supervisors")
                if not supervisors or (isinstance(supervisors, list) and len(supervisors) == 0):
                    records.append(r)
                else:
                    self.progress.already_has_supervisors += 1
            
            self.progress.total_records = len(records)
            return records
            
        except Exception as e:
            logger.error(f"Failed to fetch records: {e}")
            return []
    
    def try_bibtex_extraction(self, eprint_id: str) -> List[str]:
        """
        Try to extract supervisors from BibTeX export.
        
        Args:
            eprint_id: EPrint ID number
            
        Returns:
            List of supervisor names found
        """
        if not HAS_BIBTEX:
            return []
        
        try:
            bib_url = f"{BASE_URL}/cgi/export/eprint/{eprint_id}/BibTeX/its-eprint-{eprint_id}.bib"
            
            response = self.session.get(bib_url, timeout=15)
            if response.status_code != 200:
                return []
            
            # Parse BibTeX
            bib_db = bibtexparser.loads(response.text)
            
            supervisors = []
            for entry in bib_db.entries:
                # Look for editor, contributor, or advisor fields
                for field in ['editor', 'contributor', 'advisor', 'supervisor']:
                    if field in entry:
                        names = entry[field].split(' and ')
                        for name in names:
                            cleaned = clean_supervisor_name(name)
                            if cleaned and cleaned not in supervisors:
                                supervisors.append(cleaned)
            
            return supervisors[:2]  # Max 2 supervisors
            
        except Exception as e:
            logger.debug(f"BibTeX extraction failed for {eprint_id}: {e}")
            return []
    
    def try_pdf_extraction(self, eprint_id: str) -> List[str]:
        """
        Try to extract supervisors from PDF first pages.
        
        Args:
            eprint_id: EPrint ID number
            
        Returns:
            List of supervisor names found
        """
        if not HAS_PYMUPDF:
            return []
        
        try:
            # Common PDF URL pattern
            pdf_url = f"{BASE_URL}/{eprint_id}/1/{eprint_id}-Undergraduate_Thesis.pdf"
            
            # Try alternative patterns
            pdf_urls = [
                pdf_url,
                f"{BASE_URL}/{eprint_id}/1/paper.pdf",
                f"{BASE_URL}/{eprint_id}/1/thesis.pdf",
            ]
            
            for url in pdf_urls:
                try:
                    response = self.session.get(url, timeout=30, stream=True)
                    if response.status_code == 200 and 'pdf' in response.headers.get('content-type', '').lower():
                        # Read first 500KB only (covers first few pages)
                        pdf_bytes = response.raw.read(500 * 1024)
                        return self._extract_from_pdf_bytes(pdf_bytes)
                except:
                    continue
            
            return []
            
        except Exception as e:
            logger.debug(f"PDF extraction failed for {eprint_id}: {e}")
            return []
    
    def _extract_from_pdf_bytes(self, pdf_bytes: bytes) -> List[str]:
        """Extract supervisor names from PDF bytes."""
        try:
            # Open PDF from bytes
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            
            # Extract text from first 5 pages (where approval/title pages usually are)
            text = ""
            for page_num in range(min(5, len(doc))):
                page = doc[page_num]
                text += page.get_text() + "\n"
            
            doc.close()
            
            # Search for supervisor patterns
            supervisors = []
            for pattern in PDF_SUPERVISOR_PATTERNS:
                try:
                    matches = re.findall(pattern, text)
                    for match in matches:
                        cleaned = clean_supervisor_name(match)
                        if cleaned and cleaned not in supervisors:
                            supervisors.append(cleaned)
                except:
                    continue
            
            return supervisors[:2]  # Max 2 supervisors
            
        except Exception as e:
            logger.debug(f"PDF text extraction error: {e}")
            return []
    
    def update_supervisors(self, record_id: str, supervisors: List[str]) -> bool:
        """
        Update supervisors for a record.
        
        Args:
            record_id: UUID of the research record
            supervisors: List of supervisor names
            
        Returns:
            True if successful
        """
        try:
            self.supabase.table("researches").update({
                "supervisors": supervisors
            }).eq("id", record_id).execute()
            return True
        except Exception as e:
            logger.error(f"Failed to update record {record_id}: {e}")
            return False
    
    def enrich_all(
        self,
        limit: Optional[int] = None,
        dry_run: bool = False,
        show_progress: bool = True
    ) -> EnricherProgress:
        """
        Enrich all records missing supervisor data.
        
        Args:
            limit: Maximum records to process
            dry_run: If True, don't update database
            show_progress: Show progress bar
            
        Returns:
            EnricherProgress with statistics
        """
        logger.info("Starting metadata enrichment...")
        
        # Fetch records needing enrichment
        records = self.fetch_records_needing_enrichment(limit or 1000)
        
        if not records:
            logger.info("No records needing enrichment")
            return self.progress
        
        logger.info(f"Found {len(records)} records to process")
        
        pbar = tqdm(records, desc="Enriching metadata", disable=not show_progress)
        
        for record in pbar:
            self.progress.total_checked += 1
            
            url = record.get("url", "")
            eprint_id = extract_eprint_id(url)
            
            if not eprint_id:
                self.progress.total_failed += 1
                continue
            
            supervisors = []
            source = None
            
            # Strategy 1: Try BibTeX export
            supervisors = self.try_bibtex_extraction(eprint_id)
            if supervisors:
                source = "bibtex"
                self.progress.bibtex_success += 1
            else:
                # Strategy 2: Try PDF extraction
                supervisors = self.try_pdf_extraction(eprint_id)
                if supervisors:
                    source = "pdf"
                    self.progress.pdf_success += 1
            
            if supervisors:
                if not dry_run:
                    self.update_supervisors(record["id"], supervisors)
                
                logger.debug(f"Found supervisors for {eprint_id} ({source}): {supervisors}")
                pbar.set_postfix({
                    "bibtex": self.progress.bibtex_success,
                    "pdf": self.progress.pdf_success
                })
            else:
                self.progress.total_failed += 1
            
            # Rate limiting
            time.sleep(REQUEST_DELAY)
        
        pbar.close()
        
        logger.info(f"Enrichment complete: BibTeX={self.progress.bibtex_success}, "
                   f"PDF={self.progress.pdf_success}, Failed={self.progress.total_failed}")
        
        return self.progress


def create_enricher() -> MetadataEnricher:
    """Factory function to create enricher."""
    return MetadataEnricher()


def main():
    """CLI entry point."""
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
    
    parser = argparse.ArgumentParser(description="Enrich research metadata")
    parser.add_argument("--limit", type=int, help="Maximum records to process")
    parser.add_argument("--dry-run", action="store_true", help="Don't update database")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    enricher = MetadataEnricher()
    
    print("\n🔍 Starting Metadata Enrichment Pipeline...\n")
    
    progress = enricher.enrich_all(limit=args.limit, dry_run=args.dry_run)
    
    print(f"\n✅ Enrichment Complete!")
    print(f"   Records checked: {progress.total_checked}")
    print(f"   Already had supervisors: {progress.already_has_supervisors}")
    print(f"   BibTeX success: {progress.bibtex_success}")
    print(f"   PDF success: {progress.pdf_success}")
    print(f"   Failed/No data: {progress.total_failed}")


if __name__ == "__main__":
    main()
