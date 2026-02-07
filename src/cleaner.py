"""
ITS Intelligence Hub v2 - Data Cleaner Module
Handles text normalization, boilerplate removal, and quality scoring.
"""

import re
import html
import unicodedata
from typing import Optional, Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)


# Academic boilerplate patterns to remove (Indonesian & English)
BOILERPLATE_PATTERNS = [
    # Indonesian patterns
    r"(?i)diajukan\s+(?:untuk|sebagai|guna)\s+memenuhi.*?(?:sarjana|magister|doktor|S1|S2|S3)",
    r"(?i)tugas\s+akhir\s+ini\s+diajukan.*?(?:persyaratan|syarat)",
    r"(?i)copyright\s*©?\s*(?:ITS|Institut Teknologi Sepuluh Nopember).*?\d{4}",
    r"(?i)hak\s+cipta\s+(?:dilindungi|milik).*?(?:undang-undang|ITS)",
    r"(?i)naskah\s+(?:skripsi|tesis|disertasi)\s+ini\s+(?:telah|sudah)\s+(?:disetujui|diperiksa)",
    r"(?i)program\s+studi\s+(?:sarjana|magister|doktor).*?(?:departemen|jurusan|fakultas)",
    
    # English patterns
    r"(?i)this\s+(?:thesis|dissertation)\s+(?:is\s+)?submitted\s+(?:to|in\s+partial)",
    r"(?i)in\s+partial\s+fulfillment\s+of\s+the\s+requirements",
    r"(?i)copyright\s*©?\s*\d{4}.*?(?:all\s+rights\s+reserved)",
    r"(?i)approved\s+by\s+(?:the\s+)?(?:examining|thesis)\s+committee",
]

# Supervisor extraction patterns
SUPERVISOR_PATTERNS = [
    r"(?i)(?:pembimbing|dosen\s+pembimbing)\s*(?:1|I|utama)?\s*:\s*([^\n\r,;]+)",
    r"(?i)(?:pembimbing|dosen\s+pembimbing)\s*(?:2|II|pendamping)?\s*:\s*([^\n\r,;]+)",
    r"(?i)supervisor\s*(?:1|I)?\s*:\s*([^\n\r,;]+)",
    r"(?i)supervisor\s*(?:2|II)?\s*:\s*([^\n\r,;]+)",
    r"(?i)advisor\s*(?:1|I)?\s*:\s*([^\n\r,;]+)",
    r"(?i)advisor\s*(?:2|II)?\s*:\s*([^\n\r,;]+)",
    r"(?i)co-?(?:advisor|supervisor)\s*:\s*([^\n\r,;]+)",
]

# Academic title patterns to preserve in supervisor names
ACADEMIC_TITLES = r"(?:Dr\.?|Prof\.?|Ir\.?|S\.T\.?|M\.T\.?|M\.Sc\.?|Ph\.?D\.?|S\.Kom\.?|M\.Kom\.?)"


class DataCleaner:
    """Handles cleaning, normalization, and validation of scraped research data."""
    
    def __init__(self) -> None:
        """Initialize cleaner with compiled regex patterns."""
        self.boilerplate_regexes = [re.compile(p, re.MULTILINE | re.DOTALL) for p in BOILERPLATE_PATTERNS]
        self.supervisor_regexes = [re.compile(p) for p in SUPERVISOR_PATTERNS]
        
    def clean_text(self, text: Optional[str]) -> str:
        """
        Full text cleaning pipeline.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned and normalized text
        """
        if not text:
            return ""
            
        # Step 1: Decode HTML entities
        text = html.unescape(text)
        
        # Step 2: Strip HTML tags
        text = self._strip_html_tags(text)
        
        # Step 3: Fix encoding issues
        text = self._fix_encoding(text)
        
        # Step 4: Normalize unicode
        text = unicodedata.normalize('NFKC', text)
        
        # Step 5: Normalize whitespace
        text = self._normalize_whitespace(text)
        
        return text.strip()
    
    def clean_abstract(self, abstract: Optional[str]) -> str:
        """
        Clean abstract with boilerplate removal.
        
        Args:
            abstract: Raw abstract text
            
        Returns:
            Cleaned abstract without academic boilerplate
        """
        if not abstract:
            return ""
            
        # Basic cleaning first
        text = self.clean_text(abstract)
        
        # Remove boilerplate patterns
        text = self._remove_boilerplate(text)
        
        # Remove common prefixes
        text = re.sub(r"^(?:abstract|abstrak)\s*:?\s*", "", text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def extract_supervisors(self, text: str, limit_chars: int = 500) -> List[str]:
        """
        Extract supervisor names from text using heuristic patterns.
        
        Args:
            text: Text to search (typically first 500 chars of abstract/intro)
            limit_chars: Maximum characters to search
            
        Returns:
            List of supervisor names found
        """
        supervisors = []
        search_text = text[:limit_chars] if text else ""
        
        for pattern in self.supervisor_regexes:
            matches = pattern.findall(search_text)
            for match in matches:
                cleaned_name = self._clean_supervisor_name(match)
                if cleaned_name and cleaned_name not in supervisors:
                    supervisors.append(cleaned_name)
        
        # Deduplicate while preserving order
        seen = set()
        unique_supervisors = []
        for s in supervisors:
            normalized = s.lower().strip()
            if normalized not in seen and len(s) > 3:
                seen.add(normalized)
                unique_supervisors.append(s)
        
        return unique_supervisors[:2]  # Max 2 supervisors
    
    def calculate_confidence(
        self,
        title: Optional[str],
        abstract: Optional[str],
        author: Optional[str],
        supervisors: Optional[List[str]],
        year: Optional[int],
        dept: Optional[str]
    ) -> float:
        """
        Calculate data_confidence score (0.0 to 1.0).
        
        Formula: 0.3×Length + 0.2×Supervisors + 0.3×Metadata + 0.2×Encoding
        
        Args:
            title: Research title
            abstract: Abstract text
            author: Author name
            supervisors: List of supervisors
            year: Publication year
            dept: Department name
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        score = 0.0
        
        # Length component (0.3): abstract completeness
        if abstract:
            abstract_len = len(abstract)
            if abstract_len >= 500:
                score += 0.3
            elif abstract_len >= 200:
                score += 0.2
            elif abstract_len >= 100:
                score += 0.1
        
        # Supervisors component (0.2): has supervisor info
        if supervisors and len(supervisors) > 0:
            score += 0.1
            if len(supervisors) >= 2:
                score += 0.1
        
        # Metadata completeness (0.3)
        metadata_fields = [title, author, year, dept]
        filled_count = sum(1 for f in metadata_fields if f)
        score += 0.3 * (filled_count / len(metadata_fields))
        
        # Encoding integrity (0.2): no suspicious characters
        all_text = f"{title or ''} {abstract or ''} {author or ''}"
        encoding_score = self._check_encoding_integrity(all_text)
        score += 0.2 * encoding_score
        
        return round(min(1.0, max(0.0, score)), 2)
    
    def clean_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean all fields in a research record.
        
        Args:
            record: Raw record with title, abstract, author, etc.
            
        Returns:
            Cleaned record with confidence score
        """
        cleaned = {}
        
        # Clean text fields
        cleaned['title'] = self.clean_text(record.get('title'))
        cleaned['abstract'] = self.clean_abstract(record.get('abstract'))
        cleaned['author'] = self.clean_text(record.get('author'))
        cleaned['dept'] = self.clean_text(record.get('dept'))
        cleaned['degree'] = self._normalize_degree(record.get('degree'))
        cleaned['url'] = record.get('url', '').strip()
        
        # Extract year
        year = record.get('year')
        if isinstance(year, str):
            year_match = re.search(r'\d{4}', year)
            year = int(year_match.group()) if year_match else None
        cleaned['year'] = year
        
        # Extract supervisors if not provided
        supervisors = record.get('supervisors', [])
        if not supervisors and cleaned['abstract']:
            supervisors = self.extract_supervisors(cleaned['abstract'])
        cleaned['supervisors'] = supervisors if supervisors else None
        
        # Calculate confidence
        cleaned['data_confidence'] = self.calculate_confidence(
            cleaned['title'],
            cleaned['abstract'],
            cleaned['author'],
            cleaned['supervisors'],
            cleaned['year'],
            cleaned['dept']
        )
        
        # Preserve metadata
        cleaned['metadata'] = record.get('metadata', {})
        
        return cleaned
    
    def _strip_html_tags(self, text: str) -> str:
        """Remove HTML tags from text."""
        # Remove script and style content
        text = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', text, flags=re.DOTALL | re.IGNORECASE)
        # Remove all remaining tags
        text = re.sub(r'<[^>]+>', ' ', text)
        return text
    
    def _fix_encoding(self, text: str) -> str:
        """Fix common encoding issues."""
        # Common mojibake fixes
        replacements = {
            'â€™': "'",
            'â€œ': '"',
            'â€': '"',
            'â€"': '-',
            'â€"': '—',
            'Ã©': 'é',
            'Ã¨': 'è',
            'Ã ': 'à',
            '\ufffd': '',  # Replacement character
            '\x00': '',    # Null bytes
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
            
        # Remove control characters except newlines and tabs
        text = ''.join(c for c in text if unicodedata.category(c) != 'Cc' or c in '\n\t\r')
        
        return text
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace: collapse multiple spaces, clean newlines."""
        # Replace various whitespace with single space
        text = re.sub(r'[ \t]+', ' ', text)
        # Collapse multiple newlines to max 2
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Remove leading/trailing whitespace from lines
        text = '\n'.join(line.strip() for line in text.split('\n'))
        return text
    
    def _remove_boilerplate(self, text: str) -> str:
        """Remove academic boilerplate patterns."""
        for regex in self.boilerplate_regexes:
            text = regex.sub('', text)
        return text
    
    def _clean_supervisor_name(self, name: str) -> str:
        """Clean and normalize supervisor name."""
        name = self.clean_text(name)
        # Remove common noise
        name = re.sub(r'\(?(?:NIP|NRP|NIK)\s*:?\s*[\d\s.-]+\)?', '', name, flags=re.IGNORECASE)
        name = re.sub(r'\b(?:pembimbing|supervisor|advisor)\b', '', name, flags=re.IGNORECASE)
        # Trim excessive punctuation
        name = re.sub(r'^[\s\-:.,;]+|[\s\-:.,;]+$', '', name)
        return name.strip()
    
    def _normalize_degree(self, degree: Optional[str]) -> str:
        """Normalize degree to S1/S2/S3 format."""
        if not degree:
            return ""
        degree_lower = degree.lower().strip()
        
        # Map various formats to S1/S2/S3
        mappings = {
            's1': 'S1', 'undergraduate': 'S1', 'sarjana': 'S1', 'bachelor': 'S1',
            's2': 'S2', 'master': 'S2', 'magister': 'S2',
            's3': 'S3', 'doctoral': 'S3', 'doktor': 'S3', 'phd': 'S3', 'dissertation': 'S3',
            'd3': 'D3', 'diploma': 'D3',
            'd4': 'D4',
        }
        
        for key, value in mappings.items():
            if key in degree_lower:
                return value
        
        return degree.upper()[:5]  # Fallback: truncate
    
    def _check_encoding_integrity(self, text: str) -> float:
        """Check text for encoding issues. Returns 0.0-1.0 score."""
        if not text:
            return 1.0
            
        issues = 0
        total_chars = len(text)
        
        if total_chars == 0:
            return 1.0
        
        # Count suspicious characters
        for char in text:
            if ord(char) == 0xFFFD:  # Replacement character
                issues += 1
            elif unicodedata.category(char) == 'Co':  # Private use
                issues += 1
            elif unicodedata.category(char) == 'Cn':  # Unassigned
                issues += 1
        
        # Calculate integrity score
        issue_ratio = issues / total_chars
        return max(0.0, 1.0 - (issue_ratio * 10))  # Penalize heavily for issues


# Convenience functions for direct use
def clean_text(text: Optional[str]) -> str:
    """Clean text with default cleaner instance."""
    return DataCleaner().clean_text(text)


def clean_abstract(abstract: Optional[str]) -> str:
    """Clean abstract with default cleaner instance."""
    return DataCleaner().clean_abstract(abstract)


def clean_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Clean record with default cleaner instance."""
    return DataCleaner().clean_record(record)
