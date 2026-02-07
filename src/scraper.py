"""
ITS Intelligence Hub v2 - Scraper Module
Hybrid OAI-PMH + HTML enrichment approach for repository.its.ac.id
"""

import os
import re
import random
import time
import json
import logging
from typing import Dict, Any, List, Optional, Generator, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from urllib.parse import urljoin, urlparse, parse_qs
import xml.etree.ElementTree as ET

import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)
from dotenv import load_dotenv

from .cleaner import DataCleaner

load_dotenv()
logger = logging.getLogger(__name__)


# Configuration from environment
REQUEST_DELAY_MIN = float(os.getenv("REQUEST_DELAY_MIN", "2"))
REQUEST_DELAY_MAX = float(os.getenv("REQUEST_DELAY_MAX", "5"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "5"))

# Repository constants
BASE_URL = "https://repository.its.ac.id"
OAI_ENDPOINT = f"{BASE_URL}/cgi/oai2"

# OAI-PMH namespaces
OAI_NS = {
    "oai": "http://www.openarchives.org/OAI/2.0/",
    "dc": "http://purl.org/dc/elements/1.1/",
    "oai_dc": "http://www.openarchives.org/OAI/2.0/oai_dc/"
}

# Division URLs for Informatics/FTEIC (pilot scope)
# Structure: Division page -> Year pages (2024.html) -> Thesis URLs
FTEIC_DIVISIONS = {
    "Informatics_S1": f"{BASE_URL}/view/divisions/S1IF/",
    "Informatics_S2": f"{BASE_URL}/view/divisions/S2IF/",
    "Informatics_S3": f"{BASE_URL}/view/divisions/S3IF/",
    "Information_Systems_S1": f"{BASE_URL}/view/divisions/S1=5FSis=5FInf/",
    "Information_Systems_S2": f"{BASE_URL}/view/divisions/S2=5FSis=5FInf/",
    "Computer_Engineering_S1": f"{BASE_URL}/view/divisions/S1=5FSis=5FKomp/",
    "Electrical_Engineering_S1": f"{BASE_URL}/view/divisions/S1=5FTek=5FElk/",
    "Electrical_Engineering_S2": f"{BASE_URL}/view/divisions/S2=5FTek=5FElk/",
    "Electrical_Engineering_S3": f"{BASE_URL}/view/divisions/S3=5FTek=5FElk/",
}


class RetryableError(Exception):
    """Exception for errors that should trigger retry."""
    pass


class NonRetryableError(Exception):
    """Exception for errors that should not retry."""
    pass


@dataclass
class ScrapingProgress:
    """Track scraping progress for checkpointing."""
    total_discovered: int = 0
    total_processed: int = 0
    total_success: int = 0
    total_failed: int = 0
    last_oai_token: Optional[str] = None
    last_url: Optional[str] = None
    processed_urls: List[str] = field(default_factory=list)
    failed_urls: List[Dict[str, str]] = field(default_factory=list)
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data["updated_at"] = datetime.now().isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScrapingProgress":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class ITSScraper:
    """
    Hybrid scraper for ITS Repository.
    Uses OAI-PMH for metadata discovery and HTML scraping for enrichment.
    """
    
    def __init__(
        self,
        delay_min: float = REQUEST_DELAY_MIN,
        delay_max: float = REQUEST_DELAY_MAX,
        max_retries: int = MAX_RETRIES
    ) -> None:
        """
        Initialize scraper with configuration.
        
        Args:
            delay_min: Minimum delay between requests (seconds)
            delay_max: Maximum delay between requests (seconds)
            max_retries: Maximum retry attempts per request
        """
        self.delay_min = delay_min
        self.delay_max = delay_max
        self.max_retries = max_retries
        
        self.session = requests.Session()
        self.ua = UserAgent(fallback="Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0")
        self.cleaner = DataCleaner()
        self.progress = ScrapingProgress()
        
        # Request counter for logging
        self._request_count = 0
    
    def _get_headers(self) -> Dict[str, str]:
        """Generate request headers with rotating User-Agent."""
        return {
            "User-Agent": self.ua.random,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5,id;q=0.3",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "DNT": "1",
        }
    
    def _delay(self) -> None:
        """Apply randomized delay between requests."""
        delay = random.uniform(self.delay_min, self.delay_max)
        logger.debug(f"Sleeping for {delay:.2f} seconds")
        time.sleep(delay)
    
    @retry(
        retry=retry_if_exception_type(RetryableError),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(MAX_RETRIES),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    def _request(
        self,
        url: str,
        params: Optional[Dict[str, str]] = None,
        accept_xml: bool = False
    ) -> requests.Response:
        """
        Make HTTP request with retry logic.
        
        Args:
            url: URL to request
            params: Query parameters
            accept_xml: If True, accept XML responses
            
        Returns:
            Response object
            
        Raises:
            RetryableError: For 429/5xx errors
            NonRetryableError: For 4xx errors (except 429)
        """
        self._request_count += 1
        headers = self._get_headers()
        
        if accept_xml:
            headers["Accept"] = "application/xml,text/xml;q=0.9,*/*;q=0.8"
        
        try:
            self._delay()
            response = self.session.get(url, params=params, headers=headers, timeout=30)
            
            logger.debug(f"Request #{self._request_count}: {url} -> {response.status_code}")
            
            if response.status_code == 429:
                logger.warning(f"Rate limited (429) on {url}")
                raise RetryableError(f"Rate limited: {url}")
            
            if response.status_code >= 500:
                logger.warning(f"Server error ({response.status_code}) on {url}")
                raise RetryableError(f"Server error {response.status_code}: {url}")
            
            if response.status_code == 404:
                logger.warning(f"Not found (404): {url}")
                raise NonRetryableError(f"Not found: {url}")
            
            if response.status_code >= 400:
                logger.warning(f"Client error ({response.status_code}) on {url}")
                raise NonRetryableError(f"Client error {response.status_code}: {url}")
            
            return response
            
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout on {url}")
            raise RetryableError(f"Timeout: {url}")
            
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"Connection error on {url}: {e}")
            raise RetryableError(f"Connection error: {url}")
    
    # =========================================================================
    # OAI-PMH Methods (Primary Metadata Discovery)
    # =========================================================================
    
    def oai_list_records(
        self,
        set_spec: Optional[str] = None,
        from_date: Optional[str] = None,
        resumption_token: Optional[str] = None
    ) -> Generator[Tuple[Dict[str, Any], Optional[str]], None, None]:
        """
        Harvest records via OAI-PMH ListRecords.
        
        Args:
            set_spec: OAI set specification (e.g., division)
            from_date: Filter records from this date (YYYY-MM-DD)
            resumption_token: Token for pagination
            
        Yields:
            Tuple of (record_dict, next_resumption_token)
        """
        params = {
            "verb": "ListRecords",
            "metadataPrefix": "oai_dc"
        }
        
        if resumption_token:
            params = {"verb": "ListRecords", "resumptionToken": resumption_token}
        else:
            if set_spec:
                params["set"] = set_spec
            if from_date:
                params["from"] = from_date
        
        try:
            response = self._request(OAI_ENDPOINT, params=params, accept_xml=True)
            root = ET.fromstring(response.content)
            
            # Extract records
            for record in root.findall(".//oai:record", OAI_NS):
                record_data = self._parse_oai_record(record)
                if record_data:
                    yield record_data
            
            # Check for resumption token
            token_elem = root.find(".//oai:resumptionToken", OAI_NS)
            if token_elem is not None and token_elem.text:
                self.progress.last_oai_token = token_elem.text
                logger.info(f"OAI resumption token found, more records available")
            else:
                self.progress.last_oai_token = None
                
        except ET.ParseError as e:
            logger.error(f"Failed to parse OAI-PMH XML: {e}")
        except NonRetryableError:
            logger.error(f"OAI-PMH request failed (non-retryable)")
        except RetryableError:
            logger.error(f"OAI-PMH request failed after retries")
    
    def _parse_oai_record(self, record_elem: ET.Element) -> Optional[Dict[str, Any]]:
        """
        Parse OAI-PMH record element to dictionary.
        
        Args:
            record_elem: XML Element for a single record
            
        Returns:
            Parsed record dict or None
        """
        try:
            header = record_elem.find("oai:header", OAI_NS)
            metadata = record_elem.find("oai:metadata/oai_dc:dc", OAI_NS)
            
            if header is None or metadata is None:
                return None
            
            # Check if deleted
            if header.get("status") == "deleted":
                return None
            
            # Extract identifier (URL)
            identifier = header.findtext("oai:identifier", namespaces=OAI_NS)
            if not identifier:
                return None
            
            # Convert OAI identifier to repository URL
            # Format: oai:repository.its.ac.id:130281 -> https://repository.its.ac.id/130281/
            eprint_id = identifier.split(":")[-1] if ":" in identifier else identifier
            url = f"{BASE_URL}/{eprint_id}/"
            
            # Extract Dublin Core fields
            title = metadata.findtext("dc:title", namespaces=OAI_NS)
            creators = metadata.findall("dc:creator", OAI_NS)
            dates = metadata.findall("dc:date", OAI_NS)
            subjects = metadata.findall("dc:subject", OAI_NS)
            description = metadata.findtext("dc:description", namespaces=OAI_NS)
            types = metadata.findall("dc:type", OAI_NS)
            
            # Parse author
            author = creators[0].text if creators and creators[0].text else "Unknown"
            
            # Parse year from date
            year = None
            for date_elem in dates:
                if date_elem.text:
                    year_match = re.search(r'\d{4}', date_elem.text)
                    if year_match:
                        year = int(year_match.group())
                        break
            
            # Parse degree from type or subjects
            degree = self._extract_degree_from_oai(types, subjects)
            
            # Parse department from subjects
            dept = self._extract_dept_from_oai(subjects)
            
            return {
                "title": title,
                "author": author,
                "year": year,
                "degree": degree,
                "dept": dept,
                "abstract": description,  # May be thin/empty
                "url": url,
                "supervisors": [],  # Need enrichment
                "metadata": {
                    "oai_identifier": identifier,
                    "oai_datestamp": header.findtext("oai:datestamp", namespaces=OAI_NS),
                    "subjects": [s.text for s in subjects if s.text]
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to parse OAI record: {e}")
            return None
    
    def _extract_degree_from_oai(
        self,
        types: List[ET.Element],
        subjects: List[ET.Element]
    ) -> str:
        """Extract degree level from OAI type/subject fields."""
        all_text = " ".join(
            (t.text or "") for t in types + subjects
        ).lower()
        
        if "s1" in all_text or "undergraduate" in all_text or "sarjana" in all_text:
            return "S1"
        if "s2" in all_text or "master" in all_text or "magister" in all_text:
            return "S2"
        if "s3" in all_text or "doctoral" in all_text or "doktor" in all_text:
            return "S3"
        if "thesis" in all_text:
            return "S1"  # Default thesis to S1
        
        return ""
    
    def _extract_dept_from_oai(self, subjects: List[ET.Element]) -> str:
        """Extract department from OAI subject fields."""
        for subject in subjects:
            if subject.text and any(kw in subject.text.lower() for kw in 
                ["department", "departemen", "jurusan", "fakultas", "faculty"]):
                return subject.text
        return ""
    
    # =========================================================================
    # HTML Scraping Methods (Enrichment Layer)
    # =========================================================================
    
    def scrape_division_listing(
        self,
        division_url: str,
        max_years: int = 50
    ) -> Generator[str, None, None]:
        """
        Scrape thesis URLs from a division listing page.
        
        Division pages contain year links (e.g., /2024.html).
        Each year page contains the actual thesis URLs.
        
        Args:
            division_url: URL of the division listing page
            max_years: Maximum year pages to scrape
            
        Yields:
            Thesis URLs
        """
        logger.info(f"Scraping division: {division_url}")
        
        # Step 1: Get year page links from division page
        year_urls = []
        try:
            response = self._request(division_url)
            soup = BeautifulSoup(response.content, "lxml")
            
            # Find year links (e.g., 2024.html, /S1IF/2024.html)
            for link in soup.find_all("a", href=True):
                href = link["href"]
                # Match year pages like 2024.html or /S1IF/2024.html or full URL
                if re.search(r"\d{4}\.html$", href):
                    full_url = urljoin(division_url, href)
                    if full_url not in year_urls:
                        year_urls.append(full_url)
            
            logger.info(f"Found {len(year_urls)} year pages in division")
            
        except (RetryableError, NonRetryableError) as e:
            logger.error(f"Failed to scrape division page: {e}")
            return
        
        # Step 2: Scrape each year page for thesis URLs
        for idx, year_url in enumerate(year_urls[:max_years]):
            logger.info(f"Scraping year page {idx + 1}/{len(year_urls)}: {year_url}")
            
            try:
                response = self._request(year_url)
                soup = BeautifulSoup(response.content, "lxml")
                
                # Find thesis links (e.g., http://repository.its.ac.id/111537/)
                for link in soup.find_all("a", href=True):
                    href = link["href"]
                    # Match pattern like /130281/ or http://repository.its.ac.id/130281/
                    if re.match(r"(?:https?://repository\.its\.ac\.id)?/\d{5,}/", href):
                        full_url = urljoin(BASE_URL, href)
                        self.progress.total_discovered += 1
                        yield full_url
                        
            except (RetryableError, NonRetryableError) as e:
                logger.error(f"Failed to scrape year page {year_url}: {e}")
    
    def enrich_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich a record with HTML scraping for full abstract and supervisors.
        
        Args:
            record: Record with URL to enrich
            
        Returns:
            Enriched record
        """
        url = record.get("url")
        if not url:
            return record
        
        try:
            response = self._request(url)
            soup = BeautifulSoup(response.content, "lxml")
            
            # Enrich abstract if missing or thin
            if not record.get("abstract") or len(record.get("abstract", "")) < 100:
                abstract = self._extract_abstract_from_html(soup)
                if abstract and len(abstract) > len(record.get("abstract", "")):
                    record["abstract"] = abstract
            
            # Extract supervisors from meta tags first
            supervisors = self._extract_supervisors_from_meta(soup)
            
            # Fallback: extract from abstract text
            if not supervisors and record.get("abstract"):
                supervisors = self.cleaner.extract_supervisors(record["abstract"])
            
            if supervisors:
                record["supervisors"] = supervisors
            
            # Extract additional metadata
            record = self._enrich_metadata_from_html(record, soup)
            
            logger.debug(f"Enriched record: {record.get('title', '')[:50]}")
            return record
            
        except (RetryableError, NonRetryableError) as e:
            logger.warning(f"Failed to enrich record {url}: {e}")
            return record
    
    def _extract_abstract_from_html(self, soup: BeautifulSoup) -> str:
        """Extract abstract from thesis HTML page."""
        # Try common abstract containers
        selectors = [
            ("h2", "Abstract"),
            ("h3", "Abstract"),
            ("h2", "Abstrak"),
            ("h3", "Abstrak"),
        ]
        
        for tag, text in selectors:
            header = soup.find(tag, string=re.compile(text, re.IGNORECASE))
            if header:
                # Get following paragraph
                next_elem = header.find_next("p")
                if next_elem:
                    return next_elem.get_text(strip=True)
        
        # Fallback: Look for div/p with abstract class
        abstract_div = soup.find(class_=re.compile(r"abstract", re.IGNORECASE))
        if abstract_div:
            return abstract_div.get_text(strip=True)
        
        # Last resort: Find main content area and extract long text
        main_content = soup.find("div", {"id": "abstract"}) or soup.find("div", class_="ep_summary_content_main")
        if main_content:
            return main_content.get_text(strip=True)
        
        return ""
    
    def _extract_supervisors_from_meta(self, soup: BeautifulSoup) -> List[str]:
        """Extract supervisors from meta tags or structured data."""
        supervisors = []
        
        # Check meta tags
        for meta in soup.find_all("meta", {"name": re.compile(r"citation_advisor|supervisor", re.IGNORECASE)}):
            content = meta.get("content")
            if content:
                supervisors.append(content.strip())
        
        # Check table rows for supervisor info
        for row in soup.find_all("tr"):
            cells = row.find_all(["th", "td"])
            if len(cells) >= 2:
                label = cells[0].get_text(strip=True).lower()
                if any(kw in label for kw in ["pembimbing", "supervisor", "advisor"]):
                    value = cells[1].get_text(strip=True)
                    if value and value not in supervisors:
                        supervisors.append(value)
        
        return supervisors[:2]  # Max 2 supervisors
    
    def _enrich_metadata_from_html(
        self,
        record: Dict[str, Any],
        soup: BeautifulSoup
    ) -> Dict[str, Any]:
        """Extract additional metadata from HTML."""
        metadata = record.get("metadata", {})
        
        # Extract department if missing
        if not record.get("dept"):
            for link in soup.find_all("a", href=re.compile(r"/view/divisions/")):
                text = link.get_text(strip=True)
                if any(kw in text.lower() for kw in ["department", "fakultas", "jurusan"]):
                    record["dept"] = text
                    break
        
        # Extract degree if missing
        if not record.get("degree"):
            for link in soup.find_all("a", href=re.compile(r"S[123]|Undergraduate|Master|Doctoral")):
                text = link.get_text(strip=True)
                if "S1" in text or "Undergraduate" in text:
                    record["degree"] = "S1"
                elif "S2" in text or "Master" in text:
                    record["degree"] = "S2"
                elif "S3" in text or "Doctoral" in text:
                    record["degree"] = "S3"
                break
        
        # Store PDF URL if available
        pdf_link = soup.find("a", href=re.compile(r"\.pdf$", re.IGNORECASE))
        if pdf_link:
            metadata["pdf_url"] = urljoin(BASE_URL, pdf_link["href"])
        
        record["metadata"] = metadata
        return record
    
    def scrape_thesis_page(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Scrape a single thesis page completely (no OAI-PMH).
        
        Args:
            url: Thesis page URL
            
        Returns:
            Parsed record or None
        """
        try:
            response = self._request(url)
            soup = BeautifulSoup(response.content, "lxml")
            
            # Extract title
            title = None
            title_elem = soup.find("h1") or soup.find("title")
            if title_elem:
                title = title_elem.get_text(strip=True)
                # Remove site suffix
                title = re.sub(r"\s*-\s*ITS Repository$", "", title)
            
            # Extract author from citation or content
            author = None
            author_meta = soup.find("meta", {"name": "citation_author"})
            if author_meta:
                author = author_meta.get("content")
            else:
                # Look for author pattern in page content
                for p in soup.find_all("p"):
                    text = p.get_text()
                    # Pattern: "Author Name (2025) Title..."
                    match = re.match(r"^([A-Za-z\s,']+)\s*\(\d{4}\)", text)
                    if match:
                        author = match.group(1).strip()
                        break
            
            # Extract year
            year = None
            year_meta = soup.find("meta", {"name": "citation_date"})
            if year_meta and year_meta.get("content"):
                year_match = re.search(r"\d{4}", year_meta["content"])
                if year_match:
                    year = int(year_match.group())
            
            if not year:
                # Search in page content
                year_match = re.search(r"\((\d{4})\)", soup.get_text())
                if year_match:
                    year = int(year_match.group(1))
            
            # Extract abstract
            abstract = self._extract_abstract_from_html(soup)
            
            # Extract supervisors
            supervisors = self._extract_supervisors_from_meta(soup)
            if not supervisors and abstract:
                supervisors = self.cleaner.extract_supervisors(abstract)
            
            # Build record
            record = {
                "title": title,
                "author": author or "Unknown",
                "year": year,
                "abstract": abstract,
                "supervisors": supervisors,
                "degree": "",
                "dept": "",
                "url": url,
                "metadata": {}
            }
            
            # Enrich with additional HTML metadata
            record = self._enrich_metadata_from_html(record, soup)
            
            return record
            
        except (RetryableError, NonRetryableError) as e:
            logger.error(f"Failed to scrape thesis page {url}: {e}")
            self.progress.failed_urls.append({"url": url, "error": str(e)})
            return None
    
    # =========================================================================
    # Main Scraping Methods
    # =========================================================================
    
    def scrape_fteic_pilot(
        self,
        divisions: Optional[List[str]] = None,
        limit: Optional[int] = None,
        skip_existing: Optional[set] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Scrape FTEIC divisions for pilot run.
        
        Args:
            divisions: List of division keys to scrape (default: all FTEIC)
            limit: Maximum records to yield
            skip_existing: Set of URLs to skip (already in DB)
            
        Yields:
            Cleaned research records
        """
        divisions = divisions or list(FTEIC_DIVISIONS.keys())
        skip_existing = skip_existing or set()
        yielded = 0
        
        for div_name in divisions:
            if div_name not in FTEIC_DIVISIONS:
                logger.warning(f"Unknown division: {div_name}")
                continue
            
            div_url = FTEIC_DIVISIONS[div_name]
            logger.info(f"Starting division: {div_name}")
            
            for thesis_url in self.scrape_division_listing(div_url):
                if limit and yielded >= limit:
                    logger.info(f"Reached limit of {limit} records")
                    return
                
                if thesis_url in skip_existing:
                    logger.debug(f"Skipping existing: {thesis_url}")
                    self.progress.total_processed += 1
                    continue
                
                if thesis_url in self.progress.processed_urls:
                    continue
                
                # Scrape the thesis page
                record = self.scrape_thesis_page(thesis_url)
                
                if record:
                    # Clean the record
                    cleaned = self.cleaner.clean_record(record)
                    
                    # Update progress
                    self.progress.total_processed += 1
                    self.progress.total_success += 1
                    self.progress.processed_urls.append(thesis_url)
                    self.progress.last_url = thesis_url
                    
                    yielded += 1
                    yield cleaned
                else:
                    self.progress.total_processed += 1
                    self.progress.total_failed += 1
    
    def save_checkpoint(self, filepath: str) -> None:
        """
        Save current progress to checkpoint file.
        
        Args:
            filepath: Path to checkpoint JSON file
        """
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(self.progress.to_dict(), f, indent=2, ensure_ascii=False)
            logger.info(f"Checkpoint saved: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def load_checkpoint(self, filepath: str) -> bool:
        """
        Load progress from checkpoint file.
        
        Args:
            filepath: Path to checkpoint JSON file
            
        Returns:
            True if checkpoint loaded successfully
        """
        try:
            if os.path.exists(filepath):
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.progress = ScrapingProgress.from_dict(data)
                logger.info(f"Checkpoint loaded: {self.progress.total_processed} processed")
                return True
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
        return False


# Convenience function
def create_scraper(**kwargs) -> ITSScraper:
    """Factory function to create scraper with configuration."""
    return ITSScraper(**kwargs)
