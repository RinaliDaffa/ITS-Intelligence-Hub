"""
ITS Intelligence Hub v2 - SINTA Scraper Module
Playwright-based scraper for sinta.kemdiktisaintek.go.id

Features:
- Stealth mode with anti-detection measures
- High-jitter delays (5-10 seconds between authors)
- Checkpoint-based resume capability
- Self-healing on 403/429 errors
"""

import os
import re
import json
import random
import time
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from fake_useragent import UserAgent
from tqdm import tqdm

try:
    from playwright.sync_api import sync_playwright, Page, Browser, BrowserContext
    from playwright.sync_api import TimeoutError as PlaywrightTimeout
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    print("⚠️  Playwright not installed. Run: pip install playwright && playwright install chromium")

load_dotenv()
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

# SINTA URLs
SINTA_BASE_URL = "https://sinta.kemdiktisaintek.go.id"
ITS_AFFILIATION_ID = "417"  # ITS (Institut Teknologi Sepuluh Nopember) - correct ID for new domain
ITS_AUTHORS_URL = f"{SINTA_BASE_URL}/affiliations/authors/{ITS_AFFILIATION_ID}"

# Scraping configuration - Speed optimized with anti-blocking
DELAY_MIN = float(os.getenv("SINTA_DELAY_MIN", "3"))    # 3 seconds min delay (faster)
DELAY_MAX = float(os.getenv("SINTA_DELAY_MAX", "5"))    # 5 seconds max delay (faster)
PAGE_TIMEOUT = int(os.getenv("SINTA_PAGE_TIMEOUT", "30000"))   # 30 seconds timeout
MAX_PUBLICATIONS = int(os.getenv("SINTA_MAX_PUBLICATIONS", "50"))  # 50 pubs per source (less requests)
MIN_YEAR = int(os.getenv("SINTA_MIN_YEAR", "2015"))
AUTHOR_RETRY_COUNT = int(os.getenv("SINTA_AUTHOR_RETRY", "3"))  # 3 retries per author
MAX_CONSECUTIVE_ERRORS = int(os.getenv("SINTA_MAX_ERRORS", "20"))  # Pause after 20 consecutive errors

# Checkpoint paths
DATA_DIR = Path(__file__).parent.parent / "data"
CHECKPOINT_FILE = DATA_DIR / "sinta_checkpoint.json"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class SINTAAuthor:
    """Represents a scraped SINTA author."""
    sinta_id: str
    name: str
    nidn: Optional[str] = None
    dept: Optional[str] = None
    h_index_scopus: int = 0
    h_index_gscholar: int = 0
    publications: List[Dict[str, Any]] = field(default_factory=list)
    expertise_tags: List[str] = field(default_factory=list)
    researches: List[Dict[str, Any]] = field(default_factory=list)
    services: List[Dict[str, Any]] = field(default_factory=list)
    iprs: List[Dict[str, Any]] = field(default_factory=list)
    books: List[Dict[str, Any]] = field(default_factory=list)
    scraped_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ScrapeProgress:
    """Track scraping progress for checkpointing."""
    total_discovered: int = 0
    total_processed: int = 0
    total_success: int = 0
    total_failed: int = 0
    current_page: int = 1
    processed_sinta_ids: List[str] = field(default_factory=list)
    failed_sinta_ids: List[Dict[str, str]] = field(default_factory=list)
    cached_authors: List[Dict[str, str]] = field(default_factory=list)  # Cache author list for resume
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    is_paused: bool = False
    pause_reason: Optional[str] = None
    consecutive_network_errors: int = 0  # Track network errors for auto-pause
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["updated_at"] = datetime.now().isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScrapeProgress":
        valid_fields = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid_fields)


# ============================================================================
# SINTA Scraper Class
# ============================================================================

class SINTAScraper:
    """
    Playwright-based scraper for SINTA academic database.
    
    Uses stealth configuration to avoid detection and implements
    checkpoint-based resume capability.
    """
    
    def __init__(
        self,
        headless: bool = True,
        delay_min: float = DELAY_MIN,
        delay_max: float = DELAY_MAX,
        max_publications: int = MAX_PUBLICATIONS
    ) -> None:
        """
        Initialize SINTA scraper.
        
        Args:
            headless: Run browser in headless mode
            delay_min: Minimum delay between requests (seconds)
            delay_max: Maximum delay between requests (seconds)
            max_publications: Maximum publications to extract per author
        """
        if not PLAYWRIGHT_AVAILABLE:
            raise ImportError("Playwright is required. Install with: pip install playwright && playwright install chromium")
        
        self.headless = headless
        self.delay_min = delay_min
        self.delay_max = delay_max
        self.max_publications = max_publications
        
        self.ua = UserAgent(fallback="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
        self.progress = ScrapeProgress()
        
        self._playwright = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        self._request_count = 0  # Track requests for context rotation
        
        # Initialize Supabase client for incremental saves
        from supabase import create_client, Client
        SUPABASE_URL = os.getenv("SUPABASE_URL")
        SUPABASE_KEY = os.getenv("SUPABASE_KEY")
        if SUPABASE_URL and SUPABASE_KEY:
            self._supabase: Optional[Client] = create_client(SUPABASE_URL, SUPABASE_KEY)
        else:
            self._supabase = None
            logger.warning("Supabase credentials not configured - dry-run mode only")
        
        # Ensure data directory exists
        DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    def _get_stealth_args(self) -> Dict[str, Any]:
        """Get browser launch arguments for stealth mode."""
        return {
            "headless": self.headless,
            "args": [
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-infobars",
                "--window-size=1920,1080",
                "--start-maximized",
            ]
        }
    
    def _get_context_options(self) -> Dict[str, Any]:
        """Get browser context options with randomized fingerprint."""
        viewports = [
            {"width": 1920, "height": 1080},
            {"width": 1366, "height": 768},
            {"width": 1536, "height": 864},
            {"width": 1440, "height": 900},
        ]
        
        return {
            "viewport": random.choice(viewports),
            "user_agent": self.ua.random,
            "locale": "id-ID",
            "timezone_id": "Asia/Jakarta",
            "permissions": [],
            "java_script_enabled": True,
        }
    
    def _delay(self) -> None:
        """Apply randomized high-jitter delay."""
        delay = random.uniform(self.delay_min, self.delay_max)
        logger.debug(f"Sleeping for {delay:.2f} seconds")
        time.sleep(delay)
    
    def start_browser(self) -> None:
        """Initialize Playwright browser with stealth configuration."""
        logger.info("Starting Playwright browser (stealth mode)...")
        
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(**self._get_stealth_args())
        self._context = self._browser.new_context(**self._get_context_options())
        self._page = self._context.new_page()
        
        # Inject stealth scripts to evade detection
        self._context.add_init_script("""
            // Remove webdriver property
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
            
            // Mock plugins
            Object.defineProperty(navigator, 'plugins', {
                get: () => [1, 2, 3, 4, 5]
            });
            
            // Mock languages
            Object.defineProperty(navigator, 'languages', {
                get: () => ['id-ID', 'id', 'en-US', 'en']
            });
            
            // Mock permissions
            const originalQuery = window.navigator.permissions.query;
            window.navigator.permissions.query = (parameters) => (
                parameters.name === 'notifications' ?
                    Promise.resolve({ state: Notification.permission }) :
                    originalQuery(parameters)
            );
        """)
        
        # Set default timeout
        self._page.set_default_timeout(PAGE_TIMEOUT)
        
        logger.info("Browser started successfully")
    
    def stop_browser(self) -> None:
        """Close browser and cleanup."""
        try:
            if self._context:
                self._context.close()
        except Exception:
            pass  # Already closed
        
        try:
            if self._browser:
                self._browser.close()
        except Exception:
            pass  # Already closed
            
        try:
            if self._playwright:
                self._playwright.stop()
        except Exception:
            pass
        
        self._page = None
        self._context = None
        self._browser = None
        self._playwright = None
        
        logger.info("Browser closed")
    
    def rotate_context(self) -> None:
        """Create fresh browser context with new fingerprint."""
        logger.info("Rotating browser context for fresh fingerprint...")
        
        try:
            if self._context:
                self._context.close()
        except Exception:
            pass  # Context may already be closed
        
        try:
            self._context = self._browser.new_context(**self._get_context_options())
            self._page = self._context.new_page()
            self._page.set_default_timeout(PAGE_TIMEOUT)
            
            # Warmup: navigate to SINTA homepage to establish connection
            logger.info("Warming up new context...")
            time.sleep(5)  # Wait before first request with new context
            try:
                self._page.goto("https://sinta.kemdiktisaintek.go.id", wait_until="domcontentloaded", timeout=30000)
                time.sleep(3)  # Let the page fully settle
            except Exception as e:
                logger.warning(f"Warmup navigation failed (non-critical): {e}")
                
        except Exception as e:
            logger.warning(f"Failed to rotate context, restarting browser: {e}")
            self.stop_browser()
            time.sleep(5)
            self.start_browser()
    
    def save_checkpoint(self) -> None:
        """Save current progress to checkpoint file."""
        try:
            with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
                json.dump(self.progress.to_dict(), f, indent=2, ensure_ascii=False)
            logger.info(f"Checkpoint saved: {self.progress.total_processed} processed")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def load_checkpoint(self) -> bool:
        """Load progress from checkpoint file."""
        try:
            if CHECKPOINT_FILE.exists():
                with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.progress = ScrapeProgress.from_dict(data)
                logger.info(f"Checkpoint loaded: {self.progress.total_processed} already processed")
                return True
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
        return False
    
    def sync_with_supabase(self) -> None:
        """
        Sync processed_sinta_ids with Supabase to avoid re-scraping.
        Fetches all sinta_ids already in database and merges with checkpoint.
        """
        if not self._supabase:
            logger.warning("Supabase not configured, skipping sync")
            return
        
        try:
            logger.info("Syncing with Supabase to find already-scraped authors...")
            
            # Fetch all sinta_ids from database
            all_ids = []
            offset = 0
            batch_size = 1000
            
            while True:
                result = self._supabase.table('lecturers').select('sinta_id').range(offset, offset + batch_size - 1).execute()
                if not result.data:
                    break
                all_ids.extend([str(r['sinta_id']) for r in result.data if r.get('sinta_id')])
                offset += batch_size
                if len(result.data) < batch_size:
                    break
            
            # Merge with existing processed_sinta_ids
            existing = set(self.progress.processed_sinta_ids)
            db_ids = set(all_ids)
            merged = list(existing | db_ids)
            
            old_count = len(existing)
            new_count = len(merged)
            
            self.progress.processed_sinta_ids = merged
            self.progress.total_processed = new_count
            self.progress.total_success = new_count
            
            logger.info(f"Synced with Supabase: {old_count} -> {new_count} processed IDs")
            
        except Exception as e:
            logger.error(f"Failed to sync with Supabase: {e}")
    
    def _handle_blocking(self, response_status: int, url: str) -> None:
        """Handle 403/429 responses by pausing and notifying."""
        self.progress.is_paused = True
        
        if response_status == 403:
            self.progress.pause_reason = f"403 Forbidden - Access blocked at {url}"
            logger.error("🚫 BLOCKED: 403 Forbidden - Site is blocking scraper")
        elif response_status == 429:
            self.progress.pause_reason = f"429 Too Many Requests - Rate limited at {url}"
            logger.error("⏱️  RATE LIMITED: 429 Too Many Requests")
        
        self.save_checkpoint()
        
        # Print notification to console
        print("\n" + "=" * 60)
        print("⚠️  SCRAPER AUTO-PAUSED")
        print(f"Reason: {self.progress.pause_reason}")
        print(f"Progress saved: {self.progress.total_processed}/{self.progress.total_discovered}")
        print("Resume with: python -m src.sinta_scraper --resume")
        print("=" * 60 + "\n")
    
    def _track_request(self) -> None:
        """Track requests and rotate context every 100 requests (FIX for connection reset)."""
        self._request_count += 1
        if self._request_count % 100 == 0:
            logger.info(f"Request count: {self._request_count}, rotating context...")
            self.rotate_context()
            time.sleep(3)  # Brief pause after rotation
    
    def fetch_author_list(self, max_pages: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Fetch list of authors from ITS affiliation page.
        
        Args:
            max_pages: Maximum pages to scrape (None for all)
            
        Returns:
            List of dicts with sinta_id and name
        """
        authors = []
        page_num = self.progress.current_page
        total_pages = None
        page_retry_count = 0  # Track retries for each page
        
        logger.info(f"Fetching author list from page {page_num}...")
        
        while True:
            if max_pages and page_num > max_pages:
                break
            
            if total_pages and page_num > total_pages:
                break
            
            url = f"{ITS_AUTHORS_URL}?page={page_num}"
            
            try:
                response = self._page.goto(url, wait_until="domcontentloaded")
                self._track_request()  # FIX #3: Track request for context rotation
                
                if response and response.status in [403, 429]:
                    self._handle_blocking(response.status, url)
                    break
                
                # Wait for author cards to load (new domain uses .au-item)
                self._page.wait_for_selector(".au-item", timeout=15000)
                
                # Extract total pages from "Page X of Y" text on first load
                if total_pages is None:
                    try:
                        page_text = self._page.inner_text("body")
                        page_match = re.search(r"Page\s+\d+\s+of\s+(\d+)", page_text)
                        if page_match:
                            total_pages = int(page_match.group(1))
                            logger.info(f"Total pages detected: {total_pages}")
                    except Exception:
                        pass
                
                # Extract author cards
                author_cards = self._page.query_selector_all(".au-item")
                
                if not author_cards:
                    logger.info(f"No more authors found on page {page_num}")
                    break
                
                for card in author_cards:
                    try:
                        # Extract SINTA ID from profile link (new format: /authors/profile/[ID])
                        link = card.query_selector("a[href*='/authors/profile/']")
                        if not link:
                            continue
                        
                        href = link.get_attribute("href")
                        # Extract ID from URL like /authors/profile/29555
                        sinta_id_match = re.search(r"/authors/profile/(\d+)", href)
                        if not sinta_id_match:
                            continue
                        
                        sinta_id = sinta_id_match.group(1)
                        
                        # Skip if already processed
                        if sinta_id in self.progress.processed_sinta_ids:
                            continue
                        
                        # Extract name from the profile link text
                        name = link.inner_text().strip() if link else "Unknown"
                        
                        # Also try to extract department from card
                        dept_link = card.query_selector("a[href*='/departments/profile/']")
                        dept = dept_link.inner_text().strip() if dept_link else None
                        
                        authors.append({
                            "sinta_id": sinta_id,
                            "name": name,
                            "dept": dept
                        })
                        self.progress.total_discovered += 1
                        
                    except Exception as e:
                        logger.warning(f"Failed to parse author card: {e}")
                
                logger.info(f"Page {page_num}/{total_pages or '?'}: Found {len(author_cards)} authors (Total: {len(authors)})")
                
                # Check if we should continue to next page
                # Use total_pages if detected, otherwise check for "Next" link
                if total_pages:
                    if page_num >= total_pages:
                        logger.info("Reached last page")
                        break
                else:
                    # Fallback: look for Next button by text content
                    next_links = self._page.query_selector_all("a.page-link")
                    has_next = False
                    for link in next_links:
                        try:
                            text = link.inner_text().strip().lower()
                            if text == "next":
                                has_next = True
                                break
                        except:
                            pass
                    
                    if not has_next:
                        logger.info("No 'Next' link found, stopping pagination")
                        break
                
                page_num += 1
                self.progress.current_page = page_num
                page_retry_count = 0  # Reset retry count on success
                self._delay()
                
            except PlaywrightTimeout:
                logger.error(f"Timeout loading page {page_num}")
                # Retry with longer timeout and warmup
                if page_retry_count < 3:
                    page_retry_count += 1
                    logger.info(f"Retrying page {page_num} (attempt {page_retry_count}/3) with warmup...")
                    time.sleep(10)  # Wait 10 seconds before retry
                    # Warmup the browser
                    try:
                        self._page.goto("https://sinta.kemdiktisaintek.go.id", wait_until="domcontentloaded", timeout=30000)
                        time.sleep(5)
                    except:
                        # Browser might be dead, restart it
                        logger.warning("Browser dead during warmup, restarting...")
                        self.stop_browser()
                        time.sleep(5)
                        self.start_browser()
                    continue  # Retry the same page
                else:
                    logger.error(f"Failed to load page {page_num} after 3 retries, stopping")
                    break
            except Exception as e:
                error_msg = str(e).lower()
                logger.error(f"Error fetching page {page_num}: {e}")
                
                # Check if browser crashed or closed
                if "crashed" in error_msg or "closed" in error_msg or "target" in error_msg:
                    logger.warning("Browser crashed, restarting...")
                    self.stop_browser()
                    time.sleep(5)
                    self.start_browser()
                
                if page_retry_count < 3:
                    page_retry_count += 1
                    logger.info(f"Retrying page {page_num} (attempt {page_retry_count}/3)...")
                    time.sleep(10)
                    continue
                else:
                    break
        
        logger.info(f"Total authors discovered: {len(authors)}")
        return authors
    
    def scrape_author_profile(self, sinta_id: str) -> Optional[SINTAAuthor]:
        """
        Scrape detailed profile for a single author.
        
        Args:
            sinta_id: SINTA author ID
            
        Returns:
            SINTAAuthor object or None on failure
        """
        # New URL format: /authors/profile/[ID]
        url = f"{SINTA_BASE_URL}/authors/profile/{sinta_id}"
        
        try:
            response = self._page.goto(url, wait_until="domcontentloaded")
            self._track_request()  # FIX #3: Track request for context rotation
            
            if response and response.status in [403, 429]:
                self._handle_blocking(response.status, url)
                return None
            
            # Wait for profile to load
            self._page.wait_for_selector(".profile-name, h3", timeout=15000)
            
            # Extract basic info from profile header
            name = self._extract_text(".profile-name") or self._extract_text("h3")
            
            # NIDN is not visible on public profiles in new domain
            nidn = None
            
            # Try to extract NIDN from page text if available
            page_text = self._page.inner_text("body")
            nidn_match = re.search(r"NIDN\s*:\s*(\d{10})", page_text)
            if nidn_match:
                nidn = nidn_match.group(1)
            
            # Extract department
            dept = self._extract_text(".profile-affil") or self._extract_text("a[href*='/departments/profile/']")
            
            # Clean NIDN (often has prefix like "NIDN: 0012345678")
            if nidn:
                nidn_match = re.search(r"\d{10}", nidn)
                nidn = nidn_match.group(0) if nidn_match else nidn
            
            # Extract H-Index scores
            h_index_scopus = self._extract_h_index("scopus")
            h_index_gscholar = self._extract_h_index("scholar")
            
            # Extract expertise tags from profile page
            expertise_tags = self._extract_expertise_tags()
            
            # Extract publications
            publications = self._extract_publications(sinta_id)
            
            # Extract additional data
            researches = self._extract_researches(sinta_id)
            services = self._extract_services(sinta_id)
            iprs = self._extract_iprs(sinta_id)
            books = self._extract_books(sinta_id)
            
            author = SINTAAuthor(
                sinta_id=sinta_id,
                name=name or "Unknown",
                nidn=nidn,
                dept=dept,
                h_index_scopus=h_index_scopus,
                h_index_gscholar=h_index_gscholar,
                publications=publications,
                expertise_tags=expertise_tags,
                researches=researches,
                services=services,
                iprs=iprs,
                books=books
            )
            
            logger.debug(f"Scraped: {author.name} ({len(publications)} pubs, {len(researches)} researches)")
            return author
            
        except PlaywrightTimeout:
            logger.warning(f"Timeout scraping author {sinta_id}")
            self.progress.failed_sinta_ids.append({
                "sinta_id": sinta_id,
                "error": "Timeout"
            })
            return None
        except Exception as e:
            error_str = str(e)
            logger.error(f"Error scraping author {sinta_id}: {e}")
            
            # Check for network errors
            network_error_keywords = [
                "ERR_NAME_NOT_RESOLVED",
                "ERR_INTERNET_DISCONNECTED", 
                "ERR_NETWORK_CHANGED",
                "ERR_CONNECTION_RESET",
                "ERR_CONNECTION_REFUSED",
                "net::ERR_"
            ]
            is_network_error = any(kw in error_str for kw in network_error_keywords)
            
            if is_network_error:
                self.progress.consecutive_network_errors += 1
                logger.warning(f"Network error detected ({self.progress.consecutive_network_errors}/{MAX_CONSECUTIVE_ERRORS})")
                
                # Auto-pause after MAX_CONSECUTIVE_ERRORS consecutive network errors
                if self.progress.consecutive_network_errors >= MAX_CONSECUTIVE_ERRORS:
                    self.progress.is_paused = True
                    self.progress.pause_reason = f"Network disconnected - {self.progress.consecutive_network_errors} consecutive failures"
                    self.save_checkpoint()
                    
                    print("\n" + "=" * 60)
                    print("⚠️  SCRAPER AUTO-PAUSED - NETWORK ERROR")
                    print(f"Reason: Internet connection lost")
                    print(f"Progress saved: {self.progress.total_processed} processed")
                    print(f"Reconnect and resume with: python -m src.sinta_scraper --resume")
                    print("=" * 60 + "\n")
            else:
                # Reset counter on non-network errors
                self.progress.consecutive_network_errors = 0
            
            self.progress.failed_sinta_ids.append({
                "sinta_id": sinta_id,
                "error": error_str
            })
            return None
    
    def _extract_text(self, selector: str) -> Optional[str]:
        """Extract text content from selector."""
        try:
            elem = self._page.query_selector(selector)
            if elem:
                return elem.inner_text().strip()
        except Exception:
            pass
        return None
    
    def _extract_h_index(self, source: str) -> int:
        """
        Extract H-Index for given source (scopus/scholar).
        
        SINTA displays H-Index in a table format like:
        'H-Index\t[scopus_value]\t[scholar_value]'
        """
        try:
            # Get page text and look for H-Index line
            body_text = self._page.inner_text("body")
            lines = body_text.split('\n')
            
            for line in lines:
                line_lower = line.lower().strip()
                
                # Look for line containing "h-index" or "h index"
                if 'h-index' in line_lower or 'h index' in line_lower:
                    # Split by tabs or multiple spaces
                    parts = [p.strip() for p in re.split(r'\t+|\s{2,}', line) if p.strip()]
                    
                    # Expected format: ['H-Index', 'scopus_val', 'scholar_val']
                    # or with labels: ['H-Index', 'Scopus', '34', 'Scholar', '45']
                    if len(parts) >= 2:
                        # Find numeric values
                        numbers = [p for p in parts if p.isdigit()]
                        
                        if source.lower() == "scopus" and len(numbers) >= 1:
                            return int(numbers[0])
                        elif source.lower() == "scholar" and len(numbers) >= 2:
                            return int(numbers[1])
                        elif source.lower() == "scholar" and len(numbers) == 1:
                            # Only one number found, might be combined
                            return int(numbers[0])
            
            # Fallback: Try table cells approach
            table_cells = self._page.query_selector_all("td, th, .stat-num, .pr-stat-num")
            for i, cell in enumerate(table_cells):
                try:
                    cell_text = cell.inner_text().strip().lower()
                    if 'h-index' in cell_text or 'h index' in cell_text:
                        # Get next cells for values
                        for j in range(1, 4):
                            if i + j < len(table_cells):
                                val = table_cells[i + j].inner_text().strip()
                                if val.isdigit():
                                    if source.lower() == "scopus" and j == 1:
                                        return int(val)
                                    elif source.lower() == "scholar" and j == 2:
                                        return int(val)
                except Exception:
                    continue
                    
        except Exception as e:
            logger.debug(f"Could not extract H-Index for {source}: {e}")
        return 0
    
    def _extract_expertise_tags(self) -> List[str]:
        """Extract expertise/subject tags from author profile."""
        tags = []
        try:
            # Look for subject links in profile
            subject_links = self._page.query_selector_all("a[href*='/subjects/detail/']")
            for link in subject_links[:10]:  # Max 10 tags
                try:
                    text = link.inner_text().strip()
                    if text and len(text) > 2 and text not in tags:
                        tags.append(text)
                except Exception:
                    pass
        except Exception as e:
            logger.debug(f"Could not extract expertise tags: {e}")
        return tags
    
    def _extract_researches(self, sinta_id: str) -> List[Dict[str, Any]]:
        """Extract research projects from SINTA."""
        researches = []
        try:
            url = f"{SINTA_BASE_URL}/authors/profile/{sinta_id}?view=researches"
            self._page.goto(url, wait_until="domcontentloaded", timeout=PAGE_TIMEOUT)
            self._track_request()  # FIX #3
            self._page.wait_for_selector(".ar-list-item, .ar-title", timeout=5000)
            
            items = self._page.query_selector_all(".ar-list-item")[:20]
            for item in items:
                try:
                    title_elem = item.query_selector(".ar-title a, .ar-title")
                    title = title_elem.inner_text().strip() if title_elem else None
                    if title and len(title) > 10:
                        year = None
                        item_text = item.inner_text()
                        year_match = re.search(r"\b(19|20)\d{2}\b", item_text)
                        if year_match:
                            year = int(year_match.group(0))
                        researches.append({"title": title, "year": year})
                except Exception:
                    pass
            logger.debug(f"Extracted {len(researches)} researches")
        except PlaywrightTimeout:
            logger.debug(f"Timeout loading researches for {sinta_id}")
        except Exception as e:
            logger.debug(f"Error extracting researches: {e}")
        return researches
    
    def _extract_services(self, sinta_id: str) -> List[Dict[str, Any]]:
        """Extract community services from SINTA."""
        services = []
        try:
            url = f"{SINTA_BASE_URL}/authors/profile/{sinta_id}?view=services"
            self._page.goto(url, wait_until="domcontentloaded", timeout=PAGE_TIMEOUT)
            self._track_request()  # FIX #3
            self._page.wait_for_selector(".ar-list-item, .ar-title", timeout=5000)
            
            items = self._page.query_selector_all(".ar-list-item")[:20]
            for item in items:
                try:
                    title_elem = item.query_selector(".ar-title a, .ar-title")
                    title = title_elem.inner_text().strip() if title_elem else None
                    if title and len(title) > 10:
                        year = None
                        item_text = item.inner_text()
                        year_match = re.search(r"\b(19|20)\d{2}\b", item_text)
                        if year_match:
                            year = int(year_match.group(0))
                        services.append({"title": title, "year": year})
                except Exception:
                    pass
            logger.debug(f"Extracted {len(services)} community services")
        except PlaywrightTimeout:
            logger.debug(f"Timeout loading services for {sinta_id}")
        except Exception as e:
            logger.debug(f"Error extracting services: {e}")
        return services
    
    def _extract_iprs(self, sinta_id: str) -> List[Dict[str, Any]]:
        """Extract IPRs (Intellectual Property Rights) from SINTA."""
        iprs = []
        try:
            url = f"{SINTA_BASE_URL}/authors/profile/{sinta_id}?view=iprs"
            self._page.goto(url, wait_until="domcontentloaded", timeout=PAGE_TIMEOUT)
            self._track_request()  # FIX #3
            self._page.wait_for_selector(".ar-list-item, .ar-title", timeout=5000)
            
            items = self._page.query_selector_all(".ar-list-item")[:20]
            for item in items:
                try:
                    title_elem = item.query_selector(".ar-title a, .ar-title")
                    title = title_elem.inner_text().strip() if title_elem else None
                    if title and len(title) > 5:
                        year = None
                        item_text = item.inner_text()
                        year_match = re.search(r"\b(19|20)\d{2}\b", item_text)
                        if year_match:
                            year = int(year_match.group(0))
                        iprs.append({"title": title, "year": year})
                except Exception:
                    pass
            logger.debug(f"Extracted {len(iprs)} IPRs")
        except PlaywrightTimeout:
            logger.debug(f"Timeout loading IPRs for {sinta_id}")
        except Exception as e:
            logger.debug(f"Error extracting IPRs: {e}")
        return iprs
    
    def _extract_books(self, sinta_id: str) -> List[Dict[str, Any]]:
        """Extract books from SINTA."""
        books = []
        try:
            url = f"{SINTA_BASE_URL}/authors/profile/{sinta_id}?view=books"
            self._page.goto(url, wait_until="domcontentloaded", timeout=PAGE_TIMEOUT)
            self._track_request()  # FIX #3
            self._page.wait_for_selector(".ar-list-item, .ar-title", timeout=5000)
            
            items = self._page.query_selector_all(".ar-list-item")[:20]
            for item in items:
                try:
                    title_elem = item.query_selector(".ar-title a, .ar-title")
                    title = title_elem.inner_text().strip() if title_elem else None
                    if title and len(title) > 5:
                        year = None
                        item_text = item.inner_text()
                        year_match = re.search(r"\b(19|20)\d{2}\b", item_text)
                        if year_match:
                            year = int(year_match.group(0))
                        books.append({"title": title, "year": year})
                except Exception:
                    pass
            logger.debug(f"Extracted {len(books)} books")
        except PlaywrightTimeout:
            logger.debug(f"Timeout loading books for {sinta_id}")
        except Exception as e:
            logger.debug(f"Error extracting books: {e}")
        return books
    
    def _extract_publications(self, sinta_id: str) -> List[Dict[str, Any]]:
        """
        Extract publication titles from both Scopus and Google Scholar tabs.
        
        Args:
            sinta_id: Author's SINTA ID
            
        Returns:
            List of publication dicts with title, source, year
        """
        publications = []
        
        # Try Scopus publications first
        scopus_pubs = self._extract_publications_from_tab(sinta_id, "scopus")
        publications.extend(scopus_pubs)
        
        # Then Google Scholar publications
        self._delay()
        gscholar_pubs = self._extract_publications_from_tab(sinta_id, "googlescholar")
        publications.extend(gscholar_pubs)
        
        # Limit to max publications
        return publications[:self.max_publications]
    
    def _extract_publications_from_tab(
        self,
        sinta_id: str,
        source: str
    ) -> List[Dict[str, Any]]:
        """Extract publications from a specific source tab with pagination."""
        pubs = []
        page_num = 1
        max_pages = 20  # Safety limit
        
        try:
            while len(pubs) < self.max_publications and page_num <= max_pages:
                # Navigate to publications view with page number
                url = f"{SINTA_BASE_URL}/authors/profile/{sinta_id}?view={source}&page={page_num}"
                self._page.goto(url, wait_until="domcontentloaded")
                self._track_request()  # FIX #3: Critical for pagination
                
                # Wait for publication list
                try:
                    self._page.wait_for_selector(".ar-title, .ar-list-item", timeout=8000)
                except PlaywrightTimeout:
                    # No more publications
                    break
                
                # Try to extract from .ar-list-item
                pub_items = self._page.query_selector_all(".ar-list-item")
                
                if not pub_items:
                    # No more publications on this page
                    break
                
                page_pubs_count = 0
                for item in pub_items:
                    if len(pubs) >= self.max_publications:
                        break
                        
                    try:
                        title_elem = item.query_selector(".ar-title a")
                        title = title_elem.inner_text().strip() if title_elem else None
                        
                        if not title or len(title) < 10:
                            continue
                        
                        # Extract year
                        year = None
                        try:
                            item_text = item.inner_text()
                            year_match = re.search(r"\b(19|20)\d{2}\b", item_text)
                            year = int(year_match.group(0)) if year_match else None
                        except Exception:
                            pass
                        
                        # Filter by MIN_YEAR
                        if year and year < MIN_YEAR:
                            continue
                        
                        pubs.append({
                            "title": title,
                            "source": source,
                            "year": year
                        })
                        page_pubs_count += 1
                        
                    except Exception as e:
                        logger.debug(f"Failed to parse publication item: {e}")
                
                # If this page had no valid publications, stop
                if page_pubs_count == 0:
                    break
                
                # Move to next page
                page_num += 1
                
                # Small delay between pagination
                if page_num <= max_pages and len(pubs) < self.max_publications:
                    time.sleep(1)
            
            logger.info(f"Extracted {len(pubs)} publications from {source} for author {sinta_id} (pages: {page_num}, year >= {MIN_YEAR})")
            
        except PlaywrightTimeout:
            logger.debug(f"Timeout loading {source} publications for {sinta_id}")
        except Exception as e:
            logger.debug(f"Error extracting {source} publications: {e}")
        
        return pubs
    
    def _save_author_to_db(self, author: SINTAAuthor) -> bool:
        """Save a single author to Supabase immediately."""
        if not self._supabase:
            logger.warning("Supabase not configured, skipping save")
            return False
        
        try:
            self._supabase.rpc("upsert_sinta_lecturer", {
                "p_sinta_id": author.sinta_id,
                "p_name": author.name,
                "p_nidn": author.nidn,
                "p_dept": author.dept,
                "p_h_index_scopus": author.h_index_scopus,
                "p_h_index_gscholar": author.h_index_gscholar,
                "p_publications": author.publications,
                "p_expertise_tags": author.expertise_tags,
                "p_researches": author.researches,
                "p_services": author.services,
                "p_iprs": author.iprs,
                "p_books": author.books
            }).execute()
            logger.debug(f"Saved to DB: {author.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to save {author.name} to DB: {e}")
            return False
    
    def scrape_all(
        self,
        limit: Optional[int] = None,
        resume: bool = False,
        dry_run: bool = False
    ) -> List[SINTAAuthor]:
        """
        Scrape all ITS authors from SINTA.
        
        Args:
            limit: Maximum authors to scrape (None for all)
            resume: Resume from checkpoint
            dry_run: If True, don't save to database
            
        Returns:
            List of SINTAAuthor objects
        """
        # ALWAYS sync with Supabase first to avoid re-scraping
        self.sync_with_supabase()
        
        if resume:
            self.load_checkpoint()
            if self.progress.is_paused:
                logger.info("Resuming from paused state...")
                self.progress.is_paused = False
                self.progress.pause_reason = None
        
        logger.info(f"Will skip {len(self.progress.processed_sinta_ids)} already processed authors")
        
        results = []
        
        try:
            self.start_browser()
            
            # Use cached author list on resume if available
            if resume and self.progress.cached_authors:
                logger.info(f"Using cached author list ({len(self.progress.cached_authors)} authors)")
                authors = self.progress.cached_authors
            else:
                # Fetch author list with retry (SINTA is often unstable)
                authors = None
                for fetch_attempt in range(10):  # 10 attempts
                    authors = self.fetch_author_list()
                    if authors:
                        break
                    wait_time = 60 if fetch_attempt < 5 else 120  # Longer wait after 5 failures
                    logger.warning(f"Fetch attempt {fetch_attempt + 1}/10 failed, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                
                if authors:
                    # Cache author list for future resumes
                    self.progress.cached_authors = authors
                    self.save_checkpoint()
            
            if not authors:
                logger.warning("No authors found to scrape")
                return results
            
            # Apply limit
            if limit:
                authors = authors[:limit]
            
            # Scrape each author
            pbar = tqdm(authors, desc="Scraping SINTA authors")
            
            for author_info in pbar:
                if self.progress.is_paused:
                    break
                
                sinta_id = author_info["sinta_id"]
                
                # Skip if already processed
                if sinta_id in self.progress.processed_sinta_ids:
                    continue
                
                pbar.set_postfix({
                    "current": author_info["name"][:20],
                    "success": self.progress.total_success
                })
                
                # Scrape author profile with retry logic
                author = None
                for retry_attempt in range(AUTHOR_RETRY_COUNT):
                    author = self.scrape_author_profile(sinta_id)
                    
                    if author:
                        break  # Success, exit retry loop
                    elif retry_attempt < AUTHOR_RETRY_COUNT - 1:
                        # Wait before retry (longer delay for recovery)
                        logger.info(f"Retrying {author_info['name'][:20]} (attempt {retry_attempt + 2}/{AUTHOR_RETRY_COUNT})...")
                        time.sleep(30)  # 30 second recovery delay
                
                if author:
                    # INCREMENTAL SAVE: Save to Supabase immediately (crash-safe)
                    if not dry_run:
                        save_success = self._save_author_to_db(author)
                        if save_success:
                            results.append(author)
                            self.progress.total_success += 1
                        else:
                            self.progress.total_failed += 1
                    else:
                        results.append(author)
                        self.progress.total_success += 1
                    
                    self.progress.processed_sinta_ids.append(sinta_id)
                    # Reset network error counter on success
                    self.progress.consecutive_network_errors = 0
                else:
                    self.progress.total_failed += 1
                
                # Check if paused due to network error
                if self.progress.is_paused:
                    break
                
                self.progress.total_processed += 1
                
                # Save checkpoint periodically
                if self.progress.total_processed % 10 == 0:
                    self.save_checkpoint()
                
                # Apply delay
                self._delay()
                
                # Cool-down every 30 authors with fresh context (30s pause)
                if self.progress.total_processed % 30 == 0 and self.progress.total_processed > 0:
                    logger.info(f"🧊 Cool-down at {self.progress.total_processed} authors - waiting 30s...")
                    time.sleep(30)  # 30 second cool-down (shorter)
                    self.rotate_context()  # Fresh context after cool-down
            
            pbar.close()
            
        finally:
            self.save_checkpoint()
            self.stop_browser()
        
        logger.info(
            f"Scraping complete: {self.progress.total_success} succeeded, "
            f"{self.progress.total_failed} failed"
        )
        
        return results
    
    def retry_failed_authors(self, dry_run: bool = False) -> List[SINTAAuthor]:
        """
        Retry only the failed authors from previous runs.
        Reads failed SINTA IDs from checkpoint and scrapes only those.
        """
        results = []
        
        # Load checkpoint
        self.load_checkpoint()
        
        failed_list = self.progress.failed_sinta_ids.copy()
        if not failed_list:
            print("✅ No failed authors to retry!")
            return results
        
        print(f"Found {len(failed_list)} failed authors to retry\n")
        
        try:
            self.start_browser()
            
            from tqdm import tqdm
            
            for failed_entry in tqdm(failed_list, desc="Retrying failed authors"):
                sinta_id = failed_entry["sinta_id"]
                
                # Scrape with retry
                author = None
                for retry_attempt in range(AUTHOR_RETRY_COUNT):
                    author = self.scrape_author_profile(sinta_id)
                    if author:
                        break
                    elif retry_attempt < AUTHOR_RETRY_COUNT - 1:
                        time.sleep(30)
                
                if author:
                    if not dry_run:
                        save_success = self._save_author_to_db(author)
                        if save_success:
                            results.append(author)
                            # Remove from failed list on success
                            self.progress.failed_sinta_ids = [
                                f for f in self.progress.failed_sinta_ids 
                                if f["sinta_id"] != sinta_id
                            ]
                            self.progress.total_success += 1
                            if sinta_id not in self.progress.processed_sinta_ids:
                                self.progress.processed_sinta_ids.append(sinta_id)
                    else:
                        results.append(author)
                
                self._delay()
            
            self.save_checkpoint()
            
        finally:
            self.stop_browser()
        
        print(f"\n✅ Retry complete: {len(results)} succeeded")
        return results


# ============================================================================
# Database Integration
# ============================================================================

def save_to_supabase(authors: List[SINTAAuthor]) -> Dict[str, int]:
    """
    Save scraped authors to Supabase.
    
    Args:
        authors: List of SINTAAuthor objects
        
    Returns:
        Stats dict with success/failed counts
    """
    from supabase import create_client, Client
    
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    
    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.error("Supabase credentials not configured")
        return {"success": 0, "failed": len(authors)}
    
    client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    stats = {"success": 0, "failed": 0}
    
    for author in tqdm(authors, desc="Saving to Supabase"):
        try:
            # Use upsert RPC function
            client.rpc("upsert_sinta_lecturer", {
                "p_sinta_id": author.sinta_id,
                "p_name": author.name,
                "p_nidn": author.nidn,
                "p_dept": author.dept,
                "p_h_index_scopus": author.h_index_scopus,
                "p_h_index_gscholar": author.h_index_gscholar,
                "p_publications": author.publications
            }).execute()
            
            stats["success"] += 1
            
        except Exception as e:
            logger.error(f"Failed to save {author.name}: {e}")
            stats["failed"] += 1
    
    return stats


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    """CLI entry point for SINTA scraper."""
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
    
    parser = argparse.ArgumentParser(description="Scrape lecturer data from SINTA")
    parser.add_argument("--limit", type=int, help="Maximum authors to scrape")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--retry-failed", action="store_true", help="Retry only failed authors from last run")
    parser.add_argument("--no-headless", action="store_true", help="Run browser in visible mode")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    parser.add_argument("--dry-run", action="store_true", help="Scrape but don't save to database")
    parser.add_argument("--delay-min", type=float, default=DELAY_MIN, help="Minimum delay (seconds)")
    parser.add_argument("--delay-max", type=float, default=DELAY_MAX, help="Maximum delay (seconds)")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("\n🎓 SINTA Scraper - ITS Intelligence Hub v2\n")
    print(f"   Target: {ITS_AUTHORS_URL}")
    print(f"   Delay: {args.delay_min}-{args.delay_max}s")
    print(f"   Mode: {'Visible' if args.no_headless else 'Headless'}")
    print()
    
    # Initialize scraper
    scraper = SINTAScraper(
        headless=not args.no_headless,
        delay_min=args.delay_min,
        delay_max=args.delay_max
    )
    
    # Handle --retry-failed mode
    if args.retry_failed:
        print("🔄 Retry-failed mode: Only retrying previously failed authors\n")
        authors = scraper.retry_failed_authors(dry_run=args.dry_run)
    else:
        # Reset pause state on resume
        if args.resume and scraper.progress.is_paused:
            scraper.progress.is_paused = False
            scraper.progress.pause_reason = None
            scraper.progress.consecutive_network_errors = 0
            scraper.save_checkpoint()
            print("📝 Checkpoint pause state reset, continuing...\n")
        
        # Run scraping (data saved incrementally now, not at the end)
        authors = scraper.scrape_all(limit=args.limit, resume=args.resume, dry_run=args.dry_run)
    
    print(f"\n📊 Scraping Results:")
    print(f"   Total processed: {scraper.progress.total_processed}")
    print(f"   Succeeded: {scraper.progress.total_success}")
    print(f"   Failed: {scraper.progress.total_failed}")
    
    if args.dry_run:
        print("\n⚠️  Dry run mode - data not saved to database")
        # Print sample output
        if authors:
            print("\n📝 Sample output (first 3 authors):")
            for author in authors[:3]:
                print(f"\n   {author.name}")
                print(f"   SINTA ID: {author.sinta_id}")
                print(f"   NIDN: {author.nidn or 'N/A'}")
                print(f"   Dept: {author.dept or 'N/A'}")
                print(f"   H-Index (Scopus): {author.h_index_scopus}")
                print(f"   Publications: {len(author.publications)}")
    else:
        print("\n✅ Data saved to Supabase incrementally during scraping")
    
    print("\n✅ Done!\n")


if __name__ == "__main__":
    main()
