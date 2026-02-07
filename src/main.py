"""
ITS Intelligence Hub v2 - Main Orchestrator
Command-line interface for the research data scraping pipeline.
"""

import os
import sys
import signal
import argparse
import logging
import csv
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Set
import json

from tqdm import tqdm
import colorlog

from .scraper import ITSScraper, FTEIC_DIVISIONS
from .cleaner import DataCleaner
from .database import get_database_client


# Setup colored logging
def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Configure colored console logging and optional file logging."""
    
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s [%(levelname)s]%(reset)s %(message)s",
        datefmt="%H:%M:%S",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        }
    ))
    
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    root_logger.addHandler(handler)
    
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        ))
        root_logger.addHandler(file_handler)


logger = logging.getLogger(__name__)


class GracefulShutdown:
    """Handle graceful shutdown on SIGINT/SIGTERM."""
    
    def __init__(self) -> None:
        self.shutdown_requested = False
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame) -> None:
        if self.shutdown_requested:
            logger.warning("Force shutdown requested, exiting...")
            sys.exit(1)
        logger.warning("Shutdown requested, finishing current operation...")
        self.shutdown_requested = True


class DataBuffer:
    """Buffer for auto-saving scraped data to CSV."""
    
    def __init__(self, filepath: str, batch_size: int = 20) -> None:
        self.filepath = filepath
        self.batch_size = batch_size
        self.buffer: List[dict] = []
        self.total_saved = 0
        
        # Create parent directory if needed
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    def add(self, record: dict) -> None:
        """Add record to buffer, auto-save when batch size reached."""
        self.buffer.append(record)
        
        if len(self.buffer) >= self.batch_size:
            self.flush()
    
    def flush(self) -> None:
        """Write buffer to CSV file."""
        if not self.buffer:
            return
        
        file_exists = os.path.exists(self.filepath)
        
        try:
            with open(self.filepath, "a", newline="", encoding="utf-8") as f:
                fieldnames = [
                    "title", "author", "abstract", "supervisors", 
                    "degree", "year", "dept", "url", "data_confidence"
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                
                if not file_exists:
                    writer.writeheader()
                
                for record in self.buffer:
                    # Convert supervisors list to string
                    record_copy = record.copy()
                    if isinstance(record_copy.get("supervisors"), list):
                        record_copy["supervisors"] = "; ".join(record_copy["supervisors"])
                    writer.writerow(record_copy)
                
                self.total_saved += len(self.buffer)
                logger.info(f"Auto-saved {len(self.buffer)} records to buffer (total: {self.total_saved})")
                self.buffer = []
                
        except Exception as e:
            logger.error(f"Failed to write buffer to CSV: {e}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="ITS Intelligence Hub - Academic Research Data Scraper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.main --degree S1 --limit 100
  python -m src.main --division Informatics_S1 --resume
  python -m src.main --dry-run --limit 10 --verbose
        """
    )
    
    # Scope filters
    parser.add_argument(
        "--degree",
        type=str,
        help="Filter by degree level (S1, S2, S3, or comma-separated)"
    )
    parser.add_argument(
        "--division",
        type=str,
        help="Specific division(s) to scrape (comma-separated, see available divisions below)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of records to scrape"
    )
    
    # Resume and checkpointing
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint"
    )
    parser.add_argument(
        "--checkpoint-file",
        type=str,
        default="data/checkpoint.json",
        help="Path to checkpoint file (default: data/checkpoint.json)"
    )
    parser.add_argument(
        "--buffer-file",
        type=str,
        default="data/buffer_data.csv",
        help="Path to buffer CSV file (default: data/buffer_data.csv)"
    )
    
    # Database options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without database insertion (for testing)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip URLs already in database (default: True)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Records per batch for buffer save and DB insert (default: 20)"
    )
    
    # Logging
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        help="Path to log file"
    )
    
    # List available divisions
    parser.add_argument(
        "--list-divisions",
        action="store_true",
        help="List available divisions and exit"
    )
    
    return parser.parse_args()


def list_divisions() -> None:
    """Print available divisions and exit."""
    print("\nAvailable FTEIC Divisions (Pilot Scope):\n")
    for name, url in FTEIC_DIVISIONS.items():
        print(f"  {name}")
        print(f"    URL: {url}\n")
    print("Use --division <name> to specify which division(s) to scrape.")
    print("Multiple divisions can be comma-separated: --division Informatics_S1,Informatics_S2")


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    # Handle --list-divisions
    if args.list_divisions:
        list_divisions()
        return 0
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else os.getenv("LOG_LEVEL", "INFO")
    setup_logging(log_level, args.log_file)
    
    logger.info("=" * 60)
    logger.info("ITS Intelligence Hub v2 - Data Ingestion Pipeline")
    logger.info("=" * 60)
    
    # Initialize graceful shutdown handler
    shutdown = GracefulShutdown()
    
    # Initialize components
    scraper = ITSScraper()
    db = get_database_client(dry_run=args.dry_run, batch_size=args.batch_size)
    buffer = DataBuffer(args.buffer_file, batch_size=args.batch_size)
    
    # Load checkpoint if resuming
    if args.resume:
        if scraper.load_checkpoint(args.checkpoint_file):
            logger.info(f"Resuming from checkpoint: {scraper.progress.total_processed} already processed")
        else:
            logger.warning("No checkpoint found, starting fresh")
    
    # Determine divisions to scrape
    divisions = None
    if args.division:
        divisions = [d.strip() for d in args.division.split(",")]
        invalid = [d for d in divisions if d not in FTEIC_DIVISIONS]
        if invalid:
            logger.error(f"Unknown division(s): {invalid}")
            logger.info("Use --list-divisions to see available options")
            return 1
    
    # Filter by degree if specified
    if args.degree and not divisions:
        degree_filter = [d.strip().upper() for d in args.degree.split(",")]
        divisions = [
            name for name in FTEIC_DIVISIONS.keys()
            if any(deg in name.upper() for deg in degree_filter)
        ]
        if not divisions:
            logger.warning(f"No divisions match degree filter: {args.degree}")
            logger.info("Available degrees in division names: S1, S2, S3")
    
    # Get existing URLs to skip
    skip_urls: Set[str] = set()
    if args.skip_existing and not args.dry_run:
        logger.info("Fetching existing URLs from database...")
        # This would require knowing URLs in advance; skip for pilot
        # In practice, we check per-record or batch
    
    # Add previously processed URLs from checkpoint
    if args.resume:
        skip_urls.update(scraper.progress.processed_urls)
    
    logger.info(f"Starting scrape with config:")
    logger.info(f"  Divisions: {divisions or 'all FTEIC'}")
    logger.info(f"  Limit: {args.limit or 'unlimited'}")
    logger.info(f"  Dry-run: {args.dry_run}")
    logger.info(f"  Checkpoint: {args.checkpoint_file}")
    
    # Main scraping loop
    records_buffer: List[dict] = []
    
    try:
        with tqdm(desc="Scraping", unit="records") as pbar:
            for record in scraper.scrape_fteic_pilot(
                divisions=divisions,
                limit=args.limit,
                skip_existing=skip_urls
            ):
                if shutdown.shutdown_requested:
                    logger.warning("Shutdown requested, stopping scrape...")
                    break
                
                # Add to buffer for auto-save
                buffer.add(record)
                
                # Add to DB batch
                records_buffer.append(record)
                
                # Batch insert to database
                if len(records_buffer) >= args.batch_size:
                    stats = db.upsert_batch(records_buffer)
                    logger.info(f"DB batch: {stats}")
                    records_buffer = []
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({
                    "success": scraper.progress.total_success,
                    "failed": scraper.progress.total_failed,
                    "confidence": f"{record.get('data_confidence', 0):.2f}"
                })
                
                # Periodic checkpoint save
                if scraper.progress.total_processed % 50 == 0:
                    scraper.save_checkpoint(args.checkpoint_file)
        
        # Final batch insert
        if records_buffer:
            stats = db.upsert_batch(records_buffer)
            logger.info(f"Final DB batch: {stats}")
        
        # Final buffer flush
        buffer.flush()
        
    except Exception as e:
        logger.exception(f"Unexpected error during scraping: {e}")
        
    finally:
        # Always save checkpoint on exit
        scraper.save_checkpoint(args.checkpoint_file)
    
    # Print summary
    logger.info("=" * 60)
    logger.info("Scraping Complete - Summary")
    logger.info("=" * 60)
    logger.info(f"  Discovered: {scraper.progress.total_discovered}")
    logger.info(f"  Processed:  {scraper.progress.total_processed}")
    logger.info(f"  Successful: {scraper.progress.total_success}")
    logger.info(f"  Failed:     {scraper.progress.total_failed}")
    logger.info(f"  DB Stats:   {db.stats}")
    
    if scraper.progress.failed_urls:
        logger.warning(f"  Failed URLs saved to checkpoint for retry")
        for fail in scraper.progress.failed_urls[:5]:
            logger.warning(f"    - {fail['url']}: {fail['error']}")
        if len(scraper.progress.failed_urls) > 5:
            logger.warning(f"    ... and {len(scraper.progress.failed_urls) - 5} more")
    
    return 0 if scraper.progress.total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
