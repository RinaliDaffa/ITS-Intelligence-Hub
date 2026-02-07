"""
Quick test script to verify SINTA publication extraction works correctly.
This bypasses the full author scraping and directly tests the publication extraction.
"""
import os
import sys
import logging
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def test_publication_extraction():
    """Test that publication extraction works for a known author."""
    from src.sinta_scraper import SINTAScraper
    
    # RIYANARTO SARNO - well-known ITS professor with many publications
    TEST_SINTA_ID = "29555"
    
    print("=" * 60)
    print("SINTA Publication Extraction Test")
    print("=" * 60)
    print(f"Test author SINTA ID: {TEST_SINTA_ID}")
    print()
    
    scraper = SINTAScraper(
        headless=True,
        delay_min=2,  # Faster for testing
        delay_max=3
    )
    
    try:
        scraper.start_browser()
        
        # Scrape a single author
        print(f"Scraping author profile...")
        author = scraper.scrape_author_profile(TEST_SINTA_ID)
        
        if author:
            print()
            print("[SUCCESS] Author scraped successfully!")
            print(f"   Name: {author.name}")
            print(f"   SINTA ID: {author.sinta_id}")
            print(f"   Department: {author.dept or 'N/A'}")
            print(f"   H-Index (Scopus): {author.h_index_scopus}")
            print(f"   H-Index (GScholar): {author.h_index_gscholar}")
            print(f"   Publications count: {len(author.publications)}")
            print()
            
            if author.publications:
                print("Sample publications extracted:")
                for i, pub in enumerate(author.publications[:5], 1):
                    title = pub.get('title', 'N/A')[:80]
                    # Remove non-ASCII chars for safe printing
                    title = title.encode('ascii', 'ignore').decode('ascii')
                    print(f"   {i}. {title}...")
                    print(f"      Source: {pub.get('source', 'N/A')}, Year: {pub.get('year', 'N/A')}")
                
                print()
                print("=" * 60)
                print("SUCCESS: Publications are being extracted correctly!")
                print(f"   Total publications: {len(author.publications)}")
                print("=" * 60)
                return True
            else:
                print()
                print("=" * 60)
                print("FAILED: No publications extracted!")
                print("   The fix may not be working correctly.")
                print("=" * 60)
                return False
        else:
            print("[FAILED] Failed to scrape author profile")
            return False
            
    finally:
        scraper.stop_browser()

if __name__ == "__main__":
    success = test_publication_extraction()
    sys.exit(0 if success else 1)
