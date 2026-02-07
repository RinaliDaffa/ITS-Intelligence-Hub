"""
Quick test script for comprehensive SINTA scraper.
Tests single author extraction with ALL new fields.

Run: python test_comprehensive_scraper.py
"""
import logging
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

from src.sinta_scraper import SINTAScraper

print("=" * 60)
print("COMPREHENSIVE SINTA SCRAPER - QUICK TEST")
print("=" * 60)

# Test with well-known ITS professor
TEST_ID = "29555"  # Riyanarto Sarno

scraper = SINTAScraper(headless=True, delay_min=2, delay_max=4)

try:
    scraper.start_browser()
    print(f"\nScraping author ID: {TEST_ID}...")
    
    author = scraper.scrape_author_profile(TEST_ID)
    
    if author:
        print(f"\n[OK] Name: {author.name}")
        print(f"[OK] SINTA ID: {author.sinta_id}")
        print(f"[OK] NIDN: {author.nidn}")
        print(f"[OK] Dept: {author.dept}")
        print(f"[{'OK' if author.h_index_scopus > 0 else '??'}] H-Index Scopus: {author.h_index_scopus}")
        print(f"[{'OK' if author.h_index_gscholar > 0 else '??'}] H-Index GScholar: {author.h_index_gscholar}")
        print(f"[{'OK' if author.expertise_tags else '??'}] Expertise Tags: {author.expertise_tags}")
        print(f"[{'OK' if author.publications else '??'}] Publications: {len(author.publications)} items")
        print(f"[{'OK' if author.researches else '--'}] Researches: {len(author.researches)} items")
        print(f"[{'OK' if author.services else '--'}] Services: {len(author.services)} items")
        print(f"[{'OK' if author.iprs else '--'}] IPRs: {len(author.iprs)} items")
        print(f"[{'OK' if author.books else '--'}] Books: {len(author.books)} items")
        
        # Test DB save
        print("\nTesting database save...")
        if scraper._save_author_to_db(author):
            print("[OK] Database save SUCCESS")
        else:
            print("[FAIL] Database save FAILED")
    else:
        print("[FAIL] Could not scrape author - check network/SINTA availability")
        
finally:
    scraper.stop_browser()

print("\n" + "=" * 60)
print("Test complete! If all [OK], you can run full scrape:")
print("python -m src.sinta_scraper --resume")
print("=" * 60)
