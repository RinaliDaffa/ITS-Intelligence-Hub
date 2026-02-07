"""Test PDF extraction from an actual thesis."""
import os
import re
import requests
import fitz
from dotenv import load_dotenv

load_dotenv()

# Test with a known PDF URL
pdf_url = "http://repository.its.ac.id/111537/1/5025201104-Undergraduate_Thesis.pdf"

print(f"Fetching PDF from: {pdf_url}")

session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0 Chrome/120.0.0.0"})

# Try to download first part of PDF
resp = session.get(pdf_url, timeout=30, stream=True)
print(f"Status: {resp.status_code}")
print(f"Content-Type: {resp.headers.get('content-type', 'unknown')}")

if resp.status_code == 200:
    # Read first 500KB
    pdf_bytes = resp.raw.read(500 * 1024)
    print(f"Downloaded: {len(pdf_bytes)} bytes")
    
    # Try to open and extract text
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        print(f"PDF pages (loaded): {len(doc)}")
        
        # Extract text from first 5 pages
        for i in range(min(5, len(doc))):
            page = doc[i]
            text = page.get_text()
            print(f"\n--- Page {i+1} (first 1000 chars) ---")
            print(text[:1000] if text else "(empty)")
            
            # Check for supervisor keywords
            keywords = ['pembimbing', 'supervisor', 'advisor', 'NIP', 'dosen']
            found = [kw for kw in keywords if kw.lower() in text.lower()]
            print(f"Keywords found: {found}")
            
        doc.close()
    except Exception as e:
        print(f"PDF parsing error: {e}")
else:
    print(f"Failed to download PDF")
