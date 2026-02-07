"""Quick inspection of abstract content."""
from supabase import create_client
import os
from dotenv import load_dotenv

load_dotenv()

client = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))

print("\n=== ABSTRACT CONTENT INSPECTION ===\n")

# Get sample records with abstracts
resp = client.table('researches').select('id, title, abstract').limit(5).execute()

for i, r in enumerate(resp.data, 1):
    print(f"--- Record {i} ---")
    print(f"Title: {r.get('title', '')[:80]}...")
    abstract = r.get('abstract', '')
    print(f"Abstract length: {len(abstract) if abstract else 0}")
    print(f"Abstract (first 500 chars):")
    print(abstract[:500] if abstract else "(empty)")
    print()
    
    # Check for supervisor keywords
    if abstract:
        keywords = ['pembimbing', 'supervisor', 'advisor', 'dosen']
        found = [kw for kw in keywords if kw.lower() in abstract.lower()]
        print(f"Keywords found: {found if found else 'NONE'}")
    print("\n" + "="*60 + "\n")
