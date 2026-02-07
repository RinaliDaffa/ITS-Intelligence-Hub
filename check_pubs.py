"""Quick check saved publications in database."""
from dotenv import load_dotenv
load_dotenv()
import os
from supabase import create_client

sb = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))

# Check first scraped author
r = sb.table('lecturers').select('name, sinta_publications').eq('sinta_id', '6005015').execute()
d = r.data[0] if r.data else {}

print(f"Name: {d.get('name')}")
pubs = d.get('sinta_publications') or []
print(f"Total publications: {len(pubs)}")
years = sorted(set(p.get('year') for p in pubs if p.get('year')))
print(f"Years: {years}")
sources = set(p.get('source') for p in pubs)
print(f"Sources: {sources}")
