"""Quick inspection script for supervisor data."""
from supabase import create_client
import os
from dotenv import load_dotenv

load_dotenv()

client = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))

print("\n=== SUPERVISOR DATA INSPECTION ===\n")

# Get sample records
resp = client.table('researches').select('id, title, supervisors, abstract').limit(10).execute()

for r in resp.data:
    sup = r.get('supervisors')
    print(f"ID: {r['id'][:8]}...")
    print(f"  Title: {r.get('title', '')[:50]}...")
    print(f"  Supervisors: {repr(sup)}")
    print(f"  Type: {type(sup)}, Length: {len(sup) if sup else 0}")
    print()

# Count stats
print("\n=== STATISTICS ===\n")

# Total records
total = client.table('researches').select('id', count='exact').execute()
print(f"Total records: {total.count}")

# Records with non-null supervisors
non_null = client.table('researches').select('id', count='exact').not_.is_('supervisors', 'null').execute()
print(f"Non-NULL supervisors: {non_null.count}")

# Check for empty arrays by fetching and filtering
sample = client.table('researches').select('supervisors').limit(500).execute()
non_empty = sum(1 for r in sample.data if r.get('supervisors') and len(r['supervisors']) > 0)
print(f"Non-empty arrays (in first 500): {non_empty}")
