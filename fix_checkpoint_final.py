"""
Fix checkpoint by fetching all sinta_ids from Supabase
"""
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

supabase = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))

print("Fetching all sinta_ids from Supabase...")

# Get all sinta_ids
all_lecturers = []
offset = 0
batch_size = 1000

while True:
    result = supabase.table('lecturers').select('sinta_id').range(offset, offset + batch_size - 1).execute()
    if not result.data:
        break
    all_lecturers.extend(result.data)
    offset += batch_size
    if len(result.data) < batch_size:
        break

sinta_ids = [str(l["sinta_id"]) for l in all_lecturers if l.get("sinta_id")]
print(f"Found {len(sinta_ids)} sinta_ids in Supabase")

# Load checkpoint
checkpoint_path = Path('data/sinta_checkpoint.json')
with open(checkpoint_path, 'r', encoding='utf-8') as f:
    checkpoint = json.load(f)

# Merge existing with Supabase IDs (avoid duplicates)
existing_ids = set(checkpoint.get('processed_sinta_ids', []))
all_ids = list(set(sinta_ids) | existing_ids)

print(f"Existing in checkpoint: {len(existing_ids)}")
print(f"After merge: {len(all_ids)}")

# Update checkpoint
checkpoint['processed_sinta_ids'] = all_ids
checkpoint['total_processed'] = len(all_ids)
checkpoint['total_success'] = len(all_ids)

# Save
with open(checkpoint_path, 'w', encoding='utf-8') as f:
    json.dump(checkpoint, f, indent=2, ensure_ascii=False)

print(f"\nCheckpoint fixed!")
print(f"Total processed_sinta_ids: {len(all_ids)}")
print(f"Cached authors: {len(checkpoint.get('cached_authors', []))}")
remaining = len(checkpoint.get('cached_authors', [])) - len(all_ids)
print(f"Remaining to scrape: {max(0, remaining)}")
