"""
Check and fix checkpoint from Supabase data
"""
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

# Connect to Supabase
supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

print("🔍 Checking Supabase lecturers table...")

# Count total lecturers
count_result = supabase.table("lecturers").select("sinta_id", count="exact").execute()
total_count = count_result.count

print(f"✅ Total dosen di Supabase: {total_count}")

# Get all sinta_ids
all_lecturers = []
offset = 0
batch_size = 1000

while True:
    result = supabase.table("lecturers").select("sinta_id, name").range(offset, offset + batch_size - 1).execute()
    if not result.data:
        break
    all_lecturers.extend(result.data)
    offset += batch_size
    if len(result.data) < batch_size:
        break

print(f"📋 Fetched {len(all_lecturers)} lecturers from Supabase")

# Get sinta_ids
sinta_ids = [str(l["sinta_id"]) for l in all_lecturers if l.get("sinta_id")]
print(f"📋 Found {len(sinta_ids)} valid sinta_ids")

# Load existing checkpoint
checkpoint_path = Path("data/sinta_checkpoint.json")
with open(checkpoint_path, "r", encoding="utf-8") as f:
    checkpoint = json.load(f)

# Get cached_authors from checkpoint
cached_authors = checkpoint.get("cached_authors", [])
print(f"📋 Cached authors in checkpoint: {len(cached_authors)}")

# Update checkpoint with correct processed_sinta_ids
checkpoint["processed_sinta_ids"] = sinta_ids
checkpoint["total_processed"] = len(sinta_ids)
checkpoint["total_success"] = len(sinta_ids)
checkpoint["total_failed"] = 0
checkpoint["failed_sinta_ids"] = []

# Save updated checkpoint
with open(checkpoint_path, "w", encoding="utf-8") as f:
    json.dump(checkpoint, f, indent=2, ensure_ascii=False)

print(f"\n✅ Checkpoint fixed!")
print(f"   - processed_sinta_ids: {len(sinta_ids)}")
print(f"   - cached_authors: {len(cached_authors)}")
print(f"\n📊 Summary:")
print(f"   - Dosen di Supabase: {total_count}")
print(f"   - Akan di-skip saat resume: {len(sinta_ids)}")
print(f"   - Sisa yang perlu di-scrape: {len(cached_authors) - len(sinta_ids)} (dari cached)")
