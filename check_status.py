"""
Check Supabase data completeness
"""
import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

supabase = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))

print("=" * 50)
print("📊 SINTA SCRAPING STATUS CHECK")
print("=" * 50)

# Count total lecturers
result = supabase.table('lecturers').select('id', count='exact').execute()
print(f"\n✅ Total dosen di Supabase: {result.count}")

# Count with sinta_id
valid_sinta = supabase.table('lecturers').select('sinta_id', count='exact').neq('sinta_id', 'null').execute()
print(f"✅ Dosen dengan sinta_id valid: {valid_sinta.count}")

# Sample lecturers
sample = supabase.table('lecturers').select('name, sinta_id, department').limit(5).execute()
print("\n📋 Sample dosen:")
for l in sample.data:
    name = l.get("name", "N/A")
    dept = l.get("department", "N/A")
    print(f"  - {name} ({dept})")

print("\n" + "=" * 50)
print("SINTA ITS memiliki ~128 halaman x 10 = ~1,280 dosen")
print(f"Kamu sudah punya: {result.count} dosen")
print(f"Persentase: {(result.count / 1280) * 100:.1f}%")
print("=" * 50)
