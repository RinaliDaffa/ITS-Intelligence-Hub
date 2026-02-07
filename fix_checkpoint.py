import json
from pathlib import Path

# Read the checkpoint with proper structure
checkpoint_path = Path("data/sinta_checkpoint.json")

# Read the file as text first
with open(checkpoint_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find processed_sinta_ids
processed_ids = []
in_processed = False
for line in lines:
    if '"processed_sinta_ids"' in line:
        in_processed = True
        continue
    if in_processed:
        if ']' in line and '"' not in line:
            in_processed = False
            break
        # Extract ID from line like '    "29555",'
        line = line.strip().rstrip(',')
        if line.startswith('"') and line.endswith('"'):
            processed_ids.append(line.strip('"'))

print(f"Found {len(processed_ids)} processed IDs")

# Create clean checkpoint
clean_checkpoint = {
    "total_discovered": 990,
    "total_processed": len(processed_ids),
    "total_success": len(processed_ids),
    "total_failed": 0,
    "current_page": 1,
    "processed_sinta_ids": processed_ids,
    "failed_sinta_ids": [],
    "cached_authors": [],
    "started_at": "2026-02-05T22:56:03.550793",
    "updated_at": "2026-02-06T20:34:00.000000",
    "is_paused": False,
    "pause_reason": None,
    "consecutive_network_errors": 0
}

# Write clean checkpoint
with open(checkpoint_path, 'w', encoding='utf-8') as f:
    json.dump(clean_checkpoint, f, indent=2, ensure_ascii=False)

print(f"Checkpoint cleaned! {len(processed_ids)} processed IDs preserved.")
print("cached_authors cleared, ready to fetch fresh author list.")
