"""Merge per-shard JSON files into final int_counts.json and float_counts.json."""

import os
import json
import time
from collections import Counter

SHARD_DIR = "shards"

int_total = Counter()
float_total = Counter()
total_docs = 0

t0 = time.time()
shard_files = sorted(f for f in os.listdir(SHARD_DIR) if f.endswith(".json"))
print(f"Merging {len(shard_files)} shard files...")

for fname in shard_files:
    with open(os.path.join(SHARD_DIR, fname)) as f:
        data = json.load(f)
    int_total.update(data["int_counts"])
    float_total.update(data["float_counts"])
    total_docs += data["doc_count"]
    print(f"  {fname}: {data['doc_count']:,} docs", flush=True)

elapsed = time.time() - t0
print(f"\nMerged in {elapsed:.1f}s")
print(f"Total docs: {total_docs:,}")
print(f"Unique integers: {len(int_total):,}  |  Unique floats: {len(float_total):,}")
print(f"Total integer occurrences: {sum(int_total.values()):,}")
print(f"Total float occurrences: {sum(float_total.values()):,}")

print("\nTop 30 integers:")
for val, cnt in int_total.most_common(30):
    print(f"  {val:>15s}  {cnt:,}")

print("\nTop 30 floats:")
for val, cnt in float_total.most_common(30):
    print(f"  {val:>15s}  {cnt:,}")

with open("int_counts.json", "w") as f:
    json.dump(int_total.most_common(), f)
with open("float_counts.json", "w") as f:
    json.dump(float_total.most_common(), f)

print("\nSaved int_counts.json and float_counts.json")
