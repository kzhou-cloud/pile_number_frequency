"""Stream The Pile in parallel with periodic checkpointing.

Resumable: re-run to pick up from last checkpoint. Completed shards are skipped.
"""

import os
import re
import json
import time
import multiprocessing as mp
from collections import Counter
from datasets import load_dataset

NUM_SHARDS = 30
SHARD_DIR = "shards"
CHECKPOINT_EVERY = 100_000
MAX_WORKERS = 20

NUMBER_RE = re.compile(
    r'\b(\d{1,3}(?:,\d{3})*\.\d+)\b'
    r'|\b(\d{1,3}(?:,\d{3})+)\b'
    r'|\b(\d+\.\d+)\b'
    r'|\b(\d+)\b'
)


def save_checkpoint(shard_idx, int_counts, float_counts, doc_count):
    path = os.path.join(SHARD_DIR, f"shard_{shard_idx:02d}.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"int_counts": int_counts, "float_counts": float_counts, "doc_count": doc_count}, f)
    os.replace(tmp, path)


def load_checkpoint(shard_idx):
    path = os.path.join(SHARD_DIR, f"shard_{shard_idx:02d}.json")
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        return Counter(data["int_counts"]), Counter(data["float_counts"]), data["doc_count"]
    return Counter(), Counter(), 0


def process_shard(shard_idx):
    done_file = os.path.join(SHARD_DIR, f"shard_{shard_idx:02d}.done")
    if os.path.exists(done_file):
        return 0

    int_counts, float_counts, start_doc = load_checkpoint(shard_idx)
    if start_doc > 0:
        print(f"  shard {shard_idx:2d}: resuming from doc {start_doc:,}", flush=True)

    ds = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
    shard = ds.shard(num_shards=NUM_SHARDS, index=shard_idx)

    doc_count = 0
    for example in shard:
        doc_count += 1
        if doc_count <= start_doc:
            continue
        for m in NUMBER_RE.finditer(example["text"]):
            if m.group(1):
                float_counts[m.group(1).replace(",", "")] += 1
            elif m.group(2):
                int_counts[m.group(2).replace(",", "")] += 1
            elif m.group(3):
                float_counts[m.group(3)] += 1
            else:
                int_counts[m.group(4)] += 1
        if doc_count % CHECKPOINT_EVERY == 0:
            save_checkpoint(shard_idx, int_counts, float_counts, doc_count)
            print(f"  shard {shard_idx:2d}: {doc_count:,} docs (saved)", flush=True)
        elif doc_count % 50_000 == 0:
            print(f"  shard {shard_idx:2d}: {doc_count:,} docs", flush=True)

    save_checkpoint(shard_idx, int_counts, float_counts, doc_count)
    with open(done_file, "w") as f:
        f.write(str(doc_count))
    print(f"  shard {shard_idx:2d}: DONE — {doc_count:,} docs", flush=True)
    return doc_count


if __name__ == "__main__":
    os.makedirs(SHARD_DIR, exist_ok=True)

    done = [i for i in range(NUM_SHARDS)
            if os.path.exists(os.path.join(SHARD_DIR, f"shard_{i:02d}.done"))]
    remaining = [i for i in range(NUM_SHARDS)
                 if not os.path.exists(os.path.join(SHARD_DIR, f"shard_{i:02d}.done"))]

    print(f"{len(done)}/{NUM_SHARDS} shards done, {len(remaining)} remaining")
    if not remaining:
        print("All shards complete. Run count_results.py to aggregate.")
        raise SystemExit(0)

    num_workers = min(len(remaining), MAX_WORKERS)
    print(f"Processing {len(remaining)} shards with {num_workers} workers...\n")

    t0 = time.time()
    with mp.Pool(num_workers) as pool:
        counts = pool.map(process_shard, remaining)

    elapsed = time.time() - t0
    total = sum(counts)
    print(f"\nDone: {total:,} docs in {elapsed:.1f}s ({total/max(elapsed,1):.0f} docs/s)")
    print("Run count_results.py to aggregate results.")
