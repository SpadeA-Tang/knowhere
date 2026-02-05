#!/usr/bin/env python3
"""
Analyze ground truth document lengths for LoTTE queries.

Usage:
    cd build && python ../scripts/analyze_gt_doc_length.py
    python scripts/analyze_gt_doc_length.py --docs build/lotte_science_docs.jsonl --queries build/lotte_science_queries.jsonl
"""

import argparse
import json
import numpy as np
from collections import defaultdict

parser = argparse.ArgumentParser(description="Analyze GT document lengths")
parser.add_argument("--docs", type=str, default="lotte_science_docs.jsonl",
                    help="Path to documents JSONL file")
parser.add_argument("--queries", type=str, default="lotte_science_queries.jsonl",
                    help="Path to queries JSONL file")
args = parser.parse_args()

# Load documents
print("Loading documents...")
docs = []
with open(args.docs, "r") as f:
    for line in f:
        doc = json.loads(line)
        docs.append(len(doc["chunks"]))  # number of vectors per doc

print(f"Loaded {len(docs)} documents")
print(f"Doc length stats: min={min(docs)}, max={max(docs)}, avg={np.mean(docs):.1f}, median={np.median(docs):.0f}")

# Load queries
print("\nLoading queries...")
queries = []
with open(args.queries, "r") as f:
    for line in f:
        query = json.loads(line)
        queries.append(len(query["chunks"]))

print(f"Loaded {len(queries)} queries")
print(f"Query length: {queries[0]} vectors (fixed)")

# We need to compute ground truth - brute force MaxSim
# This is expensive, so let's load pre-computed if available or compute a sample

print("\n" + "="*60)
print("To get GT doc lengths, we need to run the test and capture GT IDs")
print("Adding analysis to test output would be more efficient")
print("="*60)

# For now, let's analyze the document length distribution
print("\n=== Document Length Distribution ===")
buckets = [0, 50, 100, 200, 500, 1000, 2000, 3000]
for i in range(len(buckets)-1):
    count = sum(1 for d in docs if buckets[i] < d <= buckets[i+1])
    pct = 100 * count / len(docs)
    print(f"  {buckets[i]+1:4d}-{buckets[i+1]:4d}: {count:5d} docs ({pct:5.1f}%)")

count = sum(1 for d in docs if d > buckets[-1])
if count > 0:
    print(f"  >{buckets[-1]:4d}: {count:5d} docs ({100*count/len(docs):5.1f}%)")

# Analyze short vs long documents
short_docs = [i for i, d in enumerate(docs) if d <= 50]
long_docs = [i for i, d in enumerate(docs) if d > 500]

print(f"\nShort docs (<=50 vectors): {len(short_docs)} ({100*len(short_docs)/len(docs):.1f}%)")
print(f"Long docs (>500 vectors): {len(long_docs)} ({100*len(long_docs)/len(docs):.1f}%)")
