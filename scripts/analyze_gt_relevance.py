#!/usr/bin/env python3
"""
Analyze query-document relevance for LoTTE Ground Truth.
Extract sample queries and their top GT documents to verify relevance.

Usage:
    cd build && python ../scripts/analyze_gt_relevance.py
    python scripts/analyze_gt_relevance.py --docs build/lotte_science_docs.jsonl --queries build/lotte_science_queries.jsonl
"""

import argparse
import json
import numpy as np
import random

parser = argparse.ArgumentParser(description="Analyze GT relevance")
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
        docs.append({
            "pid": doc["pid"],
            "text": doc["text"],
            "num_vecs": len(doc["chunks"])
        })

print(f"Loaded {len(docs)} documents")

# Load queries
print("Loading queries...")
queries = []
with open(args.queries, "r") as f:
    for line in f:
        query = json.loads(line)
        queries.append({
            "pid": query["pid"],
            "text": query["text"],
            "num_vecs": len(query["chunks"])
        })

print(f"Loaded {len(queries)} queries")

# We need to compute Ground Truth using MaxSim
# For efficiency, let's compute for a sample of queries

def load_vectors(jsonl_path, max_items=None):
    """Load vectors from JSONL file."""
    all_vectors = []
    offsets = [0]

    with open(jsonl_path, "r") as f:
        for i, line in enumerate(f):
            if max_items and i >= max_items:
                break
            doc = json.loads(line)
            for chunk in doc["chunks"]:
                all_vectors.append(chunk["emb"])
            offsets.append(len(all_vectors))

    return np.array(all_vectors, dtype=np.float32), offsets

print("\nLoading vectors for MaxSim computation...")
doc_vectors, doc_offsets = load_vectors(args.docs)
query_vectors, query_offsets = load_vectors(args.queries)

print(f"Doc vectors: {doc_vectors.shape}")
print(f"Query vectors: {query_vectors.shape}")

def compute_maxsim(q_vecs, d_vecs):
    """Compute MaxSim score between query and document vectors."""
    # q_vecs: [nq, dim], d_vecs: [nd, dim]
    # MaxSim = sum over q of max over d of (q · d)
    scores = q_vecs @ d_vecs.T  # [nq, nd]
    max_scores = scores.max(axis=1)  # [nq]
    return max_scores.sum()

def get_top_k_docs(query_idx, k=10):
    """Get top-k documents for a query using MaxSim."""
    q_start = query_offsets[query_idx]
    q_end = query_offsets[query_idx + 1]
    q_vecs = query_vectors[q_start:q_end]

    scores = []
    for doc_idx in range(len(docs)):
        d_start = doc_offsets[doc_idx]
        d_end = doc_offsets[doc_idx + 1]
        d_vecs = doc_vectors[d_start:d_end]

        score = compute_maxsim(q_vecs, d_vecs)
        scores.append((doc_idx, score))

    scores.sort(key=lambda x: -x[1])
    return scores[:k]

# Sample 10 queries
print("\n" + "="*80)
print("ANALYZING 10 SAMPLE QUERIES AND THEIR TOP-3 GT DOCUMENTS")
print("="*80)

sample_indices = random.sample(range(len(queries)), min(10, len(queries)))
sample_indices.sort()

for i, q_idx in enumerate(sample_indices):
    print(f"\n{'='*80}")
    print(f"QUERY {i+1} (idx={q_idx})")
    print(f"{'='*80}")
    print(f"Query text: {queries[q_idx]['text'][:300]}...")
    print(f"Query vectors: {queries[q_idx]['num_vecs']}")

    print(f"\nComputing top documents...")
    top_docs = get_top_k_docs(q_idx, k=3)

    for rank, (doc_idx, score) in enumerate(top_docs):
        doc = docs[doc_idx]
        print(f"\n--- Top {rank+1} Document (idx={doc_idx}, score={score:.2f}, vectors={doc['num_vecs']}) ---")
        print(f"Text: {doc['text'][:500]}...")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
