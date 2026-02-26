#!/usr/bin/env python3
"""
Diagnose LEMUR failure: compare label distributions between datasets.

Replicates LEMUR's MaxSim label computation in Python:
  For each sampled doc token v, for each document d:
    label[v, d] = max_{t in d} IP(v, t)

Then analyzes:
  1. Label distribution (mean, std, min, max)
  2. Per-document avg label vs document length (correlation)
  3. Label variance per document (discriminativeness)
"""

import json
import numpy as np
import sys
import os
from collections import defaultdict


def load_jsonl_embeddings(jsonl_path, max_docs=5000):
    """Load embeddings from JSONL, return (all_vectors, offsets, doc_lengths)."""
    all_vecs = []
    offsets = [0]
    doc_lengths = []

    with open(jsonl_path) as f:
        for i, line in enumerate(f):
            if i >= max_docs:
                break
            doc = json.loads(line)
            chunks = doc["chunks"]
            n_vecs = len(chunks)
            doc_lengths.append(n_vecs)
            for chunk in chunks:
                all_vecs.append(chunk["emb"])
            offsets.append(offsets[-1] + n_vecs)

    all_vecs = np.array(all_vecs, dtype=np.float32)
    offsets = np.array(offsets, dtype=np.int64)
    doc_lengths = np.array(doc_lengths, dtype=np.int64)

    print(f"  Loaded {len(doc_lengths)} docs, {all_vecs.shape[0]} vectors, dim={all_vecs.shape[1]}")
    print(f"  Doc lengths: min={doc_lengths.min()}, max={doc_lengths.max()}, "
          f"avg={doc_lengths.mean():.1f}, median={np.median(doc_lengths):.0f}")

    return all_vecs, offsets, doc_lengths


def compute_maxsim_labels(samples, all_vecs, offsets, num_docs, batch_size=100):
    """
    Compute MaxSim labels: label[v, d] = max_{t in d} IP(v, t)
    Returns label matrix [num_samples, num_docs].
    """
    num_samples = samples.shape[0]
    total_vecs = all_vecs.shape[0]
    labels = np.zeros((num_samples, num_docs), dtype=np.float32)

    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        batch = samples[start:end]  # [batch, dim]

        # Compute all IPs: [batch, total_vecs]
        all_ips = batch @ all_vecs.T

        # Reduce to MaxSim per document
        for d in range(num_docs):
            doc_start = offsets[d]
            doc_end = offsets[d + 1]
            labels[start:end, d] = all_ips[:, doc_start:doc_end].max(axis=1)

        if (end - start) == batch_size:
            print(f"    Computed {end}/{num_samples} samples...", end='\r')

    print(f"    Computed {num_samples}/{num_samples} samples    ")
    return labels


def analyze_labels(labels, doc_lengths, dataset_name):
    """Analyze label distribution."""
    num_samples, num_docs = labels.shape

    print(f"\n{'='*60}")
    print(f"  {dataset_name}: Label Analysis ({num_samples} samples x {num_docs} docs)")
    print(f"{'='*60}")

    # 1. Overall distribution
    print(f"\n  [1] Overall label distribution:")
    print(f"      mean={labels.mean():.6f}, std={labels.std():.6f}")
    print(f"      min={labels.min():.6f}, max={labels.max():.6f}")
    print(f"      P10={np.percentile(labels, 10):.6f}, P50={np.percentile(labels, 50):.6f}, P90={np.percentile(labels, 90):.6f}")

    # 2. Per-document avg label vs doc length
    doc_avg_labels = labels.mean(axis=0)  # [num_docs]
    doc_std_labels = labels.std(axis=0)   # [num_docs]

    # Pearson correlation between doc length and avg label
    corr = np.corrcoef(doc_lengths, doc_avg_labels)[0, 1]
    print(f"\n  [2] Per-document avg label vs doc length:")
    print(f"      Pearson correlation: {corr:.4f}")
    print(f"      Doc avg label: min={doc_avg_labels.min():.6f}, max={doc_avg_labels.max():.6f}, "
          f"mean={doc_avg_labels.mean():.6f}")

    # Bucket by doc length
    length_buckets = [
        (0, 50, "0-50"),
        (50, 100, "50-100"),
        (100, 200, "100-200"),
        (200, 500, "200-500"),
        (500, 1000, "500-1K"),
        (1000, 5000, "1K-5K"),
    ]

    print(f"\n      Doc length buckets:")
    print(f"      {'Bucket':<12} {'Count':>6} {'Avg Label':>12} {'Std Label':>12}")
    for lo, hi, name in length_buckets:
        mask = (doc_lengths >= lo) & (doc_lengths < hi)
        if mask.sum() == 0:
            continue
        bucket_avg = doc_avg_labels[mask].mean()
        bucket_std = doc_std_labels[mask].mean()
        print(f"      {name:<12} {mask.sum():>6} {bucket_avg:>12.6f} {bucket_std:>12.6f}")

    # 3. Label discriminativeness (std across documents for each sample)
    per_sample_std = labels.std(axis=1)  # [num_samples]
    print(f"\n  [3] Label discriminativeness (std across docs per sample):")
    print(f"      mean={per_sample_std.mean():.6f}, min={per_sample_std.min():.6f}, max={per_sample_std.max():.6f}")

    # 4. Label range per sample
    per_sample_range = labels.max(axis=1) - labels.min(axis=1)
    print(f"\n  [4] Label range per sample (max - min):")
    print(f"      mean={per_sample_range.mean():.6f}, min={per_sample_range.min():.6f}, max={per_sample_range.max():.6f}")

    # 5. Normalized label stats (z-score, same as C++ code)
    label_mean = labels.mean()
    label_std = labels.std()
    if label_std < 1e-6:
        label_std = 1.0
    normalized = (labels - label_mean) / label_std
    print(f"\n  [5] Z-score normalization:")
    print(f"      label_mean={label_mean:.6f}, label_std={label_std:.6f}")

    return {
        'corr': corr,
        'label_mean': labels.mean(),
        'label_std': labels.std(),
        'doc_avg_labels': doc_avg_labels,
        'per_sample_std': per_sample_std,
    }


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "build"
    num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 2000

    datasets = {
        "LoTTE": os.path.join(data_dir, "lotte_lifestyle_gt_docs.jsonl"),
        "MSMARCO": os.path.join(data_dir, "msmarco_gt_docs.jsonl"),
        "SciFact": os.path.join(data_dir, "scifact_gt_docs.jsonl"),
    }

    results = {}

    for name, path in datasets.items():
        if not os.path.exists(path):
            print(f"\n  Skipping {name}: {path} not found")
            continue

        print(f"\n{'#'*60}")
        print(f"  Loading {name} from {path}")
        print(f"{'#'*60}")

        all_vecs, offsets, doc_lengths = load_jsonl_embeddings(path)
        num_docs = len(doc_lengths)
        total_vecs = all_vecs.shape[0]

        # Sample tokens (same as LEMUR: random from corpus)
        np.random.seed(42)
        actual_samples = min(num_samples, total_vecs)
        sample_indices = np.random.choice(total_vecs, actual_samples, replace=False)
        samples = all_vecs[sample_indices]

        print(f"  Sampled {actual_samples} tokens for label computation")

        # Compute labels
        print(f"  Computing MaxSim labels...")
        labels = compute_maxsim_labels(samples, all_vecs, offsets, num_docs, batch_size=200)

        # Analyze
        results[name] = analyze_labels(labels, doc_lengths, name)

    # Comparison
    if len(results) >= 2:
        print(f"\n{'#'*60}")
        print(f"  COMPARISON")
        print(f"{'#'*60}")
        print(f"\n  {'Dataset':<12} {'Label Mean':>12} {'Label Std':>12} {'Corr(len,label)':>16} {'Discrimin.':>12}")
        for name, r in results.items():
            print(f"  {name:<12} {r['label_mean']:>12.6f} {r['label_std']:>12.6f} {r['corr']:>16.4f} {r['per_sample_std'].mean():>12.6f}")


if __name__ == "__main__":
    main()
