#!/usr/bin/env python3
"""
Prepare Dense embeddings for comparison with Multi-Vector retrieval.

Uses BGE-M3 dense mode so that dense and multi-vector share the same backbone,
enabling a fair comparison (same model, same dim=1024, only retrieval strategy differs).

This script:
1. Loads documents and queries from existing multi-vector JSONL files
2. Encodes them with BGE-M3 dense mode
3. Saves embeddings in binary format for C++ loading

Usage:
    python scripts/prepare_dense_embeddings.py --input-dir build --output-dir build --dataset scifact

Requirements:
    pip install torch FlagEmbedding tqdm numpy
"""

import argparse
import json
import os
import struct
from pathlib import Path
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

BGE_M3_MODEL_PATH = '/home/spadea/models/bge-m3'


def load_texts_from_jsonl(jsonl_path: str, max_items: int = -1) -> Tuple[List[str], List[List[int]]]:
    """Load texts and GT labels from JSONL file."""
    texts = []
    gt_pids = []

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_items > 0 and i >= max_items:
                break

            try:
                item = json.loads(line)
                text = item.get('text', '')
                texts.append(text)

                # Load GT if present (for queries)
                if 'gt_pids' in item:
                    gt_pids.append(item['gt_pids'])
            except Exception as e:
                print(f"Error parsing line {i}: {e}")
                continue

    return texts, gt_pids


def encode_dense_with_bgem3(texts: List[str], model, batch_size: int = 32) -> np.ndarray:
    """Encode texts with BGE-M3 dense mode."""
    print(f"Encoding {len(texts)} texts with BGE-M3 dense mode...")

    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        batch = texts[i:i + batch_size]
        output = model.encode(batch, batch_size=batch_size, max_length=512)
        all_embeddings.append(output['dense_vecs'])

    return np.vstack(all_embeddings).astype(np.float32)


def save_dense_binary(embeddings: np.ndarray, output_path: str):
    """Save embeddings in binary format for C++ loading.

    Format:
        int32: dim
        int64: num_vectors
        float32[num_vectors * dim]: embeddings (row-major)
    """
    dim = embeddings.shape[1]
    num_vectors = embeddings.shape[0]

    with open(output_path, 'wb') as f:
        f.write(struct.pack('i', dim))
        f.write(struct.pack('q', num_vectors))
        embeddings.tofile(f)

    file_size = os.path.getsize(output_path)
    print(f"Saved {num_vectors} vectors (dim={dim}) to {output_path}")
    print(f"  File size: {file_size / 1e6:.1f} MB")


def save_gt_binary(gt_pids: List[List[int]], output_path: str):
    """Save ground truth labels in binary format.

    Format:
        int64: num_queries
        For each query:
            int64: num_gt
            int64[num_gt]: gt_pids
    """
    with open(output_path, 'wb') as f:
        f.write(struct.pack('q', len(gt_pids)))
        for gt in gt_pids:
            f.write(struct.pack('q', len(gt)))
            for pid in gt:
                f.write(struct.pack('q', pid))

    print(f"Saved GT for {len(gt_pids)} queries to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare Dense embeddings (BGE-M3) for comparison")
    parser.add_argument("--input-dir", type=str, default=".",
                        help="Directory containing multi-vector JSONL files")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Output directory for dense embeddings")
    parser.add_argument("--dataset", type=str, default="scifact",
                        choices=["msmarco", "scifact", "lotte"],
                        help="Dataset to process (default: scifact)")
    parser.add_argument("--max-docs", type=int, default=-1,
                        help="Maximum documents to encode (-1 for all)")
    parser.add_argument("--max-queries", type=int, default=-1,
                        help="Maximum queries to encode (-1 for all)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for encoding")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dataset file paths
    dataset_files = {
        "msmarco": ("msmarco_gt_docs.jsonl", "msmarco_gt_queries.jsonl"),
        "scifact": ("scifact_gt_docs.jsonl", "scifact_gt_queries.jsonl"),
        "lotte": ("lotte_science_gt_docs.jsonl", "lotte_science_gt_queries.jsonl"),
    }

    docs_file, queries_file = dataset_files[args.dataset]
    docs_path = input_dir / docs_file
    queries_path = input_dir / queries_file

    # Check files exist
    if not docs_path.exists():
        print(f"Error: Documents file not found: {docs_path}")
        print(f"Please run the multi-vector data preparation script first.")
        return

    if not queries_path.exists():
        print(f"Error: Queries file not found: {queries_path}")
        return

    # Load BGE-M3
    print(f"\n=== Loading BGE-M3 ===")
    from FlagEmbedding import BGEM3FlagModel
    model = BGEM3FlagModel(BGE_M3_MODEL_PATH, use_fp16=True)

    # ========== Process Documents ==========
    print(f"\n=== Processing Documents ===")
    print(f"Input: {docs_path}")

    doc_texts, _ = load_texts_from_jsonl(str(docs_path), args.max_docs)
    print(f"Loaded {len(doc_texts)} document texts")

    doc_embeddings = encode_dense_with_bgem3(doc_texts, model, args.batch_size)

    doc_output_path = output_dir / f"{args.dataset}_dense_docs.bin"
    save_dense_binary(doc_embeddings, str(doc_output_path))

    del doc_embeddings

    # ========== Process Queries ==========
    print(f"\n=== Processing Queries ===")
    print(f"Input: {queries_path}")

    query_texts, gt_pids = load_texts_from_jsonl(str(queries_path), args.max_queries)
    print(f"Loaded {len(query_texts)} query texts")

    query_embeddings = encode_dense_with_bgem3(query_texts, model, args.batch_size)

    query_output_path = output_dir / f"{args.dataset}_dense_queries.bin"
    save_dense_binary(query_embeddings, str(query_output_path))

    # Save GT labels
    gt_output_path = output_dir / f"{args.dataset}_dense_gt.bin"
    save_gt_binary(gt_pids, str(gt_output_path))

    # ========== Summary ==========
    print(f"\n=== Summary ===")
    print(f"Model: BGE-M3 (dense mode)")
    print(f"Dataset: {args.dataset}")
    print(f"Documents: {len(doc_texts)}, dim=1024")
    print(f"Queries: {len(query_texts)}, dim={query_embeddings.shape[1]}")
    print(f"\nOutput files:")
    print(f"  Documents: {doc_output_path}")
    print(f"  Queries: {query_output_path}")
    print(f"  GT labels: {gt_output_path}")
    print(f"\nTo run comparison test:")
    print(f"  ./tests/ut/knowhere_tests '[dense_vs_multivec]'")


if __name__ == "__main__":
    main()
