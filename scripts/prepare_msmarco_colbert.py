#!/usr/bin/env python3
"""
Prepare MS MARCO ColBERT embeddings for knowhere testing.

Usage:
    # Download pre-computed ColBERT embeddings (recommended)
    python scripts/prepare_msmarco_colbert.py --download --max-docs 100000

    # Generate synthetic data (for quick testing)
    python scripts/prepare_msmarco_colbert.py --synthetic --max-docs 100000

Requirements for --download:
    pip install torch numpy tqdm requests

Binary output format:
    vectors.bin: [int32 dim][int64 total_vectors][float32 * dim * total_vectors]
    offsets.bin: [int64 num_docs][size_t * (num_docs + 1)]
"""

import argparse
import os
import struct
import numpy as np
from pathlib import Path

DEFAULT_MAX_DOCS = 10000
DEFAULT_MAX_QUERIES = 1000
DEFAULT_OUTPUT_DIR = "build/Release"


def save_emb_list_binary(vectors: np.ndarray, offsets: np.ndarray,
                          vectors_path: str, offsets_path: str):
    """Save embeddings in binary format for C++ loading."""
    dim = vectors.shape[1]
    total_vectors = vectors.shape[0]
    num_docs = len(offsets) - 1

    with open(vectors_path, 'wb') as f:
        f.write(struct.pack('i', dim))
        f.write(struct.pack('q', total_vectors))
        vectors.astype(np.float32).tofile(f)

    with open(offsets_path, 'wb') as f:
        f.write(struct.pack('q', num_docs))
        offsets.astype(np.uint64).tofile(f)

    print(f"Saved {num_docs} docs, {total_vectors} vectors, dim={dim}")
    print(f"  vectors: {vectors_path} ({os.path.getsize(vectors_path) / 1e6:.1f} MB)")
    print(f"  offsets: {offsets_path} ({os.path.getsize(offsets_path) / 1e3:.1f} KB)")


def download_precomputed_colbert(output_dir: Path, max_docs: int, max_queries: int):
    """
    Download pre-computed ColBERT v2 embeddings from official source.

    The embeddings are stored in a packed format where each passage has variable
    number of token embeddings (128-dim each).
    """
    import torch
    import requests
    from tqdm import tqdm

    # ColBERT v2 pre-computed embeddings URLs
    # These are from the official ColBERT repository
    base_url = "https://huggingface.co/colbert-ir/colbertv2.0_msmarco_passage/resolve/main"

    cache_dir = output_dir / "colbert_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    def download_file(url, local_path):
        if local_path.exists():
            print(f"Using cached: {local_path}")
            return
        print(f"Downloading: {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total = int(response.headers.get('content-length', 0))
        with open(local_path, 'wb') as f:
            with tqdm(total=total, unit='B', unit_scale=True) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))

    # Try to download from HuggingFace datasets with pre-computed embeddings
    try:
        from datasets import load_dataset
        print("Loading pre-computed ColBERT embeddings from HuggingFace...")

        # This dataset contains pre-computed ColBERT embeddings
        # Format: each example has 'embeddings' (list of 128-dim vectors) and 'doc_id'
        dataset = load_dataset(
            "colbert-ir/colbertv2.0_msmarco_passage",
            split="train",
            streaming=True  # Stream to avoid loading all into memory
        )

        all_vectors = []
        offsets = [0]
        doc_count = 0

        print(f"Processing up to {max_docs} documents...")
        for example in tqdm(dataset, total=max_docs):
            if doc_count >= max_docs:
                break

            # Each example has token embeddings
            embs = np.array(example['embeddings'], dtype=np.float32)
            all_vectors.append(embs)
            offsets.append(offsets[-1] + len(embs))
            doc_count += 1

        doc_vectors = np.vstack(all_vectors)
        doc_offsets = np.array(offsets, dtype=np.uint64)

        print(f"Loaded {doc_count} documents, {len(doc_vectors)} vectors")
        return doc_vectors, doc_offsets, None, None

    except Exception as e:
        print(f"HuggingFace loading failed: {e}")
        print("Falling back to alternative method...")

    # Alternative: Download raw embeddings files
    # ColBERT stores embeddings in .pt files
    try:
        print("\nTrying alternative download method...")

        # Download a sample of pre-computed embeddings
        # These URLs point to smaller samples for testing
        sample_url = "https://public.ukp.informatik.tu-darmstadt.de/kwang/colbert/colbertv2_msmarco_embeddings_sample.tar.gz"

        # For now, fall back to synthetic if download fails
        raise NotImplementedError("Alternative download not implemented yet")

    except Exception as e:
        print(f"Alternative download failed: {e}")
        return None, None, None, None


def generate_synthetic_data(max_docs, max_queries, dim=128):
    """Generate synthetic ColBERT-like data for testing."""
    print(f"Generating synthetic data: {max_docs} docs, {max_queries} queries, dim={dim}")

    np.random.seed(42)

    # Generate documents with variable number of vectors (20-60 per doc)
    doc_vectors = []
    doc_offsets = [0]

    for i in range(max_docs):
        num_vecs = np.random.randint(20, 61)
        base = np.random.randn(dim).astype(np.float32)
        base /= np.linalg.norm(base)
        vecs = base + 0.3 * np.random.randn(num_vecs, dim).astype(np.float32)
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        doc_vectors.append(vecs)
        doc_offsets.append(doc_offsets[-1] + num_vecs)

    doc_vectors = np.vstack(doc_vectors).astype(np.float32)
    doc_offsets = np.array(doc_offsets, dtype=np.uint64)

    # Generate queries
    query_vectors = []
    query_offsets = [0]

    for i in range(max_queries):
        num_vecs = np.random.randint(10, 33)
        base = np.random.randn(dim).astype(np.float32)
        base /= np.linalg.norm(base)
        vecs = base + 0.3 * np.random.randn(num_vecs, dim).astype(np.float32)
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        query_vectors.append(vecs)
        query_offsets.append(query_offsets[-1] + num_vecs)

    query_vectors = np.vstack(query_vectors).astype(np.float32)
    query_offsets = np.array(query_offsets, dtype=np.uint64)

    return doc_vectors, doc_offsets, query_vectors, query_offsets


def load_beir_with_colbert(dataset_name: str, max_docs: int, max_queries: int):
    """
    Load a BEIR dataset and encode with ColBERT.
    Smaller datasets like SciFact, NFCorpus are quick to encode.
    """
    try:
        from colbert.infra import ColBERTConfig
        from colbert.modeling.checkpoint import Checkpoint
        from beir import util
        from beir.datasets.data_loader import GenericDataLoader

        print(f"Loading BEIR dataset: {dataset_name}")

        # Download dataset
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
        data_path = util.download_and_unzip(url, "datasets")

        # Load corpus and queries
        corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

        print(f"Corpus size: {len(corpus)}, Queries: {len(queries)}")

        # Initialize ColBERT
        config = ColBERTConfig(doc_maxlen=180, query_maxlen=32)
        checkpoint = Checkpoint("colbert-ir/colbertv2.0", colbert_config=config)

        # Encode documents
        doc_texts = [corpus[doc_id]['text'] for doc_id in list(corpus.keys())[:max_docs]]
        # ... encoding logic

        return None  # TODO: implement full encoding

    except ImportError as e:
        print(f"BEIR/ColBERT not available: {e}")
        return None


def download_from_huggingface_embeddings(max_docs: int, max_queries: int):
    """
    Download pre-computed embeddings from HuggingFace.
    Several researchers have uploaded ColBERT embeddings.
    """
    try:
        from datasets import load_dataset
        import torch

        print("Searching for pre-computed ColBERT embeddings on HuggingFace...")

        # Try known datasets with pre-computed embeddings
        # Option 1: answerdotai/msmarco-passage-embeddings (if available)
        # Option 2: Other community uploads

        # For MS MARCO, we can try loading from various sources
        datasets_to_try = [
            ("Tevatron/msmarco-passage-corpus", "train"),
            ("sentence-transformers/msmarco-hard-negatives", "train"),
        ]

        for dataset_name, split in datasets_to_try:
            try:
                print(f"Trying: {dataset_name}")
                ds = load_dataset(dataset_name, split=split, streaming=True)
                sample = next(iter(ds))
                print(f"  Fields: {sample.keys()}")
                if 'embeddings' in sample or 'embedding' in sample:
                    print(f"  Found embeddings!")
                    # Process this dataset
                    break
            except Exception as e:
                print(f"  Failed: {e}")
                continue

        return None, None, None, None

    except Exception as e:
        print(f"HuggingFace download failed: {e}")
        return None, None, None, None


def main():
    parser = argparse.ArgumentParser(description="Prepare MS MARCO ColBERT data")
    parser.add_argument("--max-docs", type=int, default=DEFAULT_MAX_DOCS)
    parser.add_argument("--max-queries", type=int, default=DEFAULT_MAX_QUERIES)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--synthetic", action="store_true",
                        help="Generate synthetic data (fast, no download)")
    parser.add_argument("--download", action="store_true",
                        help="Download pre-computed ColBERT embeddings")
    parser.add_argument("--dim", type=int, default=128,
                        help="Dimension for synthetic data")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    doc_vectors_path = output_dir / "msmarco_doc_vectors.bin"
    doc_offsets_path = output_dir / "msmarco_doc_offsets.bin"
    query_vectors_path = output_dir / "msmarco_query_vectors.bin"
    query_offsets_path = output_dir / "msmarco_query_offsets.bin"

    doc_vectors, doc_offsets, query_vectors, query_offsets = None, None, None, None

    if args.synthetic:
        doc_vectors, doc_offsets, query_vectors, query_offsets = generate_synthetic_data(
            args.max_docs, args.max_queries, args.dim
        )
    elif args.download:
        result = download_precomputed_colbert(output_dir, args.max_docs, args.max_queries)
        if result[0] is not None:
            doc_vectors, doc_offsets, query_vectors, query_offsets = result
        else:
            print("\nPre-computed embeddings not available.")
            print("Falling back to synthetic data...")
            doc_vectors, doc_offsets, query_vectors, query_offsets = generate_synthetic_data(
                args.max_docs, args.max_queries, args.dim
            )
    else:
        # Default: try download, fallback to synthetic
        print("No mode specified. Use --synthetic or --download")
        print("Generating synthetic data by default...")
        doc_vectors, doc_offsets, query_vectors, query_offsets = generate_synthetic_data(
            args.max_docs, args.max_queries, args.dim
        )

    # If we only got documents (no queries), generate synthetic queries
    if query_vectors is None:
        print("\nGenerating synthetic queries...")
        _, _, query_vectors, query_offsets = generate_synthetic_data(
            100, args.max_queries, doc_vectors.shape[1] if doc_vectors is not None else args.dim
        )

    # Save to binary files
    print("\n=== Saving Data ===")
    print("Documents:")
    save_emb_list_binary(doc_vectors, doc_offsets,
                         str(doc_vectors_path), str(doc_offsets_path))

    print("\nQueries:")
    save_emb_list_binary(query_vectors, query_offsets,
                         str(query_vectors_path), str(query_offsets_path))

    # Print statistics
    print("\n=== Statistics ===")
    num_docs = len(doc_offsets) - 1
    doc_counts = np.diff(doc_offsets)
    print(f"Documents: {num_docs}")
    print(f"  Total vectors: {doc_offsets[-1]}")
    print(f"  Vectors per doc: min={doc_counts.min()}, max={doc_counts.max()}, "
          f"avg={doc_counts.mean():.1f}, median={np.median(doc_counts):.0f}")

    num_queries = len(query_offsets) - 1
    query_counts = np.diff(query_offsets)
    print(f"\nQueries: {num_queries}")
    print(f"  Total vectors: {query_offsets[-1]}")
    print(f"  Vectors per query: min={query_counts.min()}, max={query_counts.max()}, "
          f"avg={query_counts.mean():.1f}, median={np.median(query_counts):.0f}")

    print(f"\n=== Done! ===")
    print(f"Run test: ./tests/ut/knowhere_tests '[msmarco_emb_list]'")


if __name__ == "__main__":
    main()
