#!/usr/bin/env python3
"""
Prepare MS MARCO ColBERT embeddings for knowhere testing.

Usage:
    # Encode real MS MARCO data with ColBERT (requires GPU)
    python scripts/prepare_msmarco_colbert.py --max-docs 10000 --max-queries 100

    # Generate synthetic data (for quick testing, no GPU needed)
    python scripts/prepare_msmarco_colbert.py --synthetic --max-docs 10000

Requirements:
    pip install torch --index-url https://download.pytorch.org/whl/cu118
    pip install colbert-ai transformers datasets tqdm numpy
"""

import argparse
import os
import struct
import numpy as np
from pathlib import Path

DEFAULT_MAX_DOCS = 10000
DEFAULT_MAX_QUERIES = 100
DEFAULT_OUTPUT_DIR = "."


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


def encode_with_colbert(texts: list, checkpoint, is_query: bool = False, batch_size: int = 32):
    """Encode texts using ColBERT model."""
    import torch
    from tqdm import tqdm

    all_embeddings = []
    offsets = [0]

    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        batch = texts[i:i+batch_size]
        with torch.no_grad():
            if is_query:
                embs = checkpoint.queryFromText(batch)
            else:
                embs = checkpoint.docFromText(batch)

            # embs shape: [batch_size, max_len, dim]
            # Each text has different actual length (non-padding)
            for emb in embs:
                # Remove padding (zero vectors or very small norm)
                norms = torch.norm(emb, dim=-1)
                mask = norms > 1e-6
                valid_emb = emb[mask].cpu().numpy()

                if len(valid_emb) == 0:
                    # Fallback: use at least one vector
                    valid_emb = emb[0:1].cpu().numpy()

                all_embeddings.append(valid_emb)
                offsets.append(offsets[-1] + len(valid_emb))

    vectors = np.vstack(all_embeddings).astype(np.float32)
    offsets = np.array(offsets, dtype=np.uint64)

    return vectors, offsets


def download_and_encode_msmarco(max_docs: int, max_queries: int, output_dir: Path):
    """Download MS MARCO and encode with ColBERT."""
    import torch
    from datasets import load_dataset

    # Check CUDA
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load ColBERT model
    print("\n=== Loading ColBERT model ===")
    from colbert.infra import ColBERTConfig
    from colbert.modeling.checkpoint import Checkpoint

    config = ColBERTConfig(
        doc_maxlen=180,
        query_maxlen=32,
    )
    checkpoint = Checkpoint("colbert-ir/colbertv2.0", colbert_config=config)
    print("ColBERT model loaded successfully")

    # Download MS MARCO passages
    print(f"\n=== Downloading MS MARCO passages (first {max_docs}) ===")
    try:
        # Try the Tevatron corpus first (cleaner format)
        corpus = load_dataset("Tevatron/msmarco-passage-corpus", split="train", streaming=True)
        passages = []
        for i, item in enumerate(corpus):
            if i >= max_docs:
                break
            passages.append(item['text'])
            if (i + 1) % 1000 == 0:
                print(f"  Loaded {i + 1} passages...")
        print(f"Loaded {len(passages)} passages")
    except Exception as e:
        print(f"Tevatron dataset failed: {e}")
        print("Trying microsoft/ms_marco...")
        corpus = load_dataset("microsoft/ms_marco", "v1.1", split="train", streaming=True)
        passages = []
        for i, item in enumerate(corpus):
            if i >= max_docs:
                break
            # ms_marco format has passages in a different structure
            if 'passages' in item:
                for p in item['passages']['passage_text']:
                    passages.append(p)
                    if len(passages) >= max_docs:
                        break
            if len(passages) >= max_docs:
                break
        passages = passages[:max_docs]
        print(f"Loaded {len(passages)} passages")

    # Encode passages
    print(f"\n=== Encoding {len(passages)} passages with ColBERT ===")
    doc_vectors, doc_offsets = encode_with_colbert(passages, checkpoint, is_query=False, batch_size=32)

    # Download queries
    print(f"\n=== Downloading MS MARCO queries (first {max_queries}) ===")
    try:
        queries_ds = load_dataset("Tevatron/msmarco-passage", split="dev", streaming=True)
        queries = []
        for i, item in enumerate(queries_ds):
            if i >= max_queries:
                break
            queries.append(item['query'])
        print(f"Loaded {len(queries)} queries")
    except Exception as e:
        print(f"Query dataset failed: {e}, using passage prefixes as queries")
        queries = [p[:100] for p in passages[:max_queries]]
        print(f"Generated {len(queries)} synthetic queries from passages")

    # Encode queries
    print(f"\n=== Encoding {len(queries)} queries with ColBERT ===")
    query_vectors, query_offsets = encode_with_colbert(queries, checkpoint, is_query=True, batch_size=32)

    return doc_vectors, doc_offsets, query_vectors, query_offsets


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


def main():
    parser = argparse.ArgumentParser(description="Prepare MS MARCO ColBERT data")
    parser.add_argument("--max-docs", type=int, default=DEFAULT_MAX_DOCS,
                        help=f"Maximum documents to encode (default: {DEFAULT_MAX_DOCS})")
    parser.add_argument("--max-queries", type=int, default=DEFAULT_MAX_QUERIES,
                        help=f"Maximum queries to encode (default: {DEFAULT_MAX_QUERIES})")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--synthetic", action="store_true",
                        help="Generate synthetic data instead of real ColBERT embeddings")
    parser.add_argument("--dim", type=int, default=128,
                        help="Dimension for synthetic data (default: 128)")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    doc_vectors_path = output_dir / "msmarco_doc_vectors.bin"
    doc_offsets_path = output_dir / "msmarco_doc_offsets.bin"
    query_vectors_path = output_dir / "msmarco_query_vectors.bin"
    query_offsets_path = output_dir / "msmarco_query_offsets.bin"

    if args.synthetic:
        doc_vectors, doc_offsets, query_vectors, query_offsets = generate_synthetic_data(
            args.max_docs, args.max_queries, args.dim
        )
    else:
        doc_vectors, doc_offsets, query_vectors, query_offsets = download_and_encode_msmarco(
            args.max_docs, args.max_queries, output_dir
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
    print(f"  Dimension: {doc_vectors.shape[1]}")

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
