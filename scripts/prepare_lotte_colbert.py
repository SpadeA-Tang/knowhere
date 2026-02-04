#!/usr/bin/env python3
"""
Prepare LoTTE ColBERT embeddings for knowhere testing.

This script downloads LoTTE dataset, splits long documents into passages,
encodes each passage with ColBERT, and merges all passage vectors into
a single entity per document.

Usage:
    # Encode LoTTE data with ColBERT (requires GPU)
    python scripts/prepare_lotte_colbert.py --domain science --max-docs 1000

    # Generate synthetic long-document data (for quick testing)
    python scripts/prepare_lotte_colbert.py --synthetic --max-docs 1000

Requirements:
    pip install torch --index-url https://download.pytorch.org/whl/cu118
    pip install colbert-ai transformers datasets tqdm numpy
"""

import argparse
import json
import os
import numpy as np
from pathlib import Path
from typing import List, Tuple

DEFAULT_MAX_DOCS = 1000
DEFAULT_MAX_QUERIES = 100
DEFAULT_OUTPUT_DIR = "."
DEFAULT_DOMAIN = "science"  # science, lifestyle, writing, recreation, technology

# ColBERT max tokens per passage
COLBERT_DOC_MAXLEN = 180


def split_document_into_passages(text: str, max_tokens: int = COLBERT_DOC_MAXLEN) -> List[str]:
    """
    Split a long document into passages of max_tokens.
    Uses simple word-based splitting (approximation of token count).
    """
    words = text.split()
    # Approximate: 1 word ~ 1.3 tokens on average
    words_per_passage = int(max_tokens / 1.3)

    passages = []
    for i in range(0, len(words), words_per_passage):
        passage = " ".join(words[i:i + words_per_passage])
        if passage.strip():
            passages.append(passage)

    # Ensure at least one passage
    if not passages:
        passages = [text[:500] if text else "empty"]

    return passages


def encode_passages_with_colbert(passages: List[str], checkpoint, batch_size: int = 32) -> np.ndarray:
    """Encode passages and return concatenated vectors."""
    import torch

    all_vectors = []

    for i in range(0, len(passages), batch_size):
        batch = passages[i:i + batch_size]
        with torch.no_grad():
            embs = checkpoint.docFromText(batch)

            for emb in embs:
                # Remove padding (zero vectors)
                norms = torch.norm(emb, dim=-1)
                mask = norms > 1e-6
                valid_emb = emb[mask].cpu().numpy()

                if len(valid_emb) == 0:
                    valid_emb = emb[0:1].cpu().numpy()

                all_vectors.append(valid_emb)

    # Concatenate all passage vectors into one array
    return np.vstack(all_vectors).astype(np.float32)


def save_to_jsonl(documents: List[dict], output_path: str):
    """Save documents to JSONL format (same as milvus_insert.1k.jsonl)."""
    with open(output_path, 'w') as f:
        for doc in documents:
            f.write(json.dumps(doc) + '\n')

    print(f"Saved {len(documents)} documents to {output_path}")
    print(f"  File size: {os.path.getsize(output_path) / 1e6:.1f} MB")


def load_tsv_documents(tsv_path: str, max_docs: int) -> List[Tuple[str, str]]:
    """Load documents from TSV file (id\\ttext format)."""
    docs = []
    with open(tsv_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_docs:
                break
            parts = line.strip().split('\t', 1)
            if len(parts) == 2:
                doc_id, text = parts
                docs.append((doc_id, text))
    return docs


def load_tsv_queries(tsv_path: str, max_queries: int) -> List[Tuple[str, str]]:
    """Load queries from TSV file (id\\ttext format)."""
    queries = []
    with open(tsv_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_queries:
                break
            parts = line.strip().split('\t', 1)
            if len(parts) == 2:
                query_id, text = parts
                queries.append((query_id, text))
    return queries


def download_and_encode_lotte(domain: str, max_docs: int, max_queries: int, output_dir: Path):
    """Load LoTTE from local TSV files and encode with ColBERT."""
    import torch
    from tqdm import tqdm

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    # Load ColBERT model
    print("\n=== Loading ColBERT model ===")
    from colbert.infra import ColBERTConfig
    from colbert.modeling.checkpoint import Checkpoint

    config = ColBERTConfig(
        doc_maxlen=COLBERT_DOC_MAXLEN,
        query_maxlen=32,
    )
    checkpoint = Checkpoint("colbert-ir/colbertv2.0", colbert_config=config)
    print("ColBERT model loaded")

    # Load from local TSV files
    # Expected path: lotte/{domain}/dev/collection.tsv
    lotte_dir = output_dir / "lotte" / domain / "dev"
    collection_path = lotte_dir / "collection.tsv"
    queries_path = lotte_dir / "questions.forum.tsv"

    if not collection_path.exists():
        raise FileNotFoundError(f"LoTTE collection not found: {collection_path}\n"
                                f"Please extract lotte.tar.gz in {output_dir}")

    print(f"\n=== Loading LoTTE from local files ===")
    print(f"  Collection: {collection_path}")
    print(f"  Queries: {queries_path}")

    # Load documents
    print(f"\n=== Loading documents (max {max_docs}) ===")
    raw_docs = load_tsv_documents(str(collection_path), max_docs)
    print(f"Loaded {len(raw_docs)} documents from TSV")

    # Process documents
    print(f"\n=== Encoding documents with ColBERT ===")
    documents = []

    for i, (doc_id, text) in enumerate(tqdm(raw_docs, desc="Processing docs")):
        if not text:
            continue

        # Split into passages
        passages = split_document_into_passages(text)

        # Encode all passages
        vectors = encode_passages_with_colbert(passages, checkpoint)

        # Create document entry (same format as milvus_insert.1k.jsonl)
        chunks = []
        for j, vec in enumerate(vectors):
            chunks.append({
                "pos": j,
                "emb": vec.tolist()
            })

        doc_entry = {
            "pid": i,
            "text": text[:500],  # Truncate text for storage
            "chunks": chunks
        }
        documents.append(doc_entry)

        if (i + 1) % 100 == 0:
            avg_vecs = np.mean([len(d['chunks']) for d in documents])
            print(f"  Processed {i + 1} docs, avg vectors/doc: {avg_vecs:.1f}")

    # Load and process queries
    print(f"\n=== Processing queries (max {max_queries}) ===")
    if queries_path.exists():
        raw_queries = load_tsv_queries(str(queries_path), max_queries)
        queries = [text for _, text in raw_queries]
        print(f"Loaded {len(queries)} queries from TSV")
    else:
        print(f"Queries file not found, using document prefixes")
        queries = [doc['text'][:100] for doc in documents[:max_queries]]

    # Encode queries
    query_documents = []
    for i, query in enumerate(tqdm(queries, desc="Encoding queries")):
        with torch.no_grad():
            embs = checkpoint.queryFromText([query])
            emb = embs[0]
            norms = torch.norm(emb, dim=-1)
            mask = norms > 1e-6
            valid_emb = emb[mask].cpu().numpy()

            if len(valid_emb) == 0:
                valid_emb = emb[0:1].cpu().numpy()

        chunks = [{"pos": j, "emb": vec.tolist()} for j, vec in enumerate(valid_emb)]
        query_documents.append({
            "pid": i,
            "text": query[:200],
            "chunks": chunks
        })

    return documents, query_documents


def generate_synthetic_long_docs(max_docs: int, max_queries: int, dim: int = 128):
    """
    Generate synthetic long-document data with high variance in vector count.
    Simulates documents with 100-2000 vectors per document.
    """
    print(f"Generating synthetic long-document data:")
    print(f"  Documents: {max_docs}, Queries: {max_queries}, Dim: {dim}")
    print(f"  Vectors per doc: 100-2000 (high variance)")

    np.random.seed(42)

    documents = []
    for i in range(max_docs):
        # High variance: some short docs (100 vecs), some very long (2000 vecs)
        # Use log-normal distribution for realistic long-tail
        num_vecs = int(np.clip(np.random.lognormal(mean=5.5, sigma=0.8), 100, 2000))

        # Generate vectors with some structure
        base = np.random.randn(dim).astype(np.float32)
        base /= np.linalg.norm(base)

        vecs = base + 0.3 * np.random.randn(num_vecs, dim).astype(np.float32)
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

        chunks = [{"pos": j, "emb": vec.tolist()} for j, vec in enumerate(vecs)]

        documents.append({
            "pid": i,
            "text": f"Synthetic document {i} with {num_vecs} vectors",
            "chunks": chunks
        })

        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1} documents...")

    # Generate queries (shorter, 10-50 vectors)
    query_documents = []
    for i in range(max_queries):
        num_vecs = np.random.randint(10, 51)

        base = np.random.randn(dim).astype(np.float32)
        base /= np.linalg.norm(base)

        vecs = base + 0.3 * np.random.randn(num_vecs, dim).astype(np.float32)
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

        chunks = [{"pos": j, "emb": vec.tolist()} for j, vec in enumerate(vecs)]

        query_documents.append({
            "pid": i,
            "text": f"Synthetic query {i}",
            "chunks": chunks
        })

    return documents, query_documents


def print_statistics(documents: List[dict], name: str):
    """Print statistics about the dataset."""
    vec_counts = [len(doc['chunks']) for doc in documents]

    print(f"\n{name} Statistics:")
    print(f"  Documents: {len(documents)}")
    print(f"  Total vectors: {sum(vec_counts)}")
    print(f"  Vectors per doc:")
    print(f"    Min: {min(vec_counts)}")
    print(f"    Max: {max(vec_counts)}")
    print(f"    Avg: {np.mean(vec_counts):.1f}")
    print(f"    Median: {np.median(vec_counts):.0f}")
    print(f"    Std: {np.std(vec_counts):.1f}")

    # Distribution buckets
    buckets = [100, 200, 500, 1000, 2000]
    print(f"  Distribution:")
    prev = 0
    for b in buckets:
        count = sum(1 for c in vec_counts if prev < c <= b)
        print(f"    {prev+1}-{b}: {count} docs ({100*count/len(documents):.1f}%)")
        prev = b
    count = sum(1 for c in vec_counts if c > buckets[-1])
    if count > 0:
        print(f"    >{buckets[-1]}: {count} docs ({100*count/len(documents):.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Prepare LoTTE ColBERT data for long-document testing")
    parser.add_argument("--domain", type=str, default=DEFAULT_DOMAIN,
                        choices=["science", "lifestyle", "writing", "recreation", "technology"],
                        help=f"LoTTE domain (default: {DEFAULT_DOMAIN})")
    parser.add_argument("--max-docs", type=int, default=DEFAULT_MAX_DOCS,
                        help=f"Maximum documents (default: {DEFAULT_MAX_DOCS})")
    parser.add_argument("--max-queries", type=int, default=DEFAULT_MAX_QUERIES,
                        help=f"Maximum queries (default: {DEFAULT_MAX_QUERIES})")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--synthetic", action="store_true",
                        help="Generate synthetic data instead of real ColBERT embeddings")
    parser.add_argument("--dim", type=int, default=128,
                        help="Dimension for synthetic data (default: 128)")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        documents, query_documents = generate_synthetic_long_docs(
            args.max_docs, args.max_queries, args.dim
        )
        output_name = "lotte_synthetic"
    else:
        documents, query_documents = download_and_encode_lotte(
            args.domain, args.max_docs, args.max_queries, output_dir
        )
        output_name = f"lotte_{args.domain}"

    # Print statistics
    print_statistics(documents, "Documents")
    print_statistics(query_documents, "Queries")

    # Save to JSONL
    doc_path = output_dir / f"{output_name}_docs.jsonl"
    query_path = output_dir / f"{output_name}_queries.jsonl"

    print(f"\n=== Saving Data ===")
    save_to_jsonl(documents, str(doc_path))
    save_to_jsonl(query_documents, str(query_path))

    print(f"\n=== Done! ===")
    print(f"Documents: {doc_path}")
    print(f"Queries: {query_path}")
    print(f"\nTo use in test, update COLBERT_JSONL_PATH in test_emb_list_msmarco.cc")


if __name__ == "__main__":
    main()
