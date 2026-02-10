#!/usr/bin/env python3
"""
Prepare BioASQ dense embeddings using E5-base-v2 for comparison with ColBERTv2 multi-vector.

This script reads from JSONL files produced by prepare_bioasq_colbertv2.py (to ensure
the same documents and queries are used for both dense and multi-vector evaluation),
then encodes with E5-base-v2 and saves in binary format for C++ test loading.

Usage:
    # Read from JSONL files produced by prepare_bioasq_colbertv2.py
    python scripts/prepare_bioasq_e5_dense.py --input-dir build --output-dir build

    # Use local E5 model
    python scripts/prepare_bioasq_e5_dense.py --input-dir build --output-dir build \
        --model /path/to/e5-base-v2

    # Directly from BioASQ data (without ColBERTv2 step)
    python scripts/prepare_bioasq_e5_dense.py --data-dir /path/to/bioasq --output-dir build

Requirements:
    pip install torch transformers tqdm numpy
"""

import argparse
import json
import os
import random
import struct
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import torch
from tqdm import tqdm

BIOASQ_BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/bioasq.zip"


# ============================================================================
# E5-base-v2 Encoder
# ============================================================================

class E5Encoder:
    """E5-base-v2 dense encoder using transformers."""

    def __init__(self, model_path="intfloat/e5-base-v2"):
        from transformers import AutoTokenizer, AutoModel

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading E5-base-v2 from {model_path} (device={self.device})...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(self.device).eval()

        # Get embedding dimension from model config
        self.dim = self.model.config.hidden_size
        print(f"E5-base-v2 loaded: dim={self.dim}")

    @staticmethod
    def _average_pool(last_hidden_state, attention_mask):
        """Average pooling with attention mask."""
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    @torch.no_grad()
    def encode(self, texts, batch_size=32, max_length=512, prefix=""):
        """Encode texts to single dense embeddings.

        Args:
            texts: List of input texts
            batch_size: Batch size for encoding
            max_length: Maximum token length
            prefix: Prefix to prepend ("query: " for queries, "passage: " for docs)

        Returns:
            numpy array of shape [num_texts, dim]
        """
        all_embeddings = []

        for i in tqdm(range(0, len(texts), batch_size), desc=f"Encoding ({prefix.strip()})"):
            batch = [f"{prefix}{t}" for t in texts[i:i + batch_size]]
            inputs = self.tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=max_length
            ).to(self.device)

            outputs = self.model(**inputs)
            embeddings = self._average_pool(outputs.last_hidden_state, inputs['attention_mask'])
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)

            all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings).astype(np.float32)


# ============================================================================
# Binary I/O (compatible with test_dense_vs_multivec.cc)
# ============================================================================

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


# ============================================================================
# Data Loading
# ============================================================================

def load_texts_from_jsonl(jsonl_path: str, max_items: int = -1):
    """Load texts and GT labels from JSONL file (produced by prepare_bioasq_colbertv2.py)."""
    texts = []
    gt_pids = []

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if 0 < max_items <= i:
                break

            item = json.loads(line)
            text = item.get('text', '')
            texts.append(text)

            if 'gt_pids' in item:
                gt_pids.append(item['gt_pids'])

    return texts, gt_pids


def load_bioasq_direct(data_dir: Path, max_queries: int, max_docs: int):
    """Load BioASQ directly from BEIR format (without ColBERTv2 JSONL step)."""
    # Load qrels
    qrels = {}
    for split in ["test.tsv", "train.tsv", "dev.tsv"]:
        qrels_path = data_dir / "qrels" / split
        if not qrels_path.exists():
            continue
        with open(qrels_path, 'r', encoding='utf-8') as f:
            next(f)
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3 and int(parts[2]) > 0:
                    qrels.setdefault(parts[0], []).append(parts[1])

    # Select queries
    selected_qids = sorted(qrels.keys())[:max_queries]
    gt_doc_ids = set()
    for qid in selected_qids:
        gt_doc_ids.update(qrels[qid])

    # Load query texts
    queries = {}
    with open(data_dir / "queries.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            q = json.loads(line)
            if q['_id'] in set(selected_qids):
                queries[q['_id']] = q['text']

    selected_qids = [qid for qid in selected_qids if qid in queries]

    # Stream through corpus with reservoir sampling
    docs = {}
    all_pids = []
    remaining_slots = max(0, max_docs - len(gt_doc_ids))
    reservoir = []
    total_non_gt = 0

    print(f"Streaming through corpus...")
    with open(data_dir / "corpus.jsonl", 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            if (line_num + 1) % 1000000 == 0:
                print(f"  Scanned {(line_num + 1) // 1000000}M docs...")
            doc = json.loads(line)
            pid = doc['_id']
            title = doc.get('title', '')
            text = doc.get('text', '')
            full_text = f"{title} {text}".strip() if title else text

            if pid in gt_doc_ids:
                docs[pid] = full_text
                all_pids.append(pid)
            else:
                total_non_gt += 1
                if len(reservoir) < remaining_slots:
                    reservoir.append((pid, full_text))
                else:
                    j = random.randint(0, total_non_gt - 1)
                    if j < remaining_slots:
                        reservoir[j] = (pid, full_text)

    for pid, text in reservoir:
        docs[pid] = text
        all_pids.append(pid)

    pid_to_idx = {pid: idx for idx, pid in enumerate(all_pids)}

    doc_texts = [docs[pid] for pid in all_pids]
    query_texts = [queries[qid] for qid in selected_qids]
    gt_pids_list = [[pid_to_idx[pid] for pid in qrels[qid] if pid in pid_to_idx]
                    for qid in selected_qids]

    return doc_texts, query_texts, gt_pids_list


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Prepare BioASQ dense embeddings with E5-base-v2"
    )
    parser.add_argument("--input-dir", type=str, default=None,
                        help="Directory containing bioasq_gt_docs.jsonl and bioasq_gt_queries.jsonl "
                             "(produced by prepare_bioasq_colbertv2.py)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="BioASQ BEIR data directory (alternative to --input-dir)")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Output directory for binary files")
    parser.add_argument("--model", type=str, default="intfloat/e5-base-v2",
                        help="E5 model path (HuggingFace ID or local path)")
    parser.add_argument("--max-docs", type=int, default=-1,
                        help="Max documents to encode (-1 for all)")
    parser.add_argument("--max-queries", type=int, default=-1,
                        help="Max queries to encode (-1 for all)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Encoding batch size")
    parser.add_argument("--max-length", type=int, default=512,
                        help="Max token length for E5 encoding")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    if args.input_dir:
        # Read from JSONL files produced by prepare_bioasq_colbertv2.py
        input_dir = Path(args.input_dir)
        docs_jsonl = input_dir / "bioasq_gt_docs.jsonl"
        queries_jsonl = input_dir / "bioasq_gt_queries.jsonl"

        if not docs_jsonl.exists() or not queries_jsonl.exists():
            print(f"ERROR: JSONL files not found in {input_dir}")
            print(f"Expected: bioasq_gt_docs.jsonl, bioasq_gt_queries.jsonl")
            print(f"Run prepare_bioasq_colbertv2.py first, or use --data-dir for direct loading.")
            return

        print("=== Loading from JSONL files ===")
        print(f"  Docs: {docs_jsonl}")
        print(f"  Queries: {queries_jsonl}")

        doc_texts, _ = load_texts_from_jsonl(str(docs_jsonl), args.max_docs)
        query_texts, gt_pids = load_texts_from_jsonl(str(queries_jsonl), args.max_queries)

        print(f"Loaded {len(doc_texts)} documents, {len(query_texts)} queries")

    elif args.data_dir:
        # Load directly from BioASQ BEIR format
        data_dir = Path(args.data_dir)
        if not (data_dir / "corpus.jsonl").exists():
            print(f"ERROR: corpus.jsonl not found in {data_dir}")
            return

        print("=== Loading directly from BioASQ BEIR data ===")
        max_q = args.max_queries if args.max_queries > 0 else 1000
        max_d = args.max_docs if args.max_docs > 0 else 10000
        doc_texts, query_texts, gt_pids = load_bioasq_direct(data_dir, max_q, max_d)

        print(f"Loaded {len(doc_texts)} documents, {len(query_texts)} queries")

    else:
        print("ERROR: Must provide either --input-dir (JSONL files) or --data-dir (BEIR data)")
        return

    # Load E5-base-v2
    print(f"\n=== Loading E5-base-v2 ===")
    encoder = E5Encoder(args.model)

    # Encode documents
    print(f"\n=== Encoding {len(doc_texts)} documents ===")
    doc_embeddings = encoder.encode(
        doc_texts, args.batch_size, args.max_length, prefix="passage: "
    )

    doc_output = output_dir / "bioasq_e5_docs.bin"
    save_dense_binary(doc_embeddings, str(doc_output))
    del doc_embeddings

    # Encode queries
    print(f"\n=== Encoding {len(query_texts)} queries ===")
    query_embeddings = encoder.encode(
        query_texts, args.batch_size, args.max_length, prefix="query: "
    )

    query_output = output_dir / "bioasq_e5_queries.bin"
    save_dense_binary(query_embeddings, str(query_output))

    # Save GT labels
    gt_output = output_dir / "bioasq_e5_gt.bin"
    save_gt_binary(gt_pids, str(gt_output))

    # Summary
    print(f"\n=== Summary ===")
    print(f"Model: E5-base-v2 (dim={encoder.dim})")
    print(f"Documents: {len(doc_texts)}, dim={encoder.dim}")
    print(f"Queries: {len(query_texts)}, dim={query_embeddings.shape[1]}")

    gt_sizes = [len(g) for g in gt_pids]
    print(f"GT/query: min={min(gt_sizes)}, max={max(gt_sizes)}, avg={np.mean(gt_sizes):.1f}")

    no_gt = sum(1 for g in gt_pids if len(g) == 0)
    if no_gt > 0:
        print(f"WARNING: {no_gt} queries have no GT!")

    print(f"\nOutput files:")
    print(f"  Documents: {doc_output}")
    print(f"  Queries: {query_output}")
    print(f"  GT labels: {gt_output}")

    print(f"\nTo run dense test:")
    print(f"  ./tests/ut/knowhere_tests '[dense_vs_multivec][bioasq]'")


if __name__ == "__main__":
    main()
