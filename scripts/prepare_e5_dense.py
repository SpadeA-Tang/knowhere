#!/usr/bin/env python3
"""
Prepare dense embeddings using E5-base-v2 for all datasets.

Reads from JSONL files produced by the per-dataset ColBERT preparation scripts
to ensure the same documents and queries are used for both dense and multi-vector
evaluation.

Supported datasets: msmarco, scifact, lotte, trec_covid

Usage:
    python scripts/prepare_e5_dense.py --input-dir build --output-dir build --dataset msmarco
    python scripts/prepare_e5_dense.py --input-dir build --output-dir build --dataset scifact
    python scripts/prepare_e5_dense.py --input-dir build --output-dir build --dataset lotte
    python scripts/prepare_e5_dense.py --input-dir build --output-dir build --dataset trec_covid

Requirements:
    pip install torch transformers tqdm numpy
"""

import argparse
import json
import os
import struct
from pathlib import Path
from typing import List

import numpy as np
import torch
from tqdm import tqdm


# ============================================================================
# Dataset configuration
# ============================================================================

DATASET_FILES = {
    "msmarco": ("msmarco_gt_docs.jsonl", "msmarco_gt_queries.jsonl"),
    "scifact": ("scifact_gt_docs.jsonl", "scifact_gt_queries.jsonl"),
    "lotte": ("lotte_science_gt_docs.jsonl", "lotte_science_gt_queries.jsonl"),
    "trec_covid": ("trec_covid_gt_docs.jsonl", "trec_covid_gt_queries.jsonl"),
}


# ============================================================================
# E5-base-v2 Encoder
# ============================================================================

class E5Encoder:
    def __init__(self, model_path="intfloat/e5-base-v2"):
        from transformers import AutoTokenizer, AutoModel

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading E5-base-v2 from {model_path} (device={self.device})...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(self.device).eval()
        self.dim = self.model.config.hidden_size
        print(f"E5-base-v2 loaded: dim={self.dim}")

    @staticmethod
    def _average_pool(last_hidden_state, attention_mask):
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    @torch.no_grad()
    def encode(self, texts, batch_size=32, max_length=512, prefix=""):
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
# Binary I/O
# ============================================================================

def save_dense_binary(embeddings: np.ndarray, output_path: str):
    dim = embeddings.shape[1]
    num_vectors = embeddings.shape[0]
    with open(output_path, 'wb') as f:
        f.write(struct.pack('i', dim))
        f.write(struct.pack('q', num_vectors))
        embeddings.tofile(f)
    print(f"Saved {num_vectors} vectors (dim={dim}) to {output_path}")
    print(f"  File size: {os.path.getsize(output_path) / 1e6:.1f} MB")


def save_gt_binary(gt_pids: List[List[int]], output_path: str):
    with open(output_path, 'wb') as f:
        f.write(struct.pack('q', len(gt_pids)))
        for gt in gt_pids:
            f.write(struct.pack('q', len(gt)))
            for pid in gt:
                f.write(struct.pack('q', pid))
    print(f"Saved GT for {len(gt_pids)} queries to {output_path}")


def save_gt_rels_binary(gt_pids: List[List[int]], gt_rels: List[dict], output_path: str):
    """Save graded relevance in binary format for nDCG computation.

    Format:
        int64: num_queries
        For each query:
            int64: num_gt
            For each GT doc:
                int64: pid
                int32: relevance grade (1=partial, 2=highly relevant)
    """
    with open(output_path, 'wb') as f:
        f.write(struct.pack('q', len(gt_pids)))
        for pids, rels in zip(gt_pids, gt_rels):
            f.write(struct.pack('q', len(pids)))
            for pid in pids:
                rel = rels.get(str(pid), 1)
                f.write(struct.pack('q', pid))
                f.write(struct.pack('i', rel))
    print(f"Saved graded GT for {len(gt_pids)} queries to {output_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Prepare dense embeddings with E5-base-v2 for any dataset"
    )
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Directory with *_gt_docs.jsonl and *_gt_queries.jsonl")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Output directory for binary files")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=list(DATASET_FILES.keys()),
                        help="Dataset to process")
    parser.add_argument("--model", type=str, default="intfloat/e5-base-v2",
                        help="E5 model path or HuggingFace ID")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)

    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = args.dataset
    docs_file, queries_file = DATASET_FILES[dataset]
    docs_jsonl = input_dir / docs_file
    queries_jsonl = input_dir / queries_file

    if not docs_jsonl.exists() or not queries_jsonl.exists():
        print(f"ERROR: JSONL files not found in {input_dir}")
        print(f"Expected: {docs_file}, {queries_file}")
        print(f"Run the corresponding ColBERT preparation script first.")
        return

    # Load texts from JSONL
    print(f"=== Loading texts from JSONL ({dataset}) ===")
    doc_texts = []
    with open(docs_jsonl, 'r') as f:
        for line in f:
            item = json.loads(line)
            doc_texts.append(item['text'])
    print(f"Loaded {len(doc_texts)} documents")

    query_texts = []
    gt_pids = []
    gt_rels = []
    with open(queries_jsonl, 'r') as f:
        for line in f:
            item = json.loads(line)
            query_texts.append(item['text'])
            gt_pids.append(item['gt_pids'])
            gt_rels.append(item.get('gt_rels', {}))
    has_rels = any(len(r) > 0 for r in gt_rels)
    print(f"Loaded {len(query_texts)} queries (graded relevance: {'yes' if has_rels else 'no'})")

    # Load E5
    print(f"\n=== Loading E5-base-v2 ===")
    encoder = E5Encoder(args.model)

    # Encode documents
    print(f"\n=== Encoding {len(doc_texts)} documents ===")
    doc_embeddings = encoder.encode(doc_texts, args.batch_size, args.max_length, prefix="passage: ")
    save_dense_binary(doc_embeddings, str(output_dir / f"{dataset}_e5_docs.bin"))
    del doc_embeddings

    # Encode queries
    print(f"\n=== Encoding {len(query_texts)} queries ===")
    query_embeddings = encoder.encode(query_texts, args.batch_size, args.max_length, prefix="query: ")
    save_dense_binary(query_embeddings, str(output_dir / f"{dataset}_e5_queries.bin"))

    # Save GT (binary pids)
    save_gt_binary(gt_pids, str(output_dir / f"{dataset}_e5_gt.bin"))

    # Save graded relevance (for nDCG)
    if has_rels:
        save_gt_rels_binary(gt_pids, gt_rels, str(output_dir / f"{dataset}_e5_gt_rels.bin"))

    # Summary
    print(f"\n=== Summary ===")
    print(f"Model: E5-base-v2 (dim={encoder.dim})")
    print(f"Dataset: {dataset}")
    print(f"Documents: {len(doc_texts)}")
    print(f"Queries: {len(query_texts)}")
    gt_sizes = [len(g) for g in gt_pids]
    print(f"GT/query: min={min(gt_sizes)}, max={max(gt_sizes)}, avg={np.mean(gt_sizes):.1f}")

    if has_rels:
        rel1 = sum(1 for rels in gt_rels for r in rels.values() if r == 1)
        rel2 = sum(1 for rels in gt_rels for r in rels.values() if r == 2)
        print(f"Graded relevance: rel=1 (partial): {rel1}, rel=2 (highly relevant): {rel2}")

    print(f"\nOutput files:")
    print(f"  {output_dir / f'{dataset}_e5_docs.bin'}")
    print(f"  {output_dir / f'{dataset}_e5_queries.bin'}")
    print(f"  {output_dir / f'{dataset}_e5_gt.bin'}")
    if has_rels:
        print(f"  {output_dir / f'{dataset}_e5_gt_rels.bin'}  (graded, for nDCG)")


if __name__ == "__main__":
    main()
