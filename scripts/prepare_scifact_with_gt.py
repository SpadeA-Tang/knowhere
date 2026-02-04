#!/usr/bin/env python3
"""
Prepare SciFact ColBERT data with official ground truth annotations.

This script:
1. Downloads SciFact dataset from BEIR
2. Uses all 5183 documents (scientific abstracts)
3. Selects 100 queries with GT annotations
4. Encodes with ColBERT

Usage:
    python scripts/prepare_scifact_with_gt.py --output-dir build

Requirements:
    pip install torch --index-url https://download.pytorch.org/whl/cu118
    pip install colbert-ai transformers tqdm numpy
"""

import argparse
import json
import os
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

SCIFACT_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
COLBERT_DOC_MAXLEN = 180


def download_scifact(output_dir: Path) -> Path:
    """Download SciFact dataset if not exists."""
    # Check multiple possible locations
    possible_dirs = [
        output_dir / "scifact",
        Path("/tmp/scifact"),
    ]

    for scifact_dir in possible_dirs:
        if scifact_dir.exists() and (scifact_dir / "corpus.jsonl").exists():
            print(f"Using cached SciFact: {scifact_dir}")
            return scifact_dir

    # Need to download
    scifact_dir = output_dir / "scifact"
    zip_path = output_dir / "scifact.zip"
    if not zip_path.exists():
        print(f"Downloading SciFact from {SCIFACT_URL}...")
        urllib.request.urlretrieve(SCIFACT_URL, zip_path)
        print(f"Saved to {zip_path}")

    print(f"Extracting to {output_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(output_dir)

    return scifact_dir


def load_corpus(scifact_dir: Path) -> Tuple[Dict[str, str], List[str]]:
    """Load all documents from corpus."""
    corpus_path = scifact_dir / "corpus.jsonl"

    docs = {}
    all_pids = []

    print(f"Loading corpus from {corpus_path}...")
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            pid = doc['_id']
            # Combine title and text
            title = doc.get('title', '')
            text = doc.get('text', '')
            full_text = f"{title} {text}".strip() if title else text

            docs[pid] = full_text
            all_pids.append(pid)

    print(f"Loaded {len(docs)} documents")
    return docs, all_pids


def load_queries_and_qrels(scifact_dir: Path, max_queries: int) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """Load queries and their ground truth annotations."""
    queries_path = scifact_dir / "queries.jsonl"
    qrels_path = scifact_dir / "qrels" / "test.tsv"

    # Load qrels first to know which queries have GT
    print(f"Loading qrels from {qrels_path}...")
    qrels = {}
    with open(qrels_path, 'r', encoding='utf-8') as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                qid, did, rel = parts[0], parts[1], int(parts[2])
                if rel > 0:
                    if qid not in qrels:
                        qrels[qid] = []
                    qrels[qid].append(did)

    print(f"Found {len(qrels)} queries with GT annotations")

    # Load query texts
    print(f"Loading queries from {queries_path}...")
    all_queries = {}
    with open(queries_path, 'r', encoding='utf-8') as f:
        for line in f:
            q = json.loads(line)
            all_queries[q['_id']] = q['text']

    # Select queries that have GT
    selected_qids = [qid for qid in qrels.keys() if qid in all_queries][:max_queries]

    queries = {qid: all_queries[qid] for qid in selected_qids}
    selected_qrels = {qid: qrels[qid] for qid in selected_qids}

    print(f"Selected {len(queries)} queries with GT")

    return queries, selected_qrels


def encode_with_colbert(texts: List[str], checkpoint, is_query: bool = False, batch_size: int = 32):
    """Encode texts with ColBERT."""
    import torch
    from tqdm import tqdm

    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        batch = texts[i:i + batch_size]
        with torch.no_grad():
            if is_query:
                embs = checkpoint.queryFromText(batch)
            else:
                embs = checkpoint.docFromText(batch)

            for emb in embs:
                norms = torch.norm(emb, dim=-1)
                mask = norms > 1e-6
                valid_emb = emb[mask].cpu().numpy()

                if len(valid_emb) == 0:
                    valid_emb = emb[0:1].cpu().numpy()

                all_embeddings.append(valid_emb)

    return all_embeddings


def main():
    parser = argparse.ArgumentParser(description="Prepare SciFact with GT annotations")
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--max-queries", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Download SciFact
    print("=== Step 1: Downloading SciFact ===")
    scifact_dir = download_scifact(output_dir)

    # Step 2: Load corpus (all documents)
    print(f"\n=== Step 2: Loading corpus ===")
    docs, all_pids = load_corpus(scifact_dir)

    # Create PID to new index mapping
    pid_to_idx = {pid: idx for idx, pid in enumerate(all_pids)}

    # Step 3: Load queries with GT
    print(f"\n=== Step 3: Loading {args.max_queries} queries with GT ===")
    queries, qrels = load_queries_and_qrels(scifact_dir, args.max_queries)
    selected_qids = list(queries.keys())

    # Step 4: Load ColBERT
    print(f"\n=== Step 4: Loading ColBERT ===")
    import torch
    from colbert.infra import ColBERTConfig
    from colbert.modeling.checkpoint import Checkpoint

    print(f"CUDA available: {torch.cuda.is_available()}")

    config = ColBERTConfig(doc_maxlen=COLBERT_DOC_MAXLEN, query_maxlen=32)
    checkpoint = Checkpoint("colbert-ir/colbertv2.0", colbert_config=config)

    # Step 5: Encode documents
    print(f"\n=== Step 5: Encoding {len(all_pids)} documents ===")

    doc_texts = [docs[pid] for pid in all_pids]
    doc_embs = encode_with_colbert(doc_texts, checkpoint, is_query=False, batch_size=args.batch_size)

    doc_entries = []
    for new_idx, (pid, emb) in enumerate(zip(all_pids, doc_embs)):
        chunks = [{"pos": j, "emb": vec.tolist()} for j, vec in enumerate(emb)]
        doc_entries.append({
            "pid": new_idx,
            "original_pid": pid,
            "text": docs[pid][:500],
            "chunks": chunks
        })

    # Step 6: Encode queries
    print(f"\n=== Step 6: Encoding {len(selected_qids)} queries ===")

    query_texts = [queries[qid] for qid in selected_qids]
    query_embs = encode_with_colbert(query_texts, checkpoint, is_query=True, batch_size=args.batch_size)

    query_entries = []
    for i, (qid, emb) in enumerate(zip(selected_qids, query_embs)):
        chunks = [{"pos": j, "emb": vec.tolist()} for j, vec in enumerate(emb)]

        # Remap GT pids to new indices
        gt_new_indices = [pid_to_idx[pid] for pid in qrels[qid] if pid in pid_to_idx]

        query_entries.append({
            "pid": i,
            "original_qid": qid,
            "text": queries[qid][:200],
            "chunks": chunks,
            "gt_pids": gt_new_indices
        })

    # Step 7: Save
    print(f"\n=== Step 7: Saving ===")
    doc_path = output_dir / "scifact_gt_docs.jsonl"
    query_path = output_dir / "scifact_gt_queries.jsonl"

    with open(doc_path, 'w') as f:
        for doc in doc_entries:
            f.write(json.dumps(doc) + '\n')
    print(f"Saved {len(doc_entries)} docs to {doc_path}")
    print(f"  File size: {os.path.getsize(doc_path) / 1e6:.1f} MB")

    with open(query_path, 'w') as f:
        for q in query_entries:
            f.write(json.dumps(q) + '\n')
    print(f"Saved {len(query_entries)} queries to {query_path}")

    # Print statistics
    print(f"\n=== Statistics ===")
    vec_counts = [len(d['chunks']) for d in doc_entries]
    print(f"Docs: {len(doc_entries)}")
    print(f"  Vectors/doc: min={min(vec_counts)}, max={max(vec_counts)}, avg={np.mean(vec_counts):.1f}")
    print(f"  P50={np.percentile(vec_counts, 50):.0f}, P90={np.percentile(vec_counts, 90):.0f}")

    query_vec_counts = [len(q['chunks']) for q in query_entries]
    gt_counts = [len(q['gt_pids']) for q in query_entries]
    print(f"Queries: {len(query_entries)}")
    print(f"  Vectors/query: min={min(query_vec_counts)}, max={max(query_vec_counts)}, avg={np.mean(query_vec_counts):.1f}")
    print(f"  GT/query: min={min(gt_counts)}, max={max(gt_counts)}, avg={np.mean(gt_counts):.1f}")

    # Check queries with no GT in final dataset
    no_gt = sum(1 for q in query_entries if len(q['gt_pids']) == 0)
    if no_gt > 0:
        print(f"  WARNING: {no_gt} queries have no GT in final dataset!")

    print(f"\n=== Done! ===")
    print(f"Documents: {doc_path}")
    print(f"Queries: {query_path}")


if __name__ == "__main__":
    main()
