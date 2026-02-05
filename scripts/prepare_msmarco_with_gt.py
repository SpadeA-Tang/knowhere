#!/usr/bin/env python3
"""
Prepare MS MARCO ColBERT data with official ground truth annotations.

This script:
1. Downloads MS MARCO qrels (relevance judgments)
2. Selects queries that have annotations
3. Collects their relevant documents + random distractors
4. Encodes with ColBERT

Usage:
    python scripts/prepare_msmarco_with_gt.py --output-dir build --max-queries 100 --max-docs 10000

Requirements:
    pip install torch --index-url https://download.pytorch.org/whl/cu118
    pip install colbert-ai transformers datasets tqdm numpy
"""

import argparse
import json
import os
import random
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

QRELS_URL = "https://msmarco.z22.web.core.windows.net/msmarcoranking/qrels.dev.small.tsv"
COLBERT_DOC_MAXLEN = 180


def download_qrels(output_dir: Path) -> Path:
    """Download MS MARCO qrels if not exists."""
    qrels_path = output_dir / "qrels.dev.small.tsv"
    if not qrels_path.exists():
        print(f"Downloading qrels from {QRELS_URL}...")
        urllib.request.urlretrieve(QRELS_URL, qrels_path)
        print(f"Saved to {qrels_path}")
    else:
        print(f"Using cached qrels: {qrels_path}")
    return qrels_path


def load_qrels(qrels_path: Path) -> Dict[str, List[str]]:
    """Load qrels: query_id -> list of relevant doc_ids."""
    qrels = defaultdict(list)
    with open(qrels_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                qid, _, pid, rel = parts
                if int(rel) > 0:
                    qrels[qid].append(pid)
    print(f"Loaded qrels: {len(qrels)} queries with relevance judgments")
    return dict(qrels)


def load_queries_from_hf(qids_needed: Set[str], max_queries: int) -> Dict[str, str]:
    """Load query texts from HuggingFace dataset."""
    from datasets import load_dataset

    print(f"Loading queries from HuggingFace (need {len(qids_needed)} queries)...")

    queries = {}
    ds = load_dataset('Tevatron/msmarco-passage', split='validation', streaming=True)

    for item in ds:
        qid = str(item['query_id'])
        if qid in qids_needed:
            queries[qid] = item['query']
            if len(queries) >= max_queries:
                break

    print(f"Loaded {len(queries)} queries")
    return queries


def load_passages_from_local(collection_path: str, pids_needed: Set[str], max_total: int) -> Tuple[Dict[str, str], List[str]]:
    """Load passage texts from local TSV file."""
    print(f"Loading passages from local file: {collection_path}")
    print(f"  Need {len(pids_needed)} relevant passages + distractors to reach {max_total}")

    passages = {}
    all_pids = []
    distractor_candidates = []

    with open(collection_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            parts = line.strip().split('\t', 1)
            if len(parts) != 2:
                continue
            pid, text = parts

            if pid in pids_needed:
                passages[pid] = text
                all_pids.append(pid)
            else:
                # Collect potential distractors (sample every 100th to save memory)
                if i % 100 == 0:
                    distractor_candidates.append((pid, text))

            if (i + 1) % 1000000 == 0:
                print(f"  Scanned {(i+1)//1000000}M passages, found {len(passages)} relevant...")

    print(f"Found {len(passages)}/{len(pids_needed)} relevant passages")

    # Add distractors to reach max_total
    if len(passages) < max_total:
        needed = max_total - len(passages)
        random.seed(42)
        random.shuffle(distractor_candidates)

        for pid, text in distractor_candidates[:needed]:
            if pid not in passages:
                passages[pid] = text
                all_pids.append(pid)

        print(f"Added {len(all_pids) - len([p for p in all_pids if p in pids_needed])} distractors")

    print(f"Total passages: {len(passages)}")
    return passages, all_pids


def load_passages_from_hf(pids_needed: Set[str], max_total: int) -> Tuple[Dict[str, str], List[str]]:
    """Load passage texts from HuggingFace dataset (SLOW - use local file instead)."""
    from datasets import load_dataset

    print(f"Loading passages from HuggingFace (need {len(pids_needed)} relevant + distractors)...")
    print(f"WARNING: This is very slow. Consider downloading collection.tsv locally.")

    passages = {}
    all_pids = []
    distractor_pids = []

    # Load corpus
    ds = load_dataset('Tevatron/msmarco-passage-corpus', split='train', streaming=True)

    for item in ds:
        pid = str(item['docid'])
        text = item['text']

        if pid in pids_needed:
            passages[pid] = text
            all_pids.append(pid)
        elif len(passages) < max_total and pid not in pids_needed:
            # Collect potential distractors
            distractor_pids.append((pid, text))

        if len(passages) >= len(pids_needed) and len(distractor_pids) >= max_total:
            break

    print(f"Found {len(passages)} relevant passages")

    # Add distractors to reach max_total
    if len(passages) < max_total:
        needed = max_total - len(passages)
        random.seed(42)
        random.shuffle(distractor_pids)

        for pid, text in distractor_pids[:needed]:
            if pid not in passages:
                passages[pid] = text
                all_pids.append(pid)

        print(f"Added {len(all_pids) - len(pids_needed.intersection(set(all_pids)))} distractors")

    print(f"Total passages: {len(passages)}")
    return passages, all_pids


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
    parser = argparse.ArgumentParser(description="Prepare MS MARCO with GT annotations")
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--max-queries", type=int, default=100)
    parser.add_argument("--max-docs", type=int, default=10000)
    parser.add_argument("--collection", type=str, default=None,
                        help="Path to local collection.tsv (download from MS MARCO if not using HuggingFace)")

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check for local collection file
    collection_path = args.collection
    if collection_path is None:
        # Try default location
        default_path = output_dir / "collection.tsv"
        if default_path.exists():
            collection_path = str(default_path)
            print(f"Found local collection: {collection_path}")

    # Step 1: Download and load qrels
    print("=== Step 1: Loading qrels ===")
    qrels_path = download_qrels(output_dir)
    qrels = load_qrels(qrels_path)

    # Step 2: Select queries with annotations
    print(f"\n=== Step 2: Selecting {args.max_queries} queries ===")
    # Sort by qid for reproducibility
    sorted_qids = sorted(qrels.keys(), key=lambda x: int(x))[:args.max_queries * 2]  # Load extra in case some missing

    # Collect all relevant pids
    relevant_pids = set()
    selected_qids = []
    query_to_pids = {}

    for qid in sorted_qids:
        if len(selected_qids) >= args.max_queries:
            break
        pids = qrels[qid]
        selected_qids.append(qid)
        query_to_pids[qid] = pids
        relevant_pids.update(pids)

    print(f"Selected {len(selected_qids)} queries")
    print(f"Total relevant passages: {len(relevant_pids)}")

    # Step 3: Load query texts
    print(f"\n=== Step 3: Loading query texts ===")
    queries = load_queries_from_hf(set(selected_qids), args.max_queries)

    # Filter to queries we actually found
    selected_qids = [qid for qid in selected_qids if qid in queries]
    print(f"Queries with text: {len(selected_qids)}")

    # Step 4: Load passages
    print(f"\n=== Step 4: Loading passages ===")
    if collection_path:
        passages, all_pids = load_passages_from_local(collection_path, relevant_pids, args.max_docs)
    else:
        print("No local collection.tsv found. Download it for faster loading:")
        print("  wget https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz")
        print("  tar -xzf collection.tar.gz")
        print("\nFalling back to HuggingFace (very slow)...")
        passages, all_pids = load_passages_from_hf(relevant_pids, args.max_docs)

    # Create PID to new index mapping
    pid_to_idx = {pid: idx for idx, pid in enumerate(all_pids)}

    # Step 5: Load ColBERT
    print(f"\n=== Step 5: Loading ColBERT ===")
    import torch
    from colbert.infra import ColBERTConfig
    from colbert.modeling.checkpoint import Checkpoint

    print(f"CUDA available: {torch.cuda.is_available()}")

    config = ColBERTConfig(doc_maxlen=COLBERT_DOC_MAXLEN, query_maxlen=32)
    checkpoint = Checkpoint("jinaai/jina-colbert-v2", colbert_config=config)

    # Step 6: Encode passages
    print(f"\n=== Step 6: Encoding {len(all_pids)} passages ===")

    passage_texts = [passages[pid] for pid in all_pids]
    passage_embs = encode_with_colbert(passage_texts, checkpoint, is_query=False)

    doc_entries = []
    for new_idx, (pid, emb) in enumerate(zip(all_pids, passage_embs)):
        chunks = [{"pos": j, "emb": vec.tolist()} for j, vec in enumerate(emb)]
        doc_entries.append({
            "pid": new_idx,
            "original_pid": pid,
            "text": passages[pid][:500],
            "chunks": chunks
        })

    # Step 7: Encode queries
    print(f"\n=== Step 7: Encoding {len(selected_qids)} queries ===")

    query_texts = [queries[qid] for qid in selected_qids]
    query_embs = encode_with_colbert(query_texts, checkpoint, is_query=True)

    query_entries = []
    for i, (qid, emb) in enumerate(zip(selected_qids, query_embs)):
        chunks = [{"pos": j, "emb": vec.tolist()} for j, vec in enumerate(emb)]

        # Remap GT pids to new indices
        gt_new_indices = [pid_to_idx[pid] for pid in query_to_pids[qid] if pid in pid_to_idx]

        query_entries.append({
            "pid": i,
            "original_qid": qid,
            "text": queries[qid][:200],
            "chunks": chunks,
            "gt_pids": gt_new_indices
        })

    # Step 8: Save
    print(f"\n=== Step 8: Saving ===")
    doc_path = output_dir / "msmarco_gt_docs.jsonl"
    query_path = output_dir / "msmarco_gt_queries.jsonl"

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
    print(f"  Relevant: {len(relevant_pids.intersection(set(all_pids)))}, Distractors: {len(doc_entries) - len(relevant_pids.intersection(set(all_pids)))}")
    print(f"  Vectors/doc: min={min(vec_counts)}, max={max(vec_counts)}, avg={np.mean(vec_counts):.1f}")

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
