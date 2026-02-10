#!/usr/bin/env python3
"""
Prepare MS MARCO BGE-M3 multi-vector data with hybrid document sampling.

MS MARCO has extremely sparse relevance judgments (~1 GT doc per query out of 8.8M),
so pure random sampling yields almost no queries with GT coverage. Instead we use
a hybrid approach:
  1. Select queries and guarantee their GT docs are in the sample
  2. Fill remaining slots with random docs from the full corpus
  3. GT docs are a tiny fraction (~2%) so bias is negligible

Usage:
    python scripts/prepare_msmarco_with_gt.py --output-dir build --collection build/collection.tsv
    python scripts/prepare_msmarco_with_gt.py --output-dir build --max-docs 5000 --max-queries 100

Requirements:
    pip install torch --index-url https://download.pytorch.org/whl/cu118
    pip install FlagEmbedding transformers datasets tqdm numpy
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
from tqdm import tqdm

QRELS_URL = "https://msmarco.z22.web.core.windows.net/msmarcoranking/qrels.dev.small.tsv"


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


def load_qrels(qrels_path: Path) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, int]]]:
    """Load qrels: query_id -> list of relevant doc_ids, and graded relevance.

    Returns (qrels, qrel_grades).
    qrels[qid] = [pid1, pid2, ...]  (docs with rel > 0)
    qrel_grades[qid][pid] = relevance grade (binary 1 for MS MARCO)
    """
    qrels = defaultdict(list)
    qrel_grades = defaultdict(dict)
    with open(qrels_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                qid, _, pid, rel = parts
                if int(rel) > 0:
                    qrels[qid].append(pid)
                    qrel_grades[qid][pid] = int(rel)
    print(f"Loaded qrels: {len(qrels)} queries with relevance judgments")
    return dict(qrels), dict(qrel_grades)


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


def select_documents_hybrid(collection_path: str, gt_pids: Set[str], max_docs: int
                            ) -> Tuple[Dict[str, str], List[str]]:
    """Hybrid sampling: include all GT docs + random fill from the full corpus.

    Two-pass approach:
    - Pass 1: count total lines, build line-number -> pid mapping for GT docs
    - Pass 2: read GT docs + randomly sampled non-GT docs
    """
    print(f"Loading corpus from {collection_path}...")

    # Pass 1: count total lines and find GT doc line numbers
    total_lines = 0
    gt_line_numbers = set()
    pid_at_line = {}  # only for GT lines

    with open(collection_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            parts = line.strip().split('\t', 1)
            if len(parts) == 2:
                pid = parts[0]
                if pid in gt_pids:
                    gt_line_numbers.add(i)
                    pid_at_line[i] = pid
            total_lines += 1

    print(f"  Total corpus: {total_lines} docs")
    print(f"  GT docs found: {len(gt_line_numbers)}/{len(gt_pids)}")

    # Determine how many random docs to sample
    random_budget = max_docs - len(gt_line_numbers)
    if random_budget < 0:
        random_budget = 0
    non_gt_indices = [i for i in range(total_lines) if i not in gt_line_numbers]

    if random_budget >= len(non_gt_indices):
        random_indices = set(non_gt_indices)
    else:
        random_indices = set(random.sample(non_gt_indices, random_budget))

    selected_indices = gt_line_numbers | random_indices
    print(f"  Sampling: {len(gt_line_numbers)} GT + {len(random_indices)} random = {len(selected_indices)} total")

    # Pass 2: read selected docs
    docs = {}
    all_pids = []
    with open(collection_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, total=total_lines, desc="Reading docs")):
            if i not in selected_indices:
                continue
            parts = line.strip().split('\t', 1)
            if len(parts) != 2:
                continue
            pid, text = parts
            docs[pid] = text
            all_pids.append(pid)

    print(f"  Loaded {len(docs)} docs")
    return docs, all_pids


def main():
    parser = argparse.ArgumentParser(description="Prepare MS MARCO with GT annotations (hybrid sampling)")
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--max-queries", type=int, default=100)
    parser.add_argument("--max-docs", type=int, default=5000)
    parser.add_argument("--collection", type=str, default=None,
                        help="Path to local collection.tsv (download from MS MARCO if not using HuggingFace)")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check for local collection file
    collection_path = args.collection
    if collection_path is None:
        default_path = output_dir / "collection.tsv"
        if default_path.exists():
            collection_path = str(default_path)
            print(f"Found local collection: {collection_path}")

    if collection_path is None:
        print("ERROR: No collection.tsv found. Download it:")
        print("  wget https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz")
        print("  tar -xzf collection.tar.gz")
        return

    # Step 1: Download and load qrels
    print("=== Step 1: Loading qrels ===")
    qrels_path = download_qrels(output_dir)
    qrels, qrel_grades = load_qrels(qrels_path)

    # Step 2: Select queries and load their texts
    print(f"\n=== Step 2: Selecting {args.max_queries} queries ===")
    sorted_qids = sorted(qrels.keys(), key=lambda x: int(x))
    candidate_qids = sorted_qids[:args.max_queries * 2]  # load extra pool
    query_texts = load_queries_from_hf(set(candidate_qids), args.max_queries * 2)

    # Pick queries that we have text for
    selected_qids = []
    for qid in candidate_qids:
        if qid in query_texts and len(selected_qids) < args.max_queries:
            selected_qids.append(qid)

    # Collect all GT pids for selected queries
    gt_pids = set()
    for qid in selected_qids:
        gt_pids.update(qrels[qid])
    print(f"Selected {len(selected_qids)} queries, {len(gt_pids)} unique GT docs")

    # Step 3: Hybrid sampling — GT docs + random fill
    print(f"\n=== Step 3: Hybrid sampling ({len(gt_pids)} GT + random fill to {args.max_docs}) ===")
    docs, all_pids = select_documents_hybrid(collection_path, gt_pids, args.max_docs)
    pid_to_idx = {pid: idx for idx, pid in enumerate(all_pids)}

    gt_in_sample = gt_pids & set(all_pids)
    print(f"  GT docs in sample: {len(gt_in_sample)}/{len(gt_pids)}")
    print(f"  GT ratio: {len(gt_in_sample)/len(all_pids)*100:.1f}%")

    # Step 4: Load BGE-M3
    print(f"\n=== Step 4: Loading BGE-M3 ===")
    from FlagEmbedding import BGEM3FlagModel

    model = BGEM3FlagModel('/home/spadea/models/bge-m3', use_fp16=True)

    # Step 5: Encode passages (streaming write)
    print(f"\n=== Step 5: Encoding {len(all_pids)} passages ===")

    doc_path = output_dir / "msmarco_gt_docs.jsonl"
    batch_size = 12
    passage_texts = [docs[pid] for pid in all_pids]
    doc_vec_counts = []
    num_docs = 0

    with open(doc_path, 'w') as f:
        for i in tqdm(range(0, len(passage_texts), batch_size), desc="Encoding docs"):
            batch = passage_texts[i:i + batch_size]
            output = model.encode(batch, batch_size=batch_size, max_length=512,
                                  return_colbert_vecs=True)

            for j, vecs in enumerate(output['colbert_vecs']):
                norms = np.linalg.norm(vecs, axis=-1)
                mask = norms > 1e-6
                valid = vecs[mask]
                if len(valid) == 0:
                    valid = vecs[0:1]

                new_idx = i + j
                pid = all_pids[new_idx]
                chunks = [{"pos": k, "emb": [round(float(x), 6) for x in vec]} for k, vec in enumerate(valid)]
                entry = {
                    "pid": new_idx,
                    "original_pid": pid,
                    "text": docs[pid][:500],
                    "chunks": chunks
                }
                f.write(json.dumps(entry) + '\n')
                doc_vec_counts.append(len(chunks))
                num_docs += 1

    del passage_texts
    print(f"Saved {num_docs} docs to {doc_path}")
    print(f"  File size: {os.path.getsize(doc_path) / 1e6:.1f} MB")

    # Step 6: Encode queries (streaming write)
    print(f"\n=== Step 6: Encoding {len(selected_qids)} queries ===")

    query_path = output_dir / "msmarco_gt_queries.jsonl"
    query_list = [query_texts[qid] for qid in selected_qids]
    query_vec_counts = []
    gt_counts = []

    with open(query_path, 'w') as f:
        for i in tqdm(range(0, len(query_list), batch_size), desc="Encoding queries"):
            batch = query_list[i:i + batch_size]
            output = model.encode(batch, batch_size=batch_size, max_length=512,
                                  return_colbert_vecs=True)

            for j, vecs in enumerate(output['colbert_vecs']):
                norms = np.linalg.norm(vecs, axis=-1)
                mask = norms > 1e-6
                valid = vecs[mask]
                if len(valid) == 0:
                    valid = vecs[0:1]

                idx = i + j
                qid = selected_qids[idx]
                chunks = [{"pos": k, "emb": [round(float(x), 6) for x in vec]} for k, vec in enumerate(valid)]
                gt_new_indices = [pid_to_idx[pid] for pid in qrels[qid] if pid in pid_to_idx]
                # Build graded relevance dict: {str(new_idx): grade}
                gt_rels_dict = {}
                for pid in qrels[qid]:
                    if pid in pid_to_idx and qid in qrel_grades and pid in qrel_grades[qid]:
                        gt_rels_dict[str(pid_to_idx[pid])] = qrel_grades[qid][pid]

                entry = {
                    "pid": idx,
                    "original_qid": qid,
                    "text": query_texts[qid][:200],
                    "chunks": chunks,
                    "gt_pids": gt_new_indices,
                    "gt_rels": gt_rels_dict
                }
                f.write(json.dumps(entry) + '\n')
                query_vec_counts.append(len(chunks))
                gt_counts.append(len(gt_new_indices))

    print(f"Saved {len(query_vec_counts)} queries to {query_path}")

    # Print statistics
    print(f"\n=== Statistics ===")
    print(f"Docs: {num_docs} (hybrid: {len(gt_in_sample)} GT + {num_docs - len(gt_in_sample)} random)")
    print(f"  GT ratio: {len(gt_in_sample)/num_docs*100:.1f}%")
    print(f"  Vectors/doc: min={min(doc_vec_counts)}, max={max(doc_vec_counts)}, avg={np.mean(doc_vec_counts):.1f}")

    print(f"Queries: {len(query_vec_counts)}")
    print(f"  Vectors/query: min={min(query_vec_counts)}, max={max(query_vec_counts)}, avg={np.mean(query_vec_counts):.1f}")
    print(f"  GT/query: min={min(gt_counts)}, max={max(gt_counts)}, avg={np.mean(gt_counts):.1f}")

    no_gt = sum(1 for c in gt_counts if c == 0)
    if no_gt > 0:
        print(f"  WARNING: {no_gt} queries have no GT in final dataset!")

    print(f"\n=== Done! ===")
    print(f"Documents: {doc_path}")
    print(f"Queries: {query_path}")
    print(f"\nNext: python scripts/prepare_e5_dense.py --input-dir {output_dir} --output-dir {output_dir} --dataset msmarco")


if __name__ == "__main__":
    main()
