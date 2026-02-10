#!/usr/bin/env python3
"""
Prepare LoTTE BGE-M3 multi-vector data with random document sampling.

This script:
1. Randomly samples N documents from the full LoTTE corpus
2. Selects queries that have sufficient GT coverage in the random sample
3. Encodes with BGE-M3 (ColBERT-style multi-vector mode)
4. Splits long documents into passages before encoding

Usage:
    python scripts/prepare_lotte_with_gt.py --domain science --output-dir build
    python scripts/prepare_lotte_with_gt.py --domain science --output-dir build --max-docs 5000
"""

import argparse
import json
import os
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Set, Tuple

from tqdm import tqdm

# BGE-M3 supports up to 8192 tokens, but we split long docs into manageable passages
BGE_M3_MAX_LENGTH = 512


def split_document_into_passages(text: str, words_per_passage: int = 300) -> List[str]:
    """Split a long document into passages."""
    words = text.split()

    passages = []
    for i in range(0, len(words), words_per_passage):
        passage = " ".join(words[i:i + words_per_passage])
        if passage.strip():
            passages.append(passage)

    if not passages:
        passages = [text[:500] if text else "empty"]

    return passages


def select_documents_random(collection_path: str, max_docs: int) -> Tuple[Dict[int, str], List[int]]:
    """Randomly sample max_docs from the full corpus.

    Two-pass approach:
    - Pass 1: count total lines
    - Pass 2: read only selected indices
    """
    print(f"Loading corpus from {collection_path}...")

    # Pass 1: count total lines
    total_lines = 0
    with open(collection_path, 'r', encoding='utf-8') as f:
        for _ in f:
            total_lines += 1
    print(f"  Total corpus: {total_lines} docs, sampling {max_docs}")

    # Generate random indices
    if max_docs >= total_lines:
        selected_indices = set(range(total_lines))
    else:
        selected_indices = set(random.sample(range(total_lines), max_docs))

    # Pass 2: read only selected docs
    docs = {}
    all_pids = []
    with open(collection_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, total=total_lines, desc="Sampling docs")):
            if i not in selected_indices:
                continue
            parts = line.strip().split('\t', 1)
            if len(parts) != 2:
                continue
            pid = int(parts[0])
            text = parts[1]
            docs[pid] = text
            all_pids.append(pid)

    print(f"  Sampled {len(docs)} docs")
    return docs, all_pids


def load_qas(qas_path: str, max_queries: int = 10000) -> List[Dict]:
    """Load all query-answer annotations (load a large pool for filtering)."""
    qas = []
    with open(qas_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_queries:
                break
            qas.append(json.loads(line))
    return qas


def encode_with_bgem3(texts: List[str], model, batch_size: int = 12, max_length: int = 512):
    """Encode texts with BGE-M3 ColBERT mode."""
    all_embeddings = []

    output = model.encode(texts, batch_size=batch_size, max_length=max_length,
                          return_colbert_vecs=True)

    for vecs in output['colbert_vecs']:
        # Filter out near-zero vectors (safety check)
        norms = np.linalg.norm(vecs, axis=-1)
        mask = norms > 1e-6
        valid = vecs[mask]
        if len(valid) == 0:
            valid = vecs[0:1]
        all_embeddings.append(valid)

    return all_embeddings


def main():
    parser = argparse.ArgumentParser(description="Prepare LoTTE with GT annotations (random sampling)")
    parser.add_argument("--domain", type=str, default="science")
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--max-queries", type=int, default=100)
    parser.add_argument("--max-docs", type=int, default=5000)
    parser.add_argument("--min-gt-per-query", type=int, default=1,
                        help="Min GT docs in sample for a query to be included (default: 1)")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    lotte_dir = output_dir / "lotte" / args.domain / "dev"

    collection_path = lotte_dir / "collection.tsv"
    qas_path = lotte_dir / "qas.forum.jsonl"
    questions_path = lotte_dir / "questions.forum.tsv"

    # Check files exist
    for p in [collection_path, qas_path, questions_path]:
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")

    # Step 1: Randomly sample documents from full corpus
    print(f"=== Step 1: Randomly sampling {args.max_docs} documents ===")
    docs, all_pids = select_documents_random(str(collection_path), args.max_docs)
    sampled_pids = set(all_pids)
    pid_to_idx = {pid: idx for idx, pid in enumerate(all_pids)}

    # Step 2: Load all query annotations
    print(f"\n=== Step 2: Loading query annotations ===")
    qas = load_qas(str(qas_path))
    print(f"Loaded {len(qas)} queries")

    # Step 3: Select queries with sufficient GT coverage in sample
    print(f"\n=== Step 3: Selecting queries (min {args.min_gt_per_query} GT in sample) ===")

    candidate_qas = []
    for qa in qas:
        pids = qa['answer_pids']
        gt_in_sample = sum(1 for pid in pids if pid in sampled_pids)
        if gt_in_sample >= args.min_gt_per_query:
            candidate_qas.append((qa, gt_in_sample, len(pids)))

    # Take up to max_queries
    selected_qas = [qa for qa, _, _ in candidate_qas[:args.max_queries]]

    gt_doc_ids = set()
    for qa in selected_qas:
        gt_doc_ids.update(pid for pid in qa['answer_pids'] if pid in sampled_pids)

    print(f"  {len(candidate_qas)} queries have >= {args.min_gt_per_query} GT, selected {len(selected_qas)}")
    for qa, gt_in, gt_total in candidate_qas[:min(10, args.max_queries)]:
        print(f"    Query {qa['qid']}: {gt_in}/{gt_total} GT in sample")
    if len(candidate_qas) > 10:
        print(f"    ... ({len(candidate_qas) - 10} more)")
    print(f"  GT docs in sample: {len(gt_doc_ids)}, "
          f"GT ratio: {len(gt_doc_ids)/len(docs)*100:.1f}%")

    # Step 4: Load BGE-M3
    print(f"\n=== Step 4: Loading BGE-M3 ===")
    from FlagEmbedding import BGEM3FlagModel

    model = BGEM3FlagModel('/home/spadea/models/bge-m3', use_fp16=True)

    # Step 5: Encode documents (streaming write)
    print(f"\n=== Step 5: Encoding {len(docs)} documents ===")

    doc_path = output_dir / f"lotte_{args.domain}_gt_docs.jsonl"
    doc_vec_counts = []
    num_docs = 0

    with open(doc_path, 'w') as f:
        for new_idx, pid in enumerate(tqdm(all_pids, desc="Encoding docs")):
            text = docs[pid]
            passages = split_document_into_passages(text)

            # Encode passages
            passage_embs = encode_with_bgem3(passages, model, max_length=BGE_M3_MAX_LENGTH)
            vectors = np.vstack(passage_embs)

            chunks = [{"pos": j, "emb": [round(float(x), 6) for x in vec]} for j, vec in enumerate(vectors)]

            entry = {
                "pid": new_idx,
                "original_pid": pid,
                "text": text[:500],
                "chunks": chunks
            }
            f.write(json.dumps(entry) + '\n')
            doc_vec_counts.append(len(chunks))
            num_docs += 1

    print(f"Saved {num_docs} docs to {doc_path}")
    print(f"  File size: {os.path.getsize(doc_path) / 1e6:.1f} MB")

    # Step 6: Encode queries (streaming write)
    print(f"\n=== Step 6: Encoding {len(selected_qas)} queries ===")

    query_path = output_dir / f"lotte_{args.domain}_gt_queries.jsonl"
    gt_counts = []

    with open(query_path, 'w') as f:
        for qa in tqdm(selected_qas, desc="Encoding queries"):
            query_text = qa['query']

            embs = encode_with_bgem3([query_text], model, max_length=BGE_M3_MAX_LENGTH)
            vectors = embs[0]

            chunks = [{"pos": j, "emb": [round(float(x), 6) for x in vec]} for j, vec in enumerate(vectors)]

            # Remap GT pids to new indices
            gt_new_indices = [pid_to_idx[pid] for pid in qa['answer_pids'] if pid in pid_to_idx]
            # Build graded relevance dict (binary for LoTTE)
            gt_rels_dict = {str(idx): 1 for idx in gt_new_indices}

            entry = {
                "pid": qa['qid'],
                "text": query_text[:200],
                "chunks": chunks,
                "gt_pids": gt_new_indices,
                "gt_rels": gt_rels_dict
            }
            f.write(json.dumps(entry) + '\n')
            gt_counts.append(len(gt_new_indices))

    print(f"Saved {len(gt_counts)} queries to {query_path}")

    # Print statistics
    print(f"\n=== Statistics ===")
    print(f"Docs: {num_docs} (randomly sampled)")
    print(f"  GT docs in sample: {len(gt_doc_ids)} ({len(gt_doc_ids)/num_docs*100:.1f}%)")
    print(f"  Non-GT docs: {num_docs - len(gt_doc_ids)} ({(num_docs - len(gt_doc_ids))/num_docs*100:.1f}%)")
    print(f"  Vectors/doc: min={min(doc_vec_counts)}, max={max(doc_vec_counts)}, avg={np.mean(doc_vec_counts):.1f}")

    print(f"Queries: {len(gt_counts)}")
    print(f"  GT/query: min={min(gt_counts)}, max={max(gt_counts)}, avg={np.mean(gt_counts):.1f}")

    no_gt = sum(1 for c in gt_counts if c == 0)
    if no_gt > 0:
        print(f"  WARNING: {no_gt} queries have no GT in final dataset!")

    print(f"\n=== Done! ===")
    print(f"Documents: {doc_path}")
    print(f"Queries: {query_path}")
    print(f"\nNext: python scripts/prepare_e5_dense.py --input-dir {output_dir} --output-dir {output_dir} --dataset lotte")


if __name__ == "__main__":
    main()
