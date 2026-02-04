#!/usr/bin/env python3
"""
Prepare LoTTE ColBERT data with official ground truth annotations.

This script:
1. Selects 100 queries
2. Collects their annotated relevant documents (up to 100 per query)
3. Pads to 10000 documents with random distractors if needed
4. Encodes with ColBERT

Usage:
    python scripts/prepare_lotte_with_gt.py --domain science --output-dir build
"""

import argparse
import json
import os
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Set, Tuple

# ColBERT max tokens per passage
COLBERT_DOC_MAXLEN = 180


def split_document_into_passages(text: str, max_tokens: int = COLBERT_DOC_MAXLEN) -> List[str]:
    """Split a long document into passages."""
    words = text.split()
    words_per_passage = int(max_tokens / 1.3)

    passages = []
    for i in range(0, len(words), words_per_passage):
        passage = " ".join(words[i:i + words_per_passage])
        if passage.strip():
            passages.append(passage)

    if not passages:
        passages = [text[:500] if text else "empty"]

    return passages


def load_qas(qas_path: str, max_queries: int = 100) -> List[Dict]:
    """Load query-answer annotations."""
    qas = []
    with open(qas_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_queries:
                break
            qas.append(json.loads(line))
    return qas


def collect_relevant_pids(qas: List[Dict], max_per_query: int = 100) -> Tuple[Set[int], Dict[int, List[int]]]:
    """Collect relevant document PIDs from annotations."""
    all_relevant_pids = set()
    query_to_pids = {}

    for qa in qas:
        qid = qa['qid']
        pids = qa['answer_pids'][:max_per_query]
        query_to_pids[qid] = pids
        all_relevant_pids.update(pids)

    return all_relevant_pids, query_to_pids


def load_documents_by_pids(collection_path: str, target_pids: Set[int],
                           total_docs: int, max_total: int = 10000) -> Tuple[Dict[int, str], List[int]]:
    """Load specific documents by PID, pad with random if needed."""
    docs = {}
    all_pids = []

    print(f"Loading {len(target_pids)} relevant documents...")

    with open(collection_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            parts = line.strip().split('\t', 1)
            if len(parts) != 2:
                continue
            pid = int(parts[0])
            text = parts[1]

            if pid in target_pids:
                docs[pid] = text
                all_pids.append(pid)

            if len(docs) == len(target_pids):
                break

    print(f"Loaded {len(docs)} relevant documents")

    # Pad with random documents if needed
    if len(docs) < max_total:
        needed = max_total - len(docs)
        print(f"Padding with {needed} random distractors...")

        random.seed(42)
        # Sample random PIDs not in relevant set
        candidate_pids = [i for i in range(total_docs) if i not in target_pids]
        random_pids = set(random.sample(candidate_pids, min(needed, len(candidate_pids))))

        with open(collection_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if len(docs) >= max_total:
                    break
                parts = line.strip().split('\t', 1)
                if len(parts) != 2:
                    continue
                pid = int(parts[0])

                if pid in random_pids and pid not in docs:
                    docs[pid] = parts[1]
                    all_pids.append(pid)

        print(f"Total documents: {len(docs)}")

    return docs, all_pids


def encode_with_colbert(texts: List[str], checkpoint, is_query: bool = False, batch_size: int = 32):
    """Encode texts with ColBERT."""
    import torch

    all_embeddings = []

    for i in range(0, len(texts), batch_size):
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
    parser = argparse.ArgumentParser(description="Prepare LoTTE with GT annotations")
    parser.add_argument("--domain", type=str, default="science")
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--max-queries", type=int, default=100)
    parser.add_argument("--max-gt-per-query", type=int, default=100)
    parser.add_argument("--max-docs", type=int, default=10000)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    lotte_dir = output_dir / "lotte" / args.domain / "dev"

    collection_path = lotte_dir / "collection.tsv"
    qas_path = lotte_dir / "qas.forum.jsonl"
    questions_path = lotte_dir / "questions.forum.tsv"

    # Check files exist
    for p in [collection_path, qas_path, questions_path]:
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")

    # Get total document count
    total_docs = sum(1 for _ in open(collection_path, 'r', encoding='utf-8'))
    print(f"Total documents in collection: {total_docs}")

    # Step 1: Load query annotations
    print(f"\n=== Loading {args.max_queries} queries with GT ===")
    qas = load_qas(str(qas_path), args.max_queries)
    print(f"Loaded {len(qas)} queries")

    # Step 2: Collect relevant PIDs
    relevant_pids, query_to_pids = collect_relevant_pids(qas, args.max_gt_per_query)
    print(f"Total relevant documents: {len(relevant_pids)}")

    # Step 3: Load documents
    print(f"\n=== Loading documents ===")
    docs, all_pids = load_documents_by_pids(
        str(collection_path), relevant_pids, total_docs, args.max_docs
    )

    # Create PID to new index mapping
    pid_to_idx = {pid: idx for idx, pid in enumerate(all_pids)}

    # Step 4: Load ColBERT
    print(f"\n=== Loading ColBERT ===")
    import torch
    from colbert.infra import ColBERTConfig
    from colbert.modeling.checkpoint import Checkpoint

    print(f"CUDA available: {torch.cuda.is_available()}")

    config = ColBERTConfig(doc_maxlen=COLBERT_DOC_MAXLEN, query_maxlen=32)
    checkpoint = Checkpoint("colbert-ir/colbertv2.0", colbert_config=config)

    # Step 5: Encode documents
    print(f"\n=== Encoding {len(docs)} documents ===")
    from tqdm import tqdm

    doc_entries = []
    for new_idx, pid in enumerate(tqdm(all_pids, desc="Encoding docs")):
        text = docs[pid]
        passages = split_document_into_passages(text)

        # Encode passages
        passage_embs = encode_with_colbert(passages, checkpoint, is_query=False)
        vectors = np.vstack(passage_embs)

        chunks = [{"pos": j, "emb": vec.tolist()} for j, vec in enumerate(vectors)]

        doc_entries.append({
            "pid": new_idx,
            "original_pid": pid,
            "text": text[:500],
            "chunks": chunks
        })

    # Step 6: Encode queries
    print(f"\n=== Encoding {len(qas)} queries ===")
    query_entries = []

    for qa in tqdm(qas, desc="Encoding queries"):
        query_text = qa['query']

        embs = encode_with_colbert([query_text], checkpoint, is_query=True)
        vectors = embs[0]

        chunks = [{"pos": j, "emb": vec.tolist()} for j, vec in enumerate(vectors)]

        # Remap GT pids to new indices
        gt_new_indices = [pid_to_idx[pid] for pid in qa['answer_pids'] if pid in pid_to_idx]

        query_entries.append({
            "pid": qa['qid'],
            "text": query_text[:200],
            "chunks": chunks,
            "gt_pids": gt_new_indices  # Ground truth with new indices
        })

    # Step 7: Save
    print(f"\n=== Saving ===")
    doc_path = output_dir / f"lotte_{args.domain}_gt_docs.jsonl"
    query_path = output_dir / f"lotte_{args.domain}_gt_queries.jsonl"

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
    print(f"  Relevant: {len(relevant_pids)}, Distractors: {len(doc_entries) - len(relevant_pids)}")
    print(f"  Vectors/doc: min={min(vec_counts)}, max={max(vec_counts)}, avg={np.mean(vec_counts):.1f}")

    gt_counts = [len(q['gt_pids']) for q in query_entries]
    print(f"Queries: {len(query_entries)}")
    print(f"  GT/query: min={min(gt_counts)}, max={max(gt_counts)}, avg={np.mean(gt_counts):.1f}")

    print(f"\n=== Done! ===")
    print(f"Documents: {doc_path}")
    print(f"Queries: {query_path}")


if __name__ == "__main__":
    main()
