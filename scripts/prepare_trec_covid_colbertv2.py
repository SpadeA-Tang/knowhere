#!/usr/bin/env python3
"""
Prepare TREC-COVID dataset with ColBERTv2 multi-vector embeddings.

This script:
1. Downloads TREC-COVID dataset from BEIR (freely available)
2. Randomly samples N documents from the full 171K corpus
3. Selects queries that have sufficient GT coverage in the sample
4. Encodes with ColBERTv2 (128-dim token-level embeddings)
5. Saves JSONL files compatible with knowhere multi-vector tests

Usage:
    python scripts/prepare_trec_covid_colbertv2.py --output-dir build
    python scripts/prepare_trec_covid_colbertv2.py --output-dir build --model /path/to/colbertv2

Requirements:
    pip install torch transformers safetensors huggingface_hub tqdm numpy
"""

import argparse
import json
import os
import random
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

TREC_COVID_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/trec-covid.zip"


# ============================================================================
# ColBERTv2 Encoder
# ============================================================================

class ColBERTv2Encoder:
    """ColBERTv2 encoder using raw transformers (no colbert-ai dependency)."""

    def __init__(self, model_path="colbert-ir/colbertv2.0"):
        from transformers import AutoTokenizer, AutoModel

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading ColBERTv2 from {model_path} (device={self.device})...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.bert = AutoModel.from_pretrained(model_path).to(self.device).eval()

        self.linear = nn.Linear(self.bert.config.hidden_size, 128, bias=False).to(self.device)
        self._load_linear_weights(model_path)

        self.q_marker_id = self.tokenizer.convert_tokens_to_ids("[unused0]")
        self.d_marker_id = self.tokenizer.convert_tokens_to_ids("[unused1]")
        self.mask_id = self.tokenizer.mask_token_id

        print(f"ColBERTv2 loaded: hidden={self.bert.config.hidden_size}, projection=128")

    def _load_linear_weights(self, model_path):
        loaded = False
        # Try safetensors
        try:
            from safetensors.torch import load_file
            if os.path.isdir(model_path):
                sf_path = os.path.join(model_path, "model.safetensors")
            else:
                from huggingface_hub import hf_hub_download
                sf_path = hf_hub_download(model_path, "model.safetensors")
            state_dict = load_file(sf_path)
            for key, value in state_dict.items():
                if 'linear' in key.lower() and 'weight' in key.lower():
                    self.linear.weight.data = value.to(self.device)
                    print(f"  Loaded linear projection from key: {key}")
                    loaded = True
                    break
        except Exception:
            pass

        # Try pytorch_model.bin
        if not loaded:
            try:
                if os.path.isdir(model_path):
                    pt_path = os.path.join(model_path, "pytorch_model.bin")
                else:
                    from huggingface_hub import hf_hub_download
                    pt_path = hf_hub_download(model_path, "pytorch_model.bin")
                state_dict = torch.load(pt_path, map_location="cpu", weights_only=True)
                for key, value in state_dict.items():
                    if 'linear' in key.lower() and 'weight' in key.lower():
                        self.linear.weight.data = value.to(self.device)
                        print(f"  Loaded linear projection from key: {key}")
                        loaded = True
                        break
            except Exception:
                pass

        if not loaded:
            print("  WARNING: Could not load linear projection weights!")

    @torch.no_grad()
    def encode_docs(self, texts, batch_size=32, max_length=512):
        """Encode documents: [CLS] [D] tok1...tokN [SEP] [PAD]...

        Standard ColBERT doc format. Output includes all attended tokens
        (everything except [PAD]).
        """
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        pad_id = self.tokenizer.pad_token_id

        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding docs"):
            batch = texts[i:i + batch_size]

            # Tokenize without special tokens
            encoded = self.tokenizer(
                batch, add_special_tokens=False, truncation=True,
                max_length=max_length - 3,  # reserve [CLS] + [D] + [SEP]
                return_attention_mask=False,
            )

            all_ids = []
            all_mask = []
            for token_ids in encoded['input_ids']:
                seq = [cls_id, self.d_marker_id] + token_ids + [sep_id]
                attn = [1] * len(seq)
                pad_len = max_length - len(seq)
                if pad_len > 0:
                    seq += [pad_id] * pad_len
                    attn += [0] * pad_len
                all_ids.append(seq[:max_length])
                all_mask.append(attn[:max_length])

            input_ids = torch.tensor(all_ids, dtype=torch.long, device=self.device)
            attention_mask = torch.tensor(all_mask, dtype=torch.long, device=self.device)

            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            token_embs = self.linear(outputs.last_hidden_state)
            token_embs = F.normalize(token_embs, p=2, dim=-1)

            for j in range(len(batch)):
                valid_embs = token_embs[j][attention_mask[j].bool()].cpu().numpy()
                all_embeddings.append(valid_embs)
        return all_embeddings

    @torch.no_grad()
    def encode_queries(self, texts, batch_size=32, max_length=32):
        """Encode queries: [CLS] [Q] tok1...tokN [SEP] [MASK]...[MASK]

        Standard ColBERT query format:
        - [MASK] tokens placed AFTER [SEP]
        - attention_mask = 1 for [CLS] through [SEP], 0 for [MASK]
        - Output ALL max_length embeddings (including [MASK] with attn=0)
        """
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id

        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding queries"):
            batch = texts[i:i + batch_size]

            # Tokenize without special tokens
            encoded = self.tokenizer(
                batch, add_special_tokens=False, truncation=True,
                max_length=max_length - 3,  # reserve [CLS] + [Q] + [SEP]
                return_attention_mask=False,
            )

            all_ids = []
            all_mask = []
            for token_ids in encoded['input_ids']:
                # [CLS] [Q] tok1 tok2 ... tokN [SEP] [MASK] [MASK] ... [MASK]
                real_seq = [cls_id, self.q_marker_id] + token_ids + [sep_id]
                attn = [1] * len(real_seq)
                # Pad remainder with [MASK] (attention_mask=0)
                num_mask = max_length - len(real_seq)
                if num_mask > 0:
                    real_seq += [self.mask_id] * num_mask
                    attn += [0] * num_mask
                all_ids.append(real_seq[:max_length])
                all_mask.append(attn[:max_length])

            input_ids = torch.tensor(all_ids, dtype=torch.long, device=self.device)
            attention_mask = torch.tensor(all_mask, dtype=torch.long, device=self.device)

            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            token_embs = self.linear(outputs.last_hidden_state)
            token_embs = F.normalize(token_embs, p=2, dim=-1)

            # Output ALL max_length embeddings (including [MASK] with attn=0)
            for j in range(len(batch)):
                all_embeddings.append(token_embs[j].cpu().numpy())
        return all_embeddings


# ============================================================================
# TREC-COVID Data Loading
# ============================================================================

def download_trec_covid(output_dir: Path) -> Path:
    """Download TREC-COVID from BEIR if not exists."""
    possible_dirs = [
        output_dir / "trec-covid",
        Path("/tmp/trec-covid"),
    ]
    for d in possible_dirs:
        if d.exists() and (d / "corpus.jsonl").exists():
            print(f"Using cached TREC-COVID: {d}")
            return d

    zip_path = output_dir / "trec-covid.zip"
    if not zip_path.exists():
        print(f"Downloading TREC-COVID from BEIR...")
        print(f"  URL: {TREC_COVID_URL}")
        urllib.request.urlretrieve(TREC_COVID_URL, zip_path)
        print(f"  Download complete: {os.path.getsize(zip_path) / 1e6:.1f} MB")

    print(f"Extracting to {output_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(output_dir)

    return output_dir / "trec-covid"


def load_qrels(qrels_dir: Path) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, int]]]:
    """Returns (qrels, qrel_grades).

    qrels[qid] = [did1, did2, ...]  (docs with rel > 0)
    qrel_grades[qid][did] = relevance grade (1 or 2 for TREC-COVID)
    """
    qrels = {}
    qrel_grades = {}
    for split in ["test.tsv", "train.tsv", "dev.tsv"]:
        qrels_path = qrels_dir / split
        if not qrels_path.exists():
            continue
        print(f"Loading qrels from {qrels_path}...")
        with open(qrels_path, 'r', encoding='utf-8') as f:
            next(f)  # skip header
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    qid, did = parts[0], parts[1]
                    rel = int(parts[2])
                    if rel > 0:
                        qrels.setdefault(qid, []).append(did)
                        qrel_grades.setdefault(qid, {})[did] = rel
        print(f"  {len(qrels)} queries with GT from {split}")
    return qrels, qrel_grades


def load_queries(queries_path: Path, needed_qids: Set[str]) -> Dict[str, str]:
    queries = {}
    with open(queries_path, 'r', encoding='utf-8') as f:
        for line in f:
            q = json.loads(line)
            qid = q['_id']
            if qid in needed_qids:
                # TREC-COVID queries may have metadata field
                text = q.get('text', '')
                metadata = q.get('metadata', {})
                if isinstance(metadata, dict) and metadata.get('query'):
                    text = metadata['query']
                queries[qid] = text
    print(f"  Found {len(queries)}/{len(needed_qids)} query texts")
    return queries


def select_documents_random(corpus_path: Path, max_docs: int
                            ) -> Tuple[Dict[str, str], List[str]]:
    """Randomly sample max_docs from the full corpus via reservoir sampling."""
    # Pass 1: count total lines for reservoir sampling
    print(f"Loading corpus from {corpus_path}...")
    total_lines = 0
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for _ in f:
            total_lines += 1
    print(f"  Total corpus: {total_lines} docs, sampling {max_docs}")

    # Generate sorted random indices for single-pass selection
    if max_docs >= total_lines:
        selected_indices = set(range(total_lines))
    else:
        selected_indices = set(random.sample(range(total_lines), max_docs))

    # Pass 2: read only selected docs
    docs = {}
    all_pids = []
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, total=total_lines, desc="Sampling docs")):
            if i not in selected_indices:
                continue
            doc = json.loads(line)
            pid = doc['_id']
            title = doc.get('title', '')
            text = doc.get('text', '')
            full_text = f"{title} {text}".strip() if title else text
            docs[pid] = full_text
            all_pids.append(pid)

    print(f"  Sampled {len(docs)} docs")
    return docs, all_pids


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Prepare TREC-COVID with ColBERTv2 multi-vector embeddings"
    )
    parser.add_argument("--data-dir", type=str, default=None,
                        help="TREC-COVID directory (BEIR format). Auto-downloads if not provided.")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Output directory for JSONL files")
    parser.add_argument("--model", type=str, default="colbert-ir/colbertv2.0",
                        help="ColBERTv2 model path or HuggingFace ID")
    parser.add_argument("--max-queries", type=int, default=50,
                        help="Max queries (TREC-COVID has ~50)")
    parser.add_argument("--max-docs", type=int, default=5000,
                        help="Max documents randomly sampled from corpus (default: 5000)")
    parser.add_argument("--min-gt-per-query", type=int, default=10,
                        help="Min GT docs in sample for a query to be included (default: 10)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--doc-max-length", type=int, default=512)
    parser.add_argument("--query-max-length", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Get TREC-COVID data
    print("=== Step 1: Loading TREC-COVID dataset ===")
    if args.data_dir:
        covid_dir = Path(args.data_dir)
        if not (covid_dir / "corpus.jsonl").exists():
            print(f"ERROR: corpus.jsonl not found in {covid_dir}")
            return
    else:
        covid_dir = download_trec_covid(output_dir)

    # Step 2: Load qrels
    print(f"\n=== Step 2: Loading qrels ===")
    qrels, qrel_grades = load_qrels(covid_dir / "qrels")
    if not qrels:
        print("ERROR: No qrels found!")
        return

    # Step 3: Randomly sample documents from full corpus
    print(f"\n=== Step 3: Randomly sampling {args.max_docs} documents ===")
    docs, all_pids = select_documents_random(covid_dir / "corpus.jsonl", args.max_docs)
    sampled_pids = set(all_pids)
    pid_to_idx = {pid: idx for idx, pid in enumerate(all_pids)}

    # Step 4: Select queries with sufficient GT coverage in sample
    print(f"\n=== Step 4: Selecting queries (min {args.min_gt_per_query} GT in sample) ===")
    all_query_texts = load_queries(covid_dir / "queries.jsonl", set(qrels.keys()))

    candidate_qids = []
    for qid in sorted(qrels.keys()):
        if qid not in all_query_texts:
            continue
        gt_in_sample = sum(1 for pid in qrels[qid] if pid in sampled_pids)
        if gt_in_sample >= args.min_gt_per_query:
            candidate_qids.append((qid, gt_in_sample, len(qrels[qid])))

    # Sort by qid, take up to max_queries
    candidate_qids.sort(key=lambda x: x[0])
    selected_qids = [qid for qid, _, _ in candidate_qids[:args.max_queries]]
    queries = {qid: all_query_texts[qid] for qid in selected_qids}

    gt_doc_ids = set()
    for qid in selected_qids:
        gt_doc_ids.update(pid for pid in qrels[qid] if pid in sampled_pids)

    print(f"  {len(candidate_qids)} queries have >= {args.min_gt_per_query} GT, selected {len(selected_qids)}")
    for qid, gt_in, gt_total in candidate_qids[:args.max_queries]:
        print(f"    Query {qid}: {gt_in}/{gt_total} GT in sample ({gt_in/gt_total*100:.1f}%)")
    print(f"  GT docs in sample: {len(gt_doc_ids)}, "
          f"GT ratio: {len(gt_doc_ids)/len(docs)*100:.1f}%")

    # Step 5: Load ColBERTv2
    print(f"\n=== Step 5: Loading ColBERTv2 ===")
    encoder = ColBERTv2Encoder(args.model)

    # Step 6: Encode documents
    print(f"\n=== Step 6: Encoding {len(all_pids)} documents ===")
    doc_texts = [docs[pid] for pid in all_pids]
    doc_embeddings = encoder.encode_docs(doc_texts, args.batch_size, args.doc_max_length)

    doc_path = output_dir / "trec_covid_gt_docs.jsonl"
    doc_vec_counts = []

    with open(doc_path, 'w') as f:
        for i, (pid, embs) in enumerate(zip(all_pids, doc_embeddings)):
            chunks = [{"pos": k, "emb": [round(float(x), 6) for x in vec]}
                      for k, vec in enumerate(embs)]
            entry = {
                "pid": i,
                "original_pid": pid,
                "text": docs[pid][:500],
                "chunks": chunks
            }
            f.write(json.dumps(entry) + '\n')
            doc_vec_counts.append(len(chunks))

    del doc_texts, doc_embeddings
    print(f"Saved {len(doc_vec_counts)} docs to {doc_path}")
    print(f"  File size: {os.path.getsize(doc_path) / 1e6:.1f} MB")

    # Step 7: Encode queries
    print(f"\n=== Step 7: Encoding {len(selected_qids)} queries ===")
    query_texts = [queries[qid] for qid in selected_qids]
    query_embeddings = encoder.encode_queries(query_texts, args.batch_size, args.query_max_length)

    query_path = output_dir / "trec_covid_gt_queries.jsonl"
    query_vec_counts = []
    gt_counts = []

    with open(query_path, 'w') as f:
        for i, (qid, embs) in enumerate(zip(selected_qids, query_embeddings)):
            chunks = [{"pos": k, "emb": [round(float(x), 6) for x in vec]}
                      for k, vec in enumerate(embs)]
            gt_new_indices = [pid_to_idx[pid] for pid in qrels[qid] if pid in pid_to_idx]
            # Build graded relevance dict: {str(new_idx): grade}
            gt_rels_dict = {}
            for pid in qrels[qid]:
                if pid in pid_to_idx and qid in qrel_grades and pid in qrel_grades[qid]:
                    gt_rels_dict[str(pid_to_idx[pid])] = qrel_grades[qid][pid]
            entry = {
                "pid": i,
                "original_qid": qid,
                "text": queries[qid][:200],
                "chunks": chunks,
                "gt_pids": gt_new_indices,
                "gt_rels": gt_rels_dict
            }
            f.write(json.dumps(entry) + '\n')
            query_vec_counts.append(len(chunks))
            gt_counts.append(len(gt_new_indices))

    print(f"Saved {len(query_vec_counts)} queries to {query_path}")

    # Statistics
    print(f"\n=== Statistics ===")
    print(f"Model: ColBERTv2 (dim=128)")
    print(f"Documents: {len(doc_vec_counts)} (randomly sampled from 171K)")
    print(f"  GT docs in sample: {len(gt_doc_ids)} ({len(gt_doc_ids)/len(doc_vec_counts)*100:.1f}%)")
    print(f"  Non-GT docs: {len(doc_vec_counts) - len(gt_doc_ids)} ({(len(doc_vec_counts) - len(gt_doc_ids))/len(doc_vec_counts)*100:.1f}%)")
    print(f"  Vectors/doc: min={min(doc_vec_counts)}, max={max(doc_vec_counts)}, "
          f"avg={np.mean(doc_vec_counts):.1f}")
    print(f"  P50={np.percentile(doc_vec_counts, 50):.0f}, "
          f"P90={np.percentile(doc_vec_counts, 90):.0f}")
    print(f"Queries: {len(query_vec_counts)}")
    print(f"  Vectors/query: min={min(query_vec_counts)}, max={max(query_vec_counts)}, "
          f"avg={np.mean(query_vec_counts):.1f}")
    print(f"  GT/query: min={min(gt_counts)}, max={max(gt_counts)}, "
          f"avg={np.mean(gt_counts):.1f}")

    no_gt = sum(1 for c in gt_counts if c == 0)
    if no_gt > 0:
        print(f"  WARNING: {no_gt} queries have no GT in final dataset!")

    print(f"\n=== Done! ===")
    print(f"Documents: {doc_path}")
    print(f"Queries: {query_path}")
    print(f"\nNext: python scripts/prepare_trec_covid_e5_dense.py --input-dir {output_dir} --output-dir {output_dir}")


if __name__ == "__main__":
    main()
