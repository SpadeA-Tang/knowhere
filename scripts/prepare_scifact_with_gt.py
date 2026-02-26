#!/usr/bin/env python3
"""
Prepare SciFact ColBERT multi-vector data with official ground truth annotations.

This script:
1. Downloads SciFact dataset from BEIR
2. Uses all 5183 documents (scientific abstracts)
3. Selects queries with GT annotations
4. Encodes with ColBERT (auto-detects projection dimension from model weights)

Usage:
    python scripts/prepare_scifact_with_gt.py --output-dir build
    python scripts/prepare_scifact_with_gt.py --output-dir build --model /path/to/colbertv2

Requirements:
    pip install torch transformers safetensors huggingface_hub tqdm numpy
"""

import argparse
import json
import os
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

SCIFACT_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"


# ============================================================================
# ColBERT Encoder (same as prepare_trec_covid_colbertv2.py)
# ============================================================================

class ColBERTEncoder:
    """ColBERT encoder using raw transformers (no colbert-ai dependency).

    Supports any ColBERT model (ColBERTv2, answerai-colbert-small-v1, jina-colbert-v2, etc.).
    Auto-detects projection dimension and special tokens from model weights.
    """

    def __init__(self, model_path="~/models/jina-colbert-v2"):
        from transformers import AutoTokenizer, AutoModel

        model_path = os.path.expanduser(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading ColBERT from {model_path} (device={self.device})...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.bert = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(self.device).eval()

        # Load linear projection (auto-detect dim from weights)
        self.dim, self.linear = self._load_linear_weights(model_path)

        # Auto-detect marker tokens (supports BERT and XLM-RoBERTa based models)
        q_marker = self.tokenizer.convert_tokens_to_ids("[QueryMarker]")
        if q_marker == self.tokenizer.unk_token_id:
            # BERT-based model (ColBERTv2, answerai-colbert-small-v1)
            self.q_marker_id = self.tokenizer.convert_tokens_to_ids("[unused0]")
            self.d_marker_id = self.tokenizer.convert_tokens_to_ids("[unused1]")
            self.attend_to_mask_tokens = False
        else:
            # jina-colbert-v2 style
            self.q_marker_id = q_marker
            self.d_marker_id = self.tokenizer.convert_tokens_to_ids("[DocumentMarker]")
            self.attend_to_mask_tokens = True
        self.mask_id = self.tokenizer.mask_token_id

        # Override from artifact.metadata if available
        artifact_path = os.path.join(model_path, "artifact.metadata")
        if os.path.isfile(artifact_path):
            with open(artifact_path) as f:
                metadata = json.load(f)
            if "attend_to_mask_tokens" in metadata:
                self.attend_to_mask_tokens = metadata["attend_to_mask_tokens"]

        print(f"ColBERT loaded: hidden={self.bert.config.hidden_size}, projection={self.dim}")
        print(f"  Q marker id={self.q_marker_id}, D marker id={self.d_marker_id}, mask id={self.mask_id}")
        print(f"  attend_to_mask_tokens={self.attend_to_mask_tokens}")

    def _load_linear_weights(self, model_path):
        """Load linear projection weights and auto-detect projection dimension."""
        weight = None
        key_name = None

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
                    weight = value
                    key_name = key
                    break
        except Exception:
            pass

        # Try pytorch_model.bin
        if weight is None:
            try:
                if os.path.isdir(model_path):
                    pt_path = os.path.join(model_path, "pytorch_model.bin")
                else:
                    from huggingface_hub import hf_hub_download
                    pt_path = hf_hub_download(model_path, "pytorch_model.bin")
                state_dict = torch.load(pt_path, map_location="cpu", weights_only=True)
                for key, value in state_dict.items():
                    if 'linear' in key.lower() and 'weight' in key.lower():
                        weight = value
                        key_name = key
                        break
            except Exception:
                pass

        if weight is not None:
            proj_dim = weight.shape[0]
            linear = nn.Linear(self.bert.config.hidden_size, proj_dim, bias=False).to(self.device)
            linear.weight.data = weight.to(self.device)
            print(f"  Loaded linear projection from key: {key_name} (dim={proj_dim})")
            return proj_dim, linear
        else:
            print("  WARNING: Could not load linear projection weights! Using 128-dim default.")
            proj_dim = 128
            linear = nn.Linear(self.bert.config.hidden_size, proj_dim, bias=False).to(self.device)
            return proj_dim, linear

    @torch.no_grad()
    def encode_docs(self, texts, batch_size=32, max_length=512):
        """Encode documents: [CLS] [D] tok1...tokN [SEP] [PAD]..."""
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        pad_id = self.tokenizer.pad_token_id

        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding docs"):
            batch = texts[i:i + batch_size]

            encoded = self.tokenizer(
                batch, add_special_tokens=False, truncation=True,
                max_length=max_length - 3,
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
            token_embs = F.normalize(token_embs, p=2, dim=-1).float()

            for j in range(len(batch)):
                valid_embs = token_embs[j][attention_mask[j].bool()].cpu().numpy()
                all_embeddings.append(valid_embs)
        return all_embeddings

    @torch.no_grad()
    def encode_queries(self, texts, batch_size=32, max_length=32):
        """Encode queries: [CLS] [Q] tok1...tokN [SEP] [MASK]...[MASK]"""
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id

        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding queries"):
            batch = texts[i:i + batch_size]

            encoded = self.tokenizer(
                batch, add_special_tokens=False, truncation=True,
                max_length=max_length - 3,
                return_attention_mask=False,
            )

            all_ids = []
            all_mask = []
            for token_ids in encoded['input_ids']:
                real_seq = [cls_id, self.q_marker_id] + token_ids + [sep_id]
                attn = [1] * len(real_seq)
                num_mask = max_length - len(real_seq)
                if num_mask > 0:
                    real_seq += [self.mask_id] * num_mask
                    attn += [1 if self.attend_to_mask_tokens else 0] * num_mask
                all_ids.append(real_seq[:max_length])
                all_mask.append(attn[:max_length])

            input_ids = torch.tensor(all_ids, dtype=torch.long, device=self.device)
            attention_mask = torch.tensor(all_mask, dtype=torch.long, device=self.device)

            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            token_embs = self.linear(outputs.last_hidden_state)
            token_embs = F.normalize(token_embs, p=2, dim=-1).float()

            for j in range(len(batch)):
                all_embeddings.append(token_embs[j].cpu().numpy())
        return all_embeddings


# ============================================================================
# SciFact Data Loading
# ============================================================================

def download_scifact(output_dir: Path) -> Path:
    """Download SciFact dataset if not exists."""
    possible_dirs = [
        output_dir / "scifact",
        Path("/tmp/scifact"),
    ]

    for scifact_dir in possible_dirs:
        if scifact_dir.exists() and (scifact_dir / "corpus.jsonl").exists():
            print(f"Using cached SciFact: {scifact_dir}")
            return scifact_dir

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
            title = doc.get('title', '')
            text = doc.get('text', '')
            full_text = f"{title} {text}".strip() if title else text

            docs[pid] = full_text
            all_pids.append(pid)

    print(f"Loaded {len(docs)} documents")
    return docs, all_pids


def load_queries_and_qrels(scifact_dir: Path, max_queries: int
                           ) -> Tuple[Dict[str, str], Dict[str, List[str]], Dict[str, Dict[str, int]]]:
    """Load queries and their ground truth annotations with graded relevance.

    Returns (queries, qrels, qrel_grades).
    qrels[qid] = [did1, did2, ...]  (docs with rel > 0)
    qrel_grades[qid][did] = relevance grade (1 or 2 for SciFact)
    """
    queries_path = scifact_dir / "queries.jsonl"
    qrels_path = scifact_dir / "qrels" / "test.tsv"

    print(f"Loading qrels from {qrels_path}...")
    qrels = {}
    qrel_grades = {}
    with open(qrels_path, 'r', encoding='utf-8') as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                qid, did, rel = parts[0], parts[1], int(parts[2])
                if rel > 0:
                    if qid not in qrels:
                        qrels[qid] = []
                        qrel_grades[qid] = {}
                    qrels[qid].append(did)
                    qrel_grades[qid][did] = rel

    print(f"Found {len(qrels)} queries with GT annotations")

    from collections import Counter
    all_rels = [r for grades in qrel_grades.values() for r in grades.values()]
    if all_rels:
        print(f"  Relevance grades: {dict(sorted(Counter(all_rels).items()))}")

    print(f"Loading queries from {queries_path}...")
    all_queries = {}
    with open(queries_path, 'r', encoding='utf-8') as f:
        for line in f:
            q = json.loads(line)
            all_queries[q['_id']] = q['text']

    selected_qids = [qid for qid in qrels.keys() if qid in all_queries][:max_queries]
    queries = {qid: all_queries[qid] for qid in selected_qids}
    selected_qrels = {qid: qrels[qid] for qid in selected_qids}
    selected_qrel_grades = {qid: qrel_grades[qid] for qid in selected_qids}

    print(f"Selected {len(queries)} queries with GT")
    return queries, selected_qrels, selected_qrel_grades


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Prepare SciFact with ColBERT multi-vector embeddings"
    )
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--model", type=str, default="~/models/jina-colbert-v2",
                        help="ColBERT model path or HuggingFace ID")
    parser.add_argument("--max-queries", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--doc-max-length", type=int, default=512)
    parser.add_argument("--query-max-length", type=int, default=32)

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Download SciFact
    print("=== Step 1: Downloading SciFact ===")
    scifact_dir = download_scifact(output_dir)

    # Step 2: Load corpus (all documents)
    print(f"\n=== Step 2: Loading corpus ===")
    docs, all_pids = load_corpus(scifact_dir)
    pid_to_idx = {pid: idx for idx, pid in enumerate(all_pids)}

    # Step 3: Load queries with GT
    print(f"\n=== Step 3: Loading {args.max_queries} queries with GT ===")
    queries, qrels, qrel_grades = load_queries_and_qrels(scifact_dir, args.max_queries)
    selected_qids = list(queries.keys())

    # Step 4: Load ColBERT
    print(f"\n=== Step 4: Loading ColBERT ===")
    encoder = ColBERTEncoder(args.model)

    # Step 5: Encode documents
    print(f"\n=== Step 5: Encoding {len(all_pids)} documents ===")
    doc_texts = [docs[pid] for pid in all_pids]
    doc_embeddings = encoder.encode_docs(doc_texts, args.batch_size, args.doc_max_length)

    doc_path = output_dir / "scifact_gt_docs.jsonl"
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

    # Step 6: Encode queries
    print(f"\n=== Step 6: Encoding {len(selected_qids)} queries ===")
    query_texts = [queries[qid] for qid in selected_qids]
    query_embeddings = encoder.encode_queries(query_texts, args.batch_size, args.query_max_length)

    query_path = output_dir / "scifact_gt_queries.jsonl"
    query_vec_counts = []
    gt_counts = []

    with open(query_path, 'w') as f:
        for i, (qid, embs) in enumerate(zip(selected_qids, query_embeddings)):
            chunks = [{"pos": k, "emb": [round(float(x), 6) for x in vec]}
                      for k, vec in enumerate(embs)]
            gt_new_indices = [pid_to_idx[pid] for pid in qrels[qid] if pid in pid_to_idx]
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
    print(f"Model: ColBERT (dim={encoder.dim})")
    print(f"Documents: {len(doc_vec_counts)}")
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
    print(f"\nNext: python scripts/prepare_e5_dense.py --input-dir {output_dir} --output-dir {output_dir} --dataset scifact")


if __name__ == "__main__":
    main()
