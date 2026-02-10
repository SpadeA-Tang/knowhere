#!/usr/bin/env python3
"""
Prepare BioASQ dataset with ColBERTv2 multi-vector embeddings.

This script:
1. Downloads BioASQ dataset from BEIR
2. Selects up to 1000 queries with GT annotations
3. Collects all GT documents + random distractors to reach 10000 docs
4. Encodes with ColBERTv2 (128-dim token-level embeddings)
5. Saves JSONL files compatible with knowhere multi-vector tests

Usage:
    # Auto-download BioASQ from BEIR (large download ~5GB)
    python scripts/prepare_bioasq_colbertv2.py --output-dir build

    # Use pre-downloaded BioASQ in BEIR format
    python scripts/prepare_bioasq_colbertv2.py --data-dir /path/to/bioasq --output-dir build

    # Use local ColBERTv2 model
    python scripts/prepare_bioasq_colbertv2.py --output-dir build --model /path/to/colbertv2

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

BIOASQ_BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/bioasq.zip"


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

        # ColBERTv2 projects BERT's 768-dim to 128-dim
        self.linear = nn.Linear(self.bert.config.hidden_size, 128, bias=False).to(self.device)
        self._load_linear_weights(model_path)

        # Special token IDs for ColBERT
        self.q_marker_id = self.tokenizer.convert_tokens_to_ids("[unused0]")
        self.d_marker_id = self.tokenizer.convert_tokens_to_ids("[unused1]")
        self.mask_id = self.tokenizer.mask_token_id

        print(f"ColBERTv2 loaded: hidden={self.bert.config.hidden_size}, projection=128")

    def _load_linear_weights(self, model_path):
        """Load the linear projection layer weights from the ColBERTv2 checkpoint."""
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
            print("  The model may not produce correct embeddings.")

    def _insert_marker(self, input_ids, attention_mask, marker_id):
        """Insert [Q] or [D] marker token after [CLS]."""
        bs, sl = input_ids.shape
        new_ids = torch.zeros(bs, sl + 1, dtype=input_ids.dtype, device=self.device)
        new_mask = torch.zeros(bs, sl + 1, dtype=attention_mask.dtype, device=self.device)

        new_ids[:, 0] = input_ids[:, 0]       # [CLS]
        new_ids[:, 1] = marker_id             # [Q] or [D]
        new_ids[:, 2:] = input_ids[:, 1:]     # rest of tokens
        new_mask[:, 0] = 1
        new_mask[:, 1] = 1
        new_mask[:, 2:] = attention_mask[:, 1:]

        return new_ids, new_mask

    @torch.no_grad()
    def encode_docs(self, texts, batch_size=32, max_length=512):
        """Encode documents. Returns list of numpy arrays [num_tokens, 128]."""
        all_embeddings = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding docs"):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=max_length - 1  # reserve 1 for [D]
            ).to(self.device)

            input_ids, attention_mask = self._insert_marker(
                inputs['input_ids'], inputs['attention_mask'], self.d_marker_id
            )

            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            token_embs = self.linear(outputs.last_hidden_state)
            token_embs = F.normalize(token_embs, p=2, dim=-1)

            mask = attention_mask.bool()
            for j in range(len(batch)):
                valid_embs = token_embs[j][mask[j]].cpu().numpy()
                # Filter near-zero vectors
                norms = np.linalg.norm(valid_embs, axis=-1)
                valid_embs = valid_embs[norms > 1e-6]
                if len(valid_embs) == 0:
                    valid_embs = token_embs[j][0:1].cpu().numpy()
                all_embeddings.append(valid_embs)

        return all_embeddings

    @torch.no_grad()
    def encode_queries(self, texts, batch_size=32, max_length=64):
        """Encode queries with [Q] marker and [MASK] padding. Returns list of numpy arrays."""
        all_embeddings = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding queries"):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=max_length - 1  # reserve 1 for [Q]
            ).to(self.device)

            input_ids, attention_mask = self._insert_marker(
                inputs['input_ids'], inputs['attention_mask'], self.q_marker_id
            )

            # Pad queries with [MASK] tokens to max_length (ColBERTv2 query augmentation)
            bs, cur_len = input_ids.shape
            if cur_len < max_length:
                pad_len = max_length - cur_len
                # Find [SEP] positions and insert [MASK] before them
                mask_pad = torch.full((bs, pad_len), self.mask_id,
                                     dtype=input_ids.dtype, device=self.device)
                mask_attn = torch.ones(bs, pad_len, dtype=attention_mask.dtype, device=self.device)
                input_ids = torch.cat([input_ids, mask_pad], dim=1)
                attention_mask = torch.cat([attention_mask, mask_attn], dim=1)

            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            token_embs = self.linear(outputs.last_hidden_state)
            token_embs = F.normalize(token_embs, p=2, dim=-1)

            mask = attention_mask.bool()
            for j in range(len(batch)):
                valid_embs = token_embs[j][mask[j]].cpu().numpy()
                norms = np.linalg.norm(valid_embs, axis=-1)
                valid_embs = valid_embs[norms > 1e-6]
                if len(valid_embs) == 0:
                    valid_embs = token_embs[j][0:1].cpu().numpy()
                all_embeddings.append(valid_embs)

        return all_embeddings


# ============================================================================
# BioASQ Data Loading (BEIR format)
# ============================================================================

def download_bioasq(data_dir: Path) -> Path:
    """Download BioASQ dataset from BEIR if not exists."""
    possible_dirs = [
        data_dir / "bioasq",
        data_dir,
        Path("/tmp/bioasq"),
    ]
    for d in possible_dirs:
        if d.exists() and (d / "corpus.jsonl").exists():
            print(f"Using cached BioASQ: {d}")
            return d

    bioasq_dir = data_dir / "bioasq"
    zip_path = data_dir / "bioasq.zip"

    if not zip_path.exists():
        print(f"Downloading BioASQ from BEIR...")
        print(f"  URL: {BIOASQ_BEIR_URL}")
        print(f"  WARNING: This is a large download (~5 GB). Press Ctrl+C to cancel.")
        print(f"  Saving to: {zip_path}")
        urllib.request.urlretrieve(BIOASQ_BEIR_URL, zip_path)
        print(f"  Download complete.")

    print(f"Extracting to {data_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(data_dir)

    return bioasq_dir


def load_qrels(qrels_dir: Path) -> Dict[str, List[str]]:
    """Load query relevance judgments from BEIR qrels directory."""
    qrels = {}

    # Try test.tsv first, then train.tsv
    for split in ["test.tsv", "train.tsv", "dev.tsv"]:
        qrels_path = qrels_dir / split
        if not qrels_path.exists():
            continue

        print(f"Loading qrels from {qrels_path}...")
        with open(qrels_path, 'r', encoding='utf-8') as f:
            header = next(f)  # skip header
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    qid, did = parts[0], parts[1]
                    rel = int(parts[2]) if len(parts) > 2 else 1
                    if rel > 0:
                        qrels.setdefault(qid, []).append(did)

        print(f"  Loaded {len(qrels)} queries with GT from {split}")

    return qrels


def load_queries(queries_path: Path, needed_qids: Set[str]) -> Dict[str, str]:
    """Load query texts for specific query IDs."""
    queries = {}
    print(f"Loading queries from {queries_path}...")

    with open(queries_path, 'r', encoding='utf-8') as f:
        for line in f:
            q = json.loads(line)
            qid = q['_id']
            if qid in needed_qids:
                queries[qid] = q['text']

    print(f"  Found {len(queries)}/{len(needed_qids)} query texts")
    return queries


def select_documents(corpus_path: Path, gt_doc_ids: Set[str],
                     max_docs: int) -> Tuple[Dict[str, str], List[str]]:
    """Stream through corpus, keeping GT docs + reservoir-sampled negatives.

    Uses reservoir sampling to handle arbitrarily large corpora with O(max_docs) memory.
    """
    docs = {}
    all_pids = []
    remaining_slots = max(0, max_docs - len(gt_doc_ids))

    # Reservoir for non-GT docs
    reservoir = []  # list of (pid, text)
    total_non_gt = 0

    print(f"Streaming through corpus...")
    print(f"  Need: {len(gt_doc_ids)} GT docs + up to {remaining_slots} negatives")

    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            if (line_num + 1) % 1000000 == 0:
                print(f"  Scanned {(line_num + 1) // 1000000}M docs, "
                      f"found {len(docs)} GT docs...")

            doc = json.loads(line)
            pid = doc['_id']
            title = doc.get('title', '')
            text = doc.get('text', '')
            full_text = f"{title} {text}".strip() if title else text

            if pid in gt_doc_ids:
                docs[pid] = full_text
                all_pids.append(pid)
            else:
                # Reservoir sampling for non-GT docs
                total_non_gt += 1
                if len(reservoir) < remaining_slots:
                    reservoir.append((pid, full_text))
                else:
                    j = random.randint(0, total_non_gt - 1)
                    if j < remaining_slots:
                        reservoir[j] = (pid, full_text)

    # Add reservoir-sampled negatives
    for pid, text in reservoir:
        docs[pid] = text
        all_pids.append(pid)

    print(f"  Total: {len(docs)} docs ({len(docs) - len(reservoir)} GT + {len(reservoir)} negatives)")
    print(f"  Scanned {total_non_gt + len(gt_doc_ids)} total docs in corpus")

    return docs, all_pids


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Prepare BioASQ dataset with ColBERTv2 multi-vector embeddings"
    )
    parser.add_argument("--data-dir", type=str, default=None,
                        help="BioASQ data directory (BEIR format). Auto-downloads if not provided.")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Output directory for JSONL files")
    parser.add_argument("--model", type=str, default="colbert-ir/colbertv2.0",
                        help="ColBERTv2 model path (HuggingFace ID or local path)")
    parser.add_argument("--max-queries", type=int, default=1000,
                        help="Maximum queries to select (default: 1000)")
    parser.add_argument("--max-docs", type=int, default=10000,
                        help="Maximum documents to select (default: 10000)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Encoding batch size (default: 32)")
    parser.add_argument("--doc-max-length", type=int, default=512,
                        help="Max token length for documents (default: 512)")
    parser.add_argument("--query-max-length", type=int, default=64,
                        help="Max token length for queries (default: 64)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Locate/download BioASQ data
    print("=== Step 1: Loading BioASQ dataset ===")
    if args.data_dir:
        bioasq_dir = Path(args.data_dir)
        if not (bioasq_dir / "corpus.jsonl").exists():
            print(f"ERROR: corpus.jsonl not found in {bioasq_dir}")
            print("Expected BEIR format: corpus.jsonl, queries.jsonl, qrels/test.tsv")
            return
    else:
        bioasq_dir = download_bioasq(output_dir)

    # Step 2: Load qrels
    print(f"\n=== Step 2: Loading qrels ===")
    qrels = load_qrels(bioasq_dir / "qrels")

    if not qrels:
        print("ERROR: No qrels found!")
        return

    # Step 3: Select queries
    print(f"\n=== Step 3: Selecting queries ===")
    sorted_qids = sorted(qrels.keys())
    selected_qids = sorted_qids[:args.max_queries]

    if len(selected_qids) < args.max_queries:
        print(f"WARNING: Only {len(selected_qids)} queries available "
              f"(requested {args.max_queries})")

    # Collect GT doc IDs
    gt_doc_ids = set()
    for qid in selected_qids:
        gt_doc_ids.update(qrels[qid])

    print(f"Selected {len(selected_qids)} queries")
    print(f"Total GT documents: {len(gt_doc_ids)}")

    # Step 4: Load query texts
    print(f"\n=== Step 4: Loading query texts ===")
    queries = load_queries(bioasq_dir / "queries.jsonl", set(selected_qids))

    # Filter to queries with text
    selected_qids = [qid for qid in selected_qids if qid in queries]
    print(f"Queries with text: {len(selected_qids)}")

    # Update GT doc IDs based on final query selection
    gt_doc_ids = set()
    for qid in selected_qids:
        gt_doc_ids.update(qrels[qid])
    print(f"GT documents for selected queries: {len(gt_doc_ids)}")

    # Step 5: Select documents
    print(f"\n=== Step 5: Selecting documents ===")
    docs, all_pids = select_documents(
        bioasq_dir / "corpus.jsonl", gt_doc_ids, args.max_docs
    )

    # Create PID to index mapping
    pid_to_idx = {pid: idx for idx, pid in enumerate(all_pids)}

    # Step 6: Load ColBERTv2
    print(f"\n=== Step 6: Loading ColBERTv2 ===")
    encoder = ColBERTv2Encoder(args.model)

    # Step 7: Encode documents
    print(f"\n=== Step 7: Encoding {len(all_pids)} documents ===")
    doc_texts = [docs[pid] for pid in all_pids]
    doc_embeddings = encoder.encode_docs(doc_texts, args.batch_size, args.doc_max_length)

    doc_path = output_dir / "bioasq_gt_docs.jsonl"
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

    # Step 8: Encode queries
    print(f"\n=== Step 8: Encoding {len(selected_qids)} queries ===")
    query_texts = [queries[qid] for qid in selected_qids]
    query_embeddings = encoder.encode_queries(query_texts, args.batch_size, args.query_max_length)

    query_path = output_dir / "bioasq_gt_queries.jsonl"
    query_vec_counts = []
    gt_counts = []

    with open(query_path, 'w') as f:
        for i, (qid, embs) in enumerate(zip(selected_qids, query_embeddings)):
            chunks = [{"pos": k, "emb": [round(float(x), 6) for x in vec]}
                      for k, vec in enumerate(embs)]
            gt_new_indices = [pid_to_idx[pid] for pid in qrels[qid] if pid in pid_to_idx]

            entry = {
                "pid": i,
                "original_qid": qid,
                "text": queries[qid][:200],
                "chunks": chunks,
                "gt_pids": gt_new_indices
            }
            f.write(json.dumps(entry) + '\n')
            query_vec_counts.append(len(chunks))
            gt_counts.append(len(gt_new_indices))

    print(f"Saved {len(query_vec_counts)} queries to {query_path}")

    # Statistics
    print(f"\n=== Statistics ===")
    print(f"Model: ColBERTv2 (dim=128)")
    print(f"Documents: {len(doc_vec_counts)}")
    print(f"  GT docs: {len(gt_doc_ids)}, Negatives: {len(doc_vec_counts) - len(gt_doc_ids)}")
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
    print(f"\nNext steps:")
    print(f"  1. Prepare E5 dense embeddings:")
    print(f"     python scripts/prepare_bioasq_e5_dense.py --input-dir {output_dir} --output-dir {output_dir}")
    print(f"  2. Run multi-vector test:")
    print(f"     ./tests/ut/knowhere_tests '[bioasq_emb_list]'")


if __name__ == "__main__":
    main()
