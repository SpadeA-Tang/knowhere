#!/usr/bin/env python3
"""
Prepare MS MARCO ColBERT multi-vector data with hybrid document sampling.

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
    pip install torch transformers safetensors huggingface_hub datasets tqdm numpy
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
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

QRELS_URL = "https://msmarco.z22.web.core.windows.net/msmarcoranking/qrels.dev.small.tsv"


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

        # Auto-detect text prompts and settings from config_sentence_transformers.json (pylate format)
        self.query_prefix = ""
        self.doc_prefix = ""
        self._attend_to_expansion = None
        self._do_query_expansion = True
        self._st_marker_q = None
        self._st_marker_d = None
        st_config_path = os.path.join(model_path, "config_sentence_transformers.json")
        if os.path.isfile(st_config_path):
            with open(st_config_path) as f:
                st_config = json.load(f)
            prompts = st_config.get("prompts", {})
            self.query_prefix = prompts.get("query", "")
            self.doc_prefix = prompts.get("document", "")
            if "attend_to_expansion_tokens" in st_config:
                self._attend_to_expansion = st_config["attend_to_expansion_tokens"]
            self._do_query_expansion = st_config.get("do_query_expansion", True)
            # Marker token strings (e.g. "[Q] ", "[unused0]")
            self._st_marker_q = st_config.get("query_prefix", None)
            self._st_marker_d = st_config.get("document_prefix", None)

        # Auto-detect marker tokens
        # Priority: config_sentence_transformers.json > [QueryMarker] > [Q]/[Q] > [unused0]
        self.q_marker_id = None
        self.attend_to_mask_tokens = False

        # 1. From config_sentence_transformers.json (most reliable for pylate models)
        if self._st_marker_q:
            tid = self.tokenizer.convert_tokens_to_ids(self._st_marker_q)
            if tid != self.tokenizer.unk_token_id:
                self.q_marker_id = tid
                self.d_marker_id = self.tokenizer.convert_tokens_to_ids(self._st_marker_d)
                self.attend_to_mask_tokens = True

        # 2. [QueryMarker] / [DocumentMarker] (jina-colbert-v2)
        if self.q_marker_id is None:
            tid = self.tokenizer.convert_tokens_to_ids("[QueryMarker]")
            if tid != self.tokenizer.unk_token_id:
                self.q_marker_id = tid
                self.d_marker_id = self.tokenizer.convert_tokens_to_ids("[DocumentMarker]")
                self.attend_to_mask_tokens = True

        # 3. [Q] / [D] with or without trailing space (pylate without config)
        if self.q_marker_id is None:
            for q_tok, d_tok in [("[Q]", "[D]"), ("[Q] ", "[D] ")]:
                tid = self.tokenizer.convert_tokens_to_ids(q_tok)
                if tid != self.tokenizer.unk_token_id:
                    self.q_marker_id = tid
                    self.d_marker_id = self.tokenizer.convert_tokens_to_ids(d_tok)
                    self.attend_to_mask_tokens = True
                    break

        # 4. Fallback: [unused0] / [unused1] (ColBERTv2, answerai without config)
        if self.q_marker_id is None:
            self.q_marker_id = self.tokenizer.convert_tokens_to_ids("[unused0]")
            self.d_marker_id = self.tokenizer.convert_tokens_to_ids("[unused1]")
            self.attend_to_mask_tokens = False

        self.mask_id = self.tokenizer.mask_token_id

        # Override attend_to_mask from config_sentence_transformers.json if available
        if self._attend_to_expansion is not None:
            self.attend_to_mask_tokens = self._attend_to_expansion

        # Override from artifact.metadata if available
        artifact_path = os.path.join(model_path, "artifact.metadata")
        if os.path.isfile(artifact_path):
            with open(artifact_path) as f:
                metadata = json.load(f)
            if "attend_to_mask_tokens" in metadata:
                self.attend_to_mask_tokens = metadata["attend_to_mask_tokens"]

        print(f"ColBERT loaded: hidden={self.bert.config.hidden_size}, projection={self.dim}")
        print(f"  Q marker id={self.q_marker_id}, D marker id={self.d_marker_id}, mask id={self.mask_id}")
        print(f"  attend_to_mask_tokens={self.attend_to_mask_tokens}, do_query_expansion={self._do_query_expansion}")
        if self.query_prefix or self.doc_prefix:
            print(f"  query_prefix='{self.query_prefix}', doc_prefix='{self.doc_prefix}'")

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

        # Try pylate Dense subdirectories (e.g., 1_Dense/, 2_Dense/)
        if weight is None and os.path.isdir(model_path):
            try:
                from safetensors.torch import load_file
                for dense_dir in ["1_Dense", "2_Dense"]:
                    dense_sf = os.path.join(model_path, dense_dir, "model.safetensors")
                    if os.path.isfile(dense_sf):
                        state_dict = load_file(dense_sf)
                        for key, value in state_dict.items():
                            if 'weight' in key.lower():
                                weight = value
                                key_name = f"{dense_dir}/{key}"
                                break
                        if weight is not None:
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
        if self.doc_prefix:
            texts = [self.doc_prefix + t for t in texts]
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
        if self.query_prefix:
            texts = [self.query_prefix + t for t in texts]
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        pad_id = self.tokenizer.pad_token_id

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
            real_lengths = []
            for token_ids in encoded['input_ids']:
                real_seq = [cls_id, self.q_marker_id] + token_ids + [sep_id]
                attn = [1] * len(real_seq)
                real_lengths.append(len(real_seq))

                if self._do_query_expansion:
                    num_mask = max_length - len(real_seq)
                    if num_mask > 0:
                        real_seq += [self.mask_id] * num_mask
                        attn += [1 if self.attend_to_mask_tokens else 0] * num_mask
                else:
                    pad_len = max_length - len(real_seq)
                    if pad_len > 0:
                        real_seq += [pad_id] * pad_len
                        attn += [0] * pad_len
                all_ids.append(real_seq[:max_length])
                all_mask.append(attn[:max_length])

            input_ids = torch.tensor(all_ids, dtype=torch.long, device=self.device)
            attention_mask = torch.tensor(all_mask, dtype=torch.long, device=self.device)

            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            token_embs = self.linear(outputs.last_hidden_state)
            token_embs = F.normalize(token_embs, p=2, dim=-1).float()

            for j in range(len(batch)):
                if self._do_query_expansion:
                    all_embeddings.append(token_embs[j].cpu().numpy())
                else:
                    all_embeddings.append(token_embs[j][:real_lengths[j]].cpu().numpy())
        return all_embeddings


# ============================================================================
# MS MARCO Data Loading
# ============================================================================

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


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Prepare MS MARCO with ColBERT multi-vector embeddings (hybrid sampling)"
    )
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--model", type=str, default="~/models/jina-colbert-v2",
                        help="ColBERT model path or HuggingFace ID")
    parser.add_argument("--max-queries", type=int, default=100)
    parser.add_argument("--max-docs", type=int, default=5000)
    parser.add_argument("--collection", type=str, default=None,
                        help="Path to local collection.tsv")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--doc-max-length", type=int, default=512)
    parser.add_argument("--query-max-length", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-tag", type=str, default="gt",
                        help="Model tag for output filenames (e.g., colbertv2, colbertzero). "
                             "Output: msmarco_{model_tag}_docs.jsonl")

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

    # Step 4: Load ColBERT
    print(f"\n=== Step 4: Loading ColBERT ===")
    encoder = ColBERTEncoder(args.model)

    # Step 5: Encode documents
    print(f"\n=== Step 5: Encoding {len(all_pids)} documents ===")
    doc_texts = [docs[pid] for pid in all_pids]
    doc_embeddings = encoder.encode_docs(doc_texts, args.batch_size, args.doc_max_length)

    doc_path = output_dir / f"msmarco_{args.model_tag}_docs.jsonl"
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
    query_list = [query_texts[qid] for qid in selected_qids]
    query_embeddings = encoder.encode_queries(query_list, args.batch_size, args.query_max_length)

    query_path = output_dir / f"msmarco_{args.model_tag}_queries.jsonl"
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
                "text": query_texts[qid][:200],
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
    print(f"Docs: {len(doc_vec_counts)} (hybrid: {len(gt_in_sample)} GT + {len(doc_vec_counts) - len(gt_in_sample)} random)")
    print(f"  GT ratio: {len(gt_in_sample)/len(doc_vec_counts)*100:.1f}%")
    print(f"  Vectors/doc: min={min(doc_vec_counts)}, max={max(doc_vec_counts)}, avg={np.mean(doc_vec_counts):.1f}")
    print(f"  P50={np.percentile(doc_vec_counts, 50):.0f}, "
          f"P90={np.percentile(doc_vec_counts, 90):.0f}")

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
