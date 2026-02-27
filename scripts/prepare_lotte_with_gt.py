#!/usr/bin/env python3
"""
Prepare LoTTE ColBERT multi-vector data with random document sampling.

This script:
1. Randomly samples N documents from the full LoTTE corpus
2. Selects queries that have sufficient GT coverage in the random sample
3. Encodes with ColBERT (auto-detects projection dimension from model weights)

Usage:
    python scripts/prepare_lotte_with_gt.py --domain science --output-dir build
    python scripts/prepare_lotte_with_gt.py --domain science --output-dir build --max-docs 5000

Requirements:
    pip install torch transformers safetensors huggingface_hub tqdm numpy
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


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
        """Encode documents with chunking: split long docs into chunks, encode each, concat.

        Each chunk: [CLS] [D] tok1...tokN [SEP] [PAD]...
        Long documents are split into multiple chunks of (max_length - 3) tokens,
        each encoded separately, and all token vectors are concatenated.
        """
        if self.doc_prefix:
            texts = [self.doc_prefix + t for t in texts]
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        pad_id = self.tokenizer.pad_token_id
        chunk_token_len = max_length - 3  # reserve [CLS] + [D] + [SEP]

        # Step 1: tokenize all texts without truncation
        all_token_ids = self.tokenizer(
            texts, add_special_tokens=False, truncation=False,
            return_attention_mask=False,
        )['input_ids']

        # Step 2: split into chunks and build (doc_idx, chunk_tokens) pairs
        chunks = []
        for doc_idx, token_ids in enumerate(all_token_ids):
            if len(token_ids) == 0:
                token_ids = [pad_id]
            for start in range(0, len(token_ids), chunk_token_len):
                chunk = token_ids[start:start + chunk_token_len]
                chunks.append((doc_idx, chunk))

        print(f"  {len(texts)} docs -> {len(chunks)} chunks (chunk_size={max_length})")

        # Step 3: encode chunks in batches
        chunk_embeddings = [[] for _ in range(len(texts))]
        for i in tqdm(range(0, len(chunks), batch_size), desc="Encoding doc chunks"):
            batch_chunks = chunks[i:i + batch_size]

            all_ids = []
            all_mask = []
            for _, token_ids in batch_chunks:
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

            for j, (doc_idx, _) in enumerate(batch_chunks):
                valid_embs = token_embs[j][attention_mask[j].bool()].cpu().numpy()
                chunk_embeddings[doc_idx].append(valid_embs)

        # Step 4: concat all chunks per document
        all_embeddings = []
        for doc_idx in range(len(texts)):
            doc_embs = np.concatenate(chunk_embeddings[doc_idx], axis=0)
            all_embeddings.append(doc_embs)
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
# LoTTE Data Loading
# ============================================================================

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


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Prepare LoTTE with ColBERT multi-vector embeddings (random sampling)"
    )
    parser.add_argument("--domain", type=str, default="lifestyle")
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--model", type=str, default="~/models/jina-colbert-v2",
                        help="ColBERT model path or HuggingFace ID")
    parser.add_argument("--max-queries", type=int, default=100)
    parser.add_argument("--max-docs", type=int, default=5000)
    parser.add_argument("--min-gt-per-query", type=int, default=1,
                        help="Min GT docs in sample for a query to be included (default: 1)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--doc-max-length", type=int, default=512)
    parser.add_argument("--max-doc-tokens", type=int, default=0,
                        help="Max token embeddings per document (0=no truncation)")
    parser.add_argument("--query-max-length", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-tag", type=str, default="gt",
                        help="Model tag for output filenames (e.g., colbertv2, colbertzero). "
                             "Output: lotte_{domain}_{model_tag}_docs.jsonl")

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    lotte_dir = output_dir / "lotte" / args.domain / "dev"

    collection_path = lotte_dir / "collection.tsv"
    qas_path = lotte_dir / "qas.forum.jsonl"

    # Check files exist
    for p in [collection_path, qas_path]:
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

    # Step 4: Load ColBERT
    print(f"\n=== Step 4: Loading ColBERT ===")
    encoder = ColBERTEncoder(args.model)

    # Step 5: Encode documents (batch)
    print(f"\n=== Step 5: Encoding {len(docs)} documents ===")
    doc_texts = [docs[pid] for pid in all_pids]
    doc_embeddings = encoder.encode_docs(doc_texts, args.batch_size, args.doc_max_length)

    doc_path = output_dir / f"lotte_{args.domain}_{args.model_tag}_docs.jsonl"
    doc_vec_counts = []

    with open(doc_path, 'w') as f:
        for i, (pid, embs) in enumerate(zip(all_pids, doc_embeddings)):
            # Truncate to max_doc_tokens if set
            if args.max_doc_tokens > 0 and len(embs) > args.max_doc_tokens:
                embs = embs[:args.max_doc_tokens]
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

    # Step 6: Encode queries (batch)
    print(f"\n=== Step 6: Encoding {len(selected_qas)} queries ===")
    query_texts = [qa['query'] for qa in selected_qas]
    query_embeddings = encoder.encode_queries(query_texts, args.batch_size, args.query_max_length)

    query_path = output_dir / f"lotte_{args.domain}_{args.model_tag}_queries.jsonl"
    gt_counts = []

    with open(query_path, 'w') as f:
        for i, (qa, embs) in enumerate(zip(selected_qas, query_embeddings)):
            chunks = [{"pos": k, "emb": [round(float(x), 6) for x in vec]}
                      for k, vec in enumerate(embs)]

            # Remap GT pids to new indices
            gt_new_indices = [pid_to_idx[pid] for pid in qa['answer_pids'] if pid in pid_to_idx]
            # Build graded relevance dict (binary for LoTTE)
            gt_rels_dict = {str(idx): 1 for idx in gt_new_indices}

            entry = {
                "pid": i,
                "original_qid": qa['qid'],
                "text": qa['query'][:200],
                "chunks": chunks,
                "gt_pids": gt_new_indices,
                "gt_rels": gt_rels_dict
            }
            f.write(json.dumps(entry) + '\n')
            gt_counts.append(len(gt_new_indices))

    print(f"Saved {len(gt_counts)} queries to {query_path}")

    # Statistics
    print(f"\n=== Statistics ===")
    print(f"Model: ColBERT (dim={encoder.dim})")
    print(f"Docs: {len(doc_vec_counts)} (randomly sampled)")
    print(f"  GT docs in sample: {len(gt_doc_ids)} ({len(gt_doc_ids)/len(doc_vec_counts)*100:.1f}%)")
    print(f"  Non-GT docs: {len(doc_vec_counts) - len(gt_doc_ids)} ({(len(doc_vec_counts) - len(gt_doc_ids))/len(doc_vec_counts)*100:.1f}%)")
    print(f"  Vectors/doc: min={min(doc_vec_counts)}, max={max(doc_vec_counts)}, avg={np.mean(doc_vec_counts):.1f}")
    print(f"  P50={np.percentile(doc_vec_counts, 50):.0f}, "
          f"P90={np.percentile(doc_vec_counts, 90):.0f}")

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
