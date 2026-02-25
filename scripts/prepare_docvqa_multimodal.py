#!/usr/bin/env python3
"""
Prepare DocVQA multi-modal data for ColQwen2 vs Qwen3-VL-Embedding comparison.

This script:
1. Downloads DocVQA test subset from HuggingFace
2. Uses GT-First Sampling (ensure all GT pages are in corpus)
3. Encodes with ColQwen2 (768 patches x 128-dim) or Qwen3-VL-Embedding (1 x 2048-dim)
4. Outputs JSONL format compatible with existing test framework

Usage:
    # ColQwen2 (multi-vector)
    python scripts/prepare_docvqa_multimodal.py --output-dir build --model vidore/colqwen2-v0.1

    # Qwen3-VL-Embedding (dense)
    python scripts/prepare_docvqa_multimodal.py --output-dir build --model Qwen/Qwen3-VL-Embedding-2B

Requirements:
    pip install datasets pillow torch transformers colpali-engine qwen-vl-utils
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm


# ============================================================================
# Model Download Helper
# ============================================================================

def download_model_from_modelscope(model_id: str, cache_dir: str = "/home/spadea/models") -> str:
    """Download model from ModelScope (faster in China) and return local path."""
    try:
        from modelscope import snapshot_download
        print(f"Downloading {model_id} from ModelScope to {cache_dir}...")
        local_path = snapshot_download(model_id, cache_dir=cache_dir)
        print(f"Model downloaded to: {local_path}")
        return local_path
    except ImportError:
        print("ModelScope not installed, falling back to HuggingFace")
        return model_id
    except Exception as e:
        print(f"ModelScope download failed: {e}, falling back to HuggingFace")
        return model_id


def download_model_from_huggingface(model_id: str, cache_dir: str = "/home/spadea/models") -> str:
    """Download model from HuggingFace Hub (supports HF_ENDPOINT mirror) and return local path."""
    import os
    try:
        from huggingface_hub import snapshot_download
        local_dir = os.path.join(cache_dir, model_id.replace("/", "_"))
        print(f"Downloading {model_id} from HuggingFace to {local_dir}...")
        local_path = snapshot_download(repo_id=model_id, local_dir=local_dir)
        print(f"Model downloaded to: {local_path}")
        return local_path
    except ImportError:
        print("huggingface_hub not installed")
        return model_id
    except Exception as e:
        print(f"HuggingFace download failed: {e}")
        return model_id


# ============================================================================
# ColQwen2 Encoder (Multi-vector, 768 patches x 128 dim)
# ============================================================================

class ColQwen2Encoder:
    """ColQwen2 encoder using colpali-engine."""

    def __init__(self, model_path: str = "vidore/colqwen2-v0.1", use_modelscope: bool = True):
        from colpali_engine.models import ColQwen2, ColQwen2Processor

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading ColQwen2 from {model_path} (device={self.device})...")

        # ColQwen2 is only on HuggingFace, use HF download (supports HF_ENDPOINT mirror)
        if "/" in model_path and not model_path.startswith("/"):
            model_path = download_model_from_huggingface(model_path)

        self.model = ColQwen2.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=self.device,
        ).eval()
        self.processor = ColQwen2Processor.from_pretrained(model_path)
        self.output_dim = 128
        self.is_multi_vector = True
        print(f"ColQwen2 loaded: patches=768, dim={self.output_dim}")

    @torch.no_grad()
    def encode_pages(self, images: List[Image.Image], batch_size: int = 4) -> List[np.ndarray]:
        """Encode document pages. Returns list of [num_patches, 128] arrays."""
        all_embeddings = []

        for i in tqdm(range(0, len(images), batch_size), desc="Encoding pages"):
            batch_images = images[i:i + batch_size]
            batch = self.processor.process_images(batch_images).to(self.device)

            with torch.cuda.amp.autocast():
                embeddings = self.model(**batch)

            for emb in embeddings:
                # emb shape: [num_patches, 128]
                all_embeddings.append(emb.cpu().float().numpy())

        return all_embeddings

    @torch.no_grad()
    def encode_queries(self, queries: List[str], batch_size: int = 8) -> List[np.ndarray]:
        """Encode queries. Returns list of [num_tokens, 128] arrays."""
        all_embeddings = []

        for i in tqdm(range(0, len(queries), batch_size), desc="Encoding queries"):
            batch_texts = queries[i:i + batch_size]
            batch = self.processor.process_queries(batch_texts).to(self.device)

            with torch.cuda.amp.autocast():
                embeddings = self.model(**batch)

            for emb in embeddings:
                # emb shape: [num_tokens, 128]
                all_embeddings.append(emb.cpu().float().numpy())

        return all_embeddings


# ============================================================================
# ColQwen3 Encoder (Multi-vector, SauerkrautLM-ColQwen3-4b)
# ============================================================================

class ColQwen3Encoder:
    """ColQwen3 encoder using sauerkrautlm_colpali."""

    def __init__(self, model_path: str = "VAGOsolutions/SauerkrautLM-ColQwen3-4b-v0.1", use_modelscope: bool = False):
        # Install sauerkrautlm_colpali if needed
        try:
            from sauerkrautlm_colpali.models import ColQwen3, ColQwen3Processor
        except ImportError:
            import subprocess
            import sys
            print("Installing sauerkrautlm_colpali...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "sauerkrautlm_colpali"])
            from sauerkrautlm_colpali.models import ColQwen3, ColQwen3Processor

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading ColQwen3 from {model_path} (device={self.device})...")

        # Download from HuggingFace if needed
        if "/" in model_path and not model_path.startswith("/"):
            model_path = download_model_from_huggingface(model_path)

        # Check for flash_attn availability
        try:
            import flash_attn
            attn_impl = "flash_attention_2"
            print("   Using FlashAttention2")
        except ImportError:
            attn_impl = "sdpa"  # PyTorch native scaled dot-product attention
            print("   flash_attn not installed, using SDPA")

        self.model = ColQwen3.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl,
            device_map=self.device,
        ).eval()
        self.processor = ColQwen3Processor.from_pretrained(model_path)
        self.output_dim = 128
        self.is_multi_vector = True
        print(f"ColQwen3 loaded: dim={self.output_dim}")

    @torch.no_grad()
    def encode_pages(self, images: List[Image.Image], batch_size: int = 2) -> List[np.ndarray]:
        """Encode document pages. Returns list of [num_patches, 128] arrays."""
        all_embeddings = []

        for i in tqdm(range(0, len(images), batch_size), desc="Encoding pages"):
            batch_images = images[i:i + batch_size]
            batch = self.processor.process_images(batch_images).to(self.device)

            with torch.cuda.amp.autocast():
                embeddings = self.model(**batch)

            for emb in embeddings:
                all_embeddings.append(emb.cpu().float().numpy())

        return all_embeddings

    @torch.no_grad()
    def encode_queries(self, queries: List[str], batch_size: int = 4) -> List[np.ndarray]:
        """Encode queries. Returns list of [num_tokens, 128] arrays."""
        all_embeddings = []

        for i in tqdm(range(0, len(queries), batch_size), desc="Encoding queries"):
            batch_texts = queries[i:i + batch_size]
            batch = self.processor.process_queries(batch_texts).to(self.device)

            with torch.cuda.amp.autocast():
                embeddings = self.model(**batch)

            for emb in embeddings:
                all_embeddings.append(emb.cpu().float().numpy())

        return all_embeddings


# ============================================================================
# Qwen2-VL Embedding Encoder (Dense, extract from Qwen2-VL-2B-Instruct)
# ============================================================================

class Qwen2VLEmbeddingEncoder:
    """Extract dense embeddings from Qwen2-VL-2B-Instruct (last hidden state pooling)."""

    def __init__(self, model_path: str = "Qwen/Qwen2-VL-2B-Instruct", use_modelscope: bool = True):
        from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Qwen2-VL from {model_path} (device={self.device})...")

        # Try ModelScope first
        if use_modelscope and "/" in model_path and not model_path.startswith("/"):
            model_path = download_model_from_modelscope(model_path)

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        ).eval()
        self.processor = Qwen2VLProcessor.from_pretrained(model_path)

        # Get hidden size from config
        self.output_dim = self.model.config.hidden_size  # 1536 for 2B
        self.is_multi_vector = False
        print(f"Qwen2-VL loaded: dim={self.output_dim}")

    @torch.no_grad()
    def encode_pages(self, images: List[Image.Image], batch_size: int = 2) -> List[np.ndarray]:
        """Encode document pages. Returns list of [1, dim] arrays."""
        all_embeddings = []

        for i in tqdm(range(0, len(images), batch_size), desc="Encoding pages"):
            batch_images = images[i:i + batch_size]

            # Process each image with a simple prompt
            for img in batch_images:
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": "Describe this document."}
                    ]
                }]

                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = self.processor(
                    text=[text],
                    images=[img],
                    padding=True,
                    return_tensors="pt"
                ).to(self.device)

                # Get last hidden state
                outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
                hidden_states = outputs.hidden_states[-1]  # [batch, seq_len, hidden_size]

                # Mean pooling over sequence (excluding padding)
                attention_mask = inputs.attention_mask
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                embedding = sum_embeddings / sum_mask  # [batch, hidden_size]

                # Normalize
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1)
                all_embeddings.append(embedding[0].cpu().float().numpy().reshape(1, -1))

        return all_embeddings

    @torch.no_grad()
    def encode_queries(self, queries: List[str], batch_size: int = 4) -> List[np.ndarray]:
        """Encode queries. Returns list of [1, dim] arrays."""
        all_embeddings = []

        for i in tqdm(range(0, len(queries), batch_size), desc="Encoding queries"):
            batch_texts = queries[i:i + batch_size]

            for text in batch_texts:
                messages = [{
                    "role": "user",
                    "content": [{"type": "text", "text": text}]
                }]

                prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = self.processor(
                    text=[prompt],
                    padding=True,
                    return_tensors="pt"
                ).to(self.device)

                # Get last hidden state
                outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
                hidden_states = outputs.hidden_states[-1]

                # Mean pooling
                attention_mask = inputs.attention_mask
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                embedding = sum_embeddings / sum_mask

                # Normalize
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1)
                all_embeddings.append(embedding[0].cpu().float().numpy().reshape(1, -1))

        return all_embeddings


# ============================================================================
# Qwen3-VL-Embedding Encoder (Dense, 1 x 2048 dim)
# ============================================================================

def setup_qwen3_vl_embedding_repo(repo_dir: str = "/home/spadea/repos/Qwen3-VL-Embedding") -> str:
    """Clone and setup Qwen3-VL-Embedding repository if needed."""
    import subprocess
    import sys

    repo_path = Path(repo_dir)
    if not repo_path.exists():
        print(f"Cloning Qwen3-VL-Embedding repository to {repo_dir}...")
        subprocess.run([
            "git", "clone", "https://github.com/QwenLM/Qwen3-VL-Embedding.git", repo_dir
        ], check=True)

    # Add repo to path
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)

    return repo_dir


class Qwen3VLEmbeddingEncoder:
    """Qwen3-VL-Embedding encoder (dense single-vector) using official repo."""

    def __init__(self, model_path: str = "Qwen/Qwen3-VL-Embedding-2B", use_modelscope: bool = True):
        # Setup the official repository
        repo_dir = setup_qwen3_vl_embedding_repo()

        # Import from official repo
        from src.models.qwen3_vl_embedding import Qwen3VLEmbedder

        print(f"Loading Qwen3-VL-Embedding from {model_path}...")

        # Download model if needed
        if use_modelscope and "/" in model_path and not model_path.startswith("/"):
            model_path = download_model_from_modelscope(model_path)

        self.model = Qwen3VLEmbedder(
            model_name_or_path=model_path,
            torch_dtype=torch.bfloat16,
        )
        # Get actual hidden size from model config (Qwen3VL uses text_config)
        config = self.model.model.config
        if hasattr(config, 'hidden_size'):
            self.output_dim = config.hidden_size
        elif hasattr(config, 'text_config') and hasattr(config.text_config, 'hidden_size'):
            self.output_dim = config.text_config.hidden_size
        else:
            # Fallback: Qwen3-VL-2B has 1536 dim
            self.output_dim = 1536
        self.is_multi_vector = False
        print(f"Qwen3-VL-Embedding loaded: dim={self.output_dim}")

    @torch.no_grad()
    def encode_pages(self, images: List[Image.Image], batch_size: int = 2) -> List[np.ndarray]:
        """Encode document pages. Returns list of [1, 2048] arrays."""
        all_embeddings = []

        for i in tqdm(range(0, len(images), batch_size), desc="Encoding pages"):
            batch_images = images[i:i + batch_size]

            # Prepare inputs for Qwen3VLEmbedder - use process() method
            inputs = [{"image": img} for img in batch_images]

            # process() returns [batch_size, dim] tensor
            embeddings = self.model.process(inputs)

            for emb in embeddings:
                # emb shape: [dim] -> reshape to [1, dim]
                # Convert to float32 for numpy (bf16 not supported)
                all_embeddings.append(emb.cpu().float().numpy().reshape(1, -1))

        return all_embeddings

    @torch.no_grad()
    def encode_queries(self, queries: List[str], batch_size: int = 8) -> List[np.ndarray]:
        """Encode queries. Returns list of [1, 2048] arrays."""
        all_embeddings = []

        for i in tqdm(range(0, len(queries), batch_size), desc="Encoding queries"):
            batch_texts = queries[i:i + batch_size]

            # Prepare inputs with instruction for queries - use process() method
            inputs = [{
                "text": text,
                "instruction": "Retrieve documents relevant to this query."
            } for text in batch_texts]

            # process() returns [batch_size, dim] tensor
            embeddings = self.model.process(inputs)

            for emb in embeddings:
                # Convert to float32 for numpy (bf16 not supported)
                all_embeddings.append(emb.cpu().float().numpy().reshape(1, -1))

        return all_embeddings


# ============================================================================
# Data Loading and Sampling
# ============================================================================

def load_docvqa_dataset(max_queries: int = 500, max_pages: int = 3000, seed: int = 42,
                        local_parquet: str = None):
    """
    Load DocVQA dataset with GT-First Sampling.

    Args:
        local_parquet: Optional path to local parquet file (skip HuggingFace download)

    Returns:
        selected_queries: List of (query_id, query_text, gt_page_id)
        selected_pages: Dict of page_id -> PIL.Image
        page_id_to_idx: Dict mapping page_id to integer index
    """
    from datasets import load_dataset

    if local_parquet:
        print(f"Loading DocVQA from local file: {local_parquet}")
        dataset = load_dataset("parquet", data_files=local_parquet, split="train")
    else:
        print("Loading DocVQA dataset from HuggingFace...")
        dataset = load_dataset("vidore/docvqa_test_subsampled", split="test")

    print(f"Dataset size: {len(dataset)} query-page pairs")

    random.seed(seed)
    np.random.seed(seed)

    # Each row is a (query, page) pair where the page is the GT for that query
    # Build unique pages and queries
    all_queries = []  # (query_id, query_text, gt_page_key)
    all_pages = {}    # page_key -> image

    for idx, row in enumerate(dataset):
        query_id = row['questionId']
        query_text = row['query']
        image = row['image']
        doc_id = row['docId']
        page = row['page']

        # Create unique page key
        page_key = f"{doc_id}_{page}"

        all_queries.append({
            'query_id': query_id,
            'query_text': query_text,
            'gt_page_key': page_key,
        })

        if page_key not in all_pages:
            all_pages[page_key] = image

    print(f"Unique pages: {len(all_pages)}")
    print(f"Total queries: {len(all_queries)}")

    # GT-First Sampling
    # 1. Randomly select queries
    if len(all_queries) > max_queries:
        selected_query_indices = random.sample(range(len(all_queries)), max_queries)
    else:
        selected_query_indices = list(range(len(all_queries)))

    selected_queries = [all_queries[i] for i in selected_query_indices]

    # 2. Collect GT pages for selected queries
    gt_page_keys = set(q['gt_page_key'] for q in selected_queries)
    print(f"Selected {len(selected_queries)} queries with {len(gt_page_keys)} GT pages")

    # 3. Add random negative pages to fill up to max_pages
    non_gt_page_keys = [k for k in all_pages.keys() if k not in gt_page_keys]

    if len(gt_page_keys) >= max_pages:
        # If we have more GT pages than max_pages, subsample
        selected_page_keys = set(random.sample(list(gt_page_keys), max_pages))
        # Filter queries to only those whose GT is in selected pages
        selected_queries = [q for q in selected_queries if q['gt_page_key'] in selected_page_keys]
    else:
        random_budget = max_pages - len(gt_page_keys)
        if random_budget > len(non_gt_page_keys):
            random_budget = len(non_gt_page_keys)

        random_page_keys = set(random.sample(non_gt_page_keys, random_budget))
        selected_page_keys = gt_page_keys | random_page_keys

    print(f"Final: {len(selected_queries)} queries, {len(selected_page_keys)} pages")
    print(f"  GT pages: {len(gt_page_keys)}, Random pages: {len(selected_page_keys) - len(gt_page_keys)}")

    # Build final page dict with integer indices
    selected_pages = {}
    page_key_to_idx = {}
    for idx, page_key in enumerate(sorted(selected_page_keys)):
        selected_pages[idx] = all_pages[page_key]
        page_key_to_idx[page_key] = idx

    # Update query GT to use integer indices
    for q in selected_queries:
        q['gt_page_idx'] = page_key_to_idx[q['gt_page_key']]

    return selected_queries, selected_pages, page_key_to_idx


# ============================================================================
# Main
# ============================================================================

def get_model_tag(model_path: str) -> str:
    """Extract a short model tag from model path for file naming."""
    model_path = model_path.rstrip('/')
    name = model_path.split('/')[-1].lower()
    mappings = {
        'colqwen2-v0.1': 'colqwen2',
        'colqwen2-v1.0': 'colqwen2',
        'colqwen2.5-v0.2': 'colqwen25',
        'sauerkrautlm-colqwen3-4b-v0.1': 'colqwen3',
        'tomoro-colqwen3-embed-4b': 'colqwen3',
        'qwen2-vl-2b-instruct': 'qwen2vl2b',
        'qwen2-vl-7b-instruct': 'qwen2vl7b',
        'qwen3-vl-embedding-2b': 'qwen3vlemb2b',
        'qwen3-vl-embedding-8b': 'qwen3vlemb8b',
    }
    return mappings.get(name, name.replace('-', '').replace('_', '').replace('.', ''))


def is_colqwen3_model(model_path: str) -> bool:
    """Check if model is ColQwen3 (uses sauerkrautlm_colpali)."""
    model_lower = model_path.lower()
    return 'colqwen3' in model_lower or 'sauerkrautlm' in model_lower


def is_colqwen2_model(model_path: str) -> bool:
    """Check if model is ColQwen2 (uses colpali_engine)."""
    model_lower = model_path.lower()
    return ('colqwen2' in model_lower or 'colqwen-' in model_lower) and not is_colqwen3_model(model_path)


def is_colqwen_model(model_path: str) -> bool:
    """Check if model is any ColQwen (multi-vector)."""
    return is_colqwen2_model(model_path) or is_colqwen3_model(model_path)


def is_qwen2vl_model(model_path: str) -> bool:
    """Check if model is Qwen2-VL (dense embedding extraction)."""
    model_lower = model_path.lower()
    return 'qwen2-vl' in model_lower or 'qwen2vl' in model_lower


def main():
    parser = argparse.ArgumentParser(
        description="Prepare DocVQA multi-modal data for ColQwen2 vs Qwen3-VL-Embedding comparison"
    )
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--model", type=str, default="vidore/colqwen2-v0.1",
                        help="Model: vidore/colqwen2-v0.1 or Qwen/Qwen3-VL-Embedding-2B")
    parser.add_argument("--model-tag", type=str, default=None,
                        help="Short model identifier for output filenames")
    parser.add_argument("--max-queries", type=int, default=500)
    parser.add_argument("--max-pages", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-parquet", type=str, default=None,
                        help="Path to local parquet file (skip HuggingFace download)")
    parser.add_argument("--no-modelscope", action="store_true",
                        help="Disable ModelScope download, use HuggingFace only")

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_tag = args.model_tag if args.model_tag else get_model_tag(args.model)
    print(f"Model: {args.model}")
    print(f"Model tag: {model_tag}")

    # Step 1: Load and sample dataset
    print(f"\n=== Step 1: Loading DocVQA dataset ===")
    selected_queries, selected_pages, page_key_to_idx = load_docvqa_dataset(
        max_queries=args.max_queries,
        max_pages=args.max_pages,
        seed=args.seed,
        local_parquet=args.local_parquet
    )

    # Step 2: Load encoder
    print(f"\n=== Step 2: Loading encoder ===")
    use_modelscope = not args.no_modelscope
    if is_colqwen3_model(args.model):
        encoder = ColQwen3Encoder(args.model, use_modelscope=use_modelscope)
    elif is_colqwen2_model(args.model):
        encoder = ColQwen2Encoder(args.model, use_modelscope=use_modelscope)
    elif is_qwen2vl_model(args.model):
        encoder = Qwen2VLEmbeddingEncoder(args.model, use_modelscope=use_modelscope)
    else:
        encoder = Qwen3VLEmbeddingEncoder(args.model, use_modelscope=use_modelscope)

    # Step 3: Encode pages
    print(f"\n=== Step 3: Encoding {len(selected_pages)} pages ===")
    page_images = [selected_pages[idx] for idx in range(len(selected_pages))]
    page_embeddings = encoder.encode_pages(page_images, batch_size=args.batch_size)

    if encoder.is_multi_vector:
        # Multi-vector: Save to JSONL format (for test_emb_list_msmarco.cc)
        docs_path = output_dir / f"docvqa_{model_tag}_docs.jsonl"
        doc_vec_counts = []

        with open(docs_path, 'w') as f:
            for idx, embs in enumerate(page_embeddings):
                chunks = [{"pos": k, "emb": [round(float(x), 6) for x in vec]}
                          for k, vec in enumerate(embs)]
                entry = {
                    "pid": idx,
                    "chunks": chunks,
                    "model": model_tag
                }
                f.write(json.dumps(entry) + '\n')
                doc_vec_counts.append(len(chunks))

        print(f"Saved {len(doc_vec_counts)} pages to {docs_path}")
        print(f"  Vectors per page: min={min(doc_vec_counts)}, max={max(doc_vec_counts)}, "
              f"avg={np.mean(doc_vec_counts):.1f}")

        # Step 5: Encode queries
        print(f"\n=== Step 5: Encoding {len(selected_queries)} queries ===")
        query_texts = [q['query_text'] for q in selected_queries]
        query_embeddings = encoder.encode_queries(query_texts, batch_size=args.batch_size * 2)

        # Step 6: Save queries to JSONL
        queries_path = output_dir / f"docvqa_{model_tag}_queries.jsonl"
        query_vec_counts = []

        with open(queries_path, 'w') as f:
            for idx, (query, embs) in enumerate(zip(selected_queries, query_embeddings)):
                chunks = [{"pos": k, "emb": [round(float(x), 6) for x in vec]}
                          for k, vec in enumerate(embs)]
                entry = {
                    "pid": idx,
                    "query_id": query['query_id'],
                    "text": query['query_text'][:200],
                    "chunks": chunks,
                    "gt_pids": [query['gt_page_idx']],
                    "gt_rels": {str(query['gt_page_idx']): 1},
                    "model": model_tag
                }
                f.write(json.dumps(entry) + '\n')
                query_vec_counts.append(len(chunks))

        print(f"Saved {len(query_vec_counts)} queries to {queries_path}")
    else:
        # Dense: Save to binary format (for test_dense_vs_multivec.cc)
        import struct

        # Save docs as binary: [dim:int32][num_vectors:int64][vectors:float32*]
        docs_path = output_dir / f"docvqa_{model_tag}_docs.bin"
        dim = encoder.output_dim
        num_docs = len(page_embeddings)

        with open(docs_path, 'wb') as f:
            f.write(struct.pack('i', dim))
            f.write(struct.pack('q', num_docs))
            for embs in page_embeddings:
                vec = embs.flatten().astype(np.float32)
                f.write(vec.tobytes())

        print(f"Saved {num_docs} pages to {docs_path} (dim={dim})")

        # Step 5: Encode queries
        print(f"\n=== Step 5: Encoding {len(selected_queries)} queries ===")
        query_texts = [q['query_text'] for q in selected_queries]
        query_embeddings = encoder.encode_queries(query_texts, batch_size=args.batch_size * 2)

        # Save queries as binary
        queries_path = output_dir / f"docvqa_{model_tag}_queries.bin"
        num_queries = len(query_embeddings)

        with open(queries_path, 'wb') as f:
            f.write(struct.pack('i', dim))
            f.write(struct.pack('q', num_queries))
            for embs in query_embeddings:
                vec = embs.flatten().astype(np.float32)
                f.write(vec.tobytes())

        print(f"Saved {num_queries} queries to {queries_path}")

        # Save GT as binary: [num_queries:int64][(num_gt:int64)(gt_pids:int64*) per query]
        gt_path = output_dir / f"docvqa_{model_tag}_gt.bin"
        with open(gt_path, 'wb') as f:
            f.write(struct.pack('q', num_queries))
            for query in selected_queries:
                gt_pids = [query['gt_page_idx']]
                f.write(struct.pack('q', len(gt_pids)))
                for pid in gt_pids:
                    f.write(struct.pack('q', pid))

        print(f"Saved GT to {gt_path}")

        # Save GT rels as binary (all rel=1 for DocVQA)
        gt_rels_path = output_dir / f"docvqa_{model_tag}_gt_rels.bin"
        with open(gt_rels_path, 'wb') as f:
            f.write(struct.pack('q', num_queries))
            for query in selected_queries:
                gt_pid = query['gt_page_idx']
                f.write(struct.pack('q', 1))  # num entries
                f.write(struct.pack('q', gt_pid))
                f.write(struct.pack('i', 1))  # rel=1

        print(f"Saved GT rels to {gt_rels_path}")

        doc_vec_counts = [1] * num_docs
        query_vec_counts = [1] * num_queries
    print(f"  Vectors per query: min={min(query_vec_counts)}, max={max(query_vec_counts)}, "
          f"avg={np.mean(query_vec_counts):.1f}")

    # Statistics
    print(f"\n=== Statistics ===")
    print(f"Model: {args.model} (tag={model_tag})")
    print(f"Type: {'Multi-vector' if encoder.is_multi_vector else 'Dense'}")
    print(f"Embedding dim: {encoder.output_dim}")
    print(f"Pages: {len(doc_vec_counts)}")
    print(f"  Total vectors: {sum(doc_vec_counts)}")
    print(f"Queries: {len(query_vec_counts)}")
    print(f"  GT per query: 1 (single GT)")

    # Estimate storage
    total_vectors = sum(doc_vec_counts) + sum(query_vec_counts)
    storage_mb = total_vectors * encoder.output_dim * 4 / 1e6  # FP32
    print(f"  Estimated storage (FP32): {storage_mb:.1f} MB")

    print(f"\n=== Done! ===")
    print(f"Pages: {docs_path}")
    print(f"Queries: {queries_path}")


if __name__ == "__main__":
    main()
