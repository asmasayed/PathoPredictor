"""
Mean-pooled DNABERT / k-mer MLM embeddings for HA sequences.
Uses the same k-mer tokenization style as `generate_mutations.py` (6-mers, 512 cap).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


def seq_to_kmers(seq: str, k: int = 6) -> str:
    seq = seq.upper().replace("-", "")
    if len(seq) < k:
        return seq
    return " ".join([seq[i : i + k] for i in range(len(seq) - k + 1)])


def windowed_subsequences(seq: str, win: int = 500, stride: int = 400) -> Iterable[str]:
    """Slide along long HA (~1.7kb); each window is embedded and averaged."""
    seq = "".join(c for c in seq.upper() if c in "ATCGN")
    if len(seq) <= win:
        if len(seq) == 0:
            return
        yield seq
        return
    for start in range(0, len(seq) - win + 1, stride):
        yield seq[start : start + win]
    last_start = len(seq) - win
    if last_start % stride != 0:
        yield seq[last_start:]


def _mean_pool_hidden(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom


def load_mlm_model(model_dir: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForMaskedLM.from_pretrained(model_dir).to(device)
    model.eval()
    return tokenizer, model


_mlm_cache: Dict[Tuple[str, str], Tuple[Any, Any]] = {}


def get_cached_mlm(model_dir: str, device: Optional[torch.device] = None):
    """Reuse one tokenizer+model per (dir, device) for API latency."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    key = (str(Path(model_dir).resolve()), str(device))
    if key not in _mlm_cache:
        tok, mod = load_mlm_model(model_dir, device)
        _mlm_cache[key] = (tok, mod)
    return _mlm_cache[key], device


def compute_embedding_vector(sequence: str, model_dir: str) -> np.ndarray:
    """Single-sequence embedding (same pooling as batch CLI)."""
    (tokenizer, model), device = get_cached_mlm(model_dir)
    return embed_sequence(model, tokenizer, sequence, device)


@torch.no_grad()
def embed_sequence(
    model: AutoModelForMaskedLM,
    tokenizer,
    sequence: str,
    device: torch.device,
    k: int = 6,
    max_length: int = 512,
) -> np.ndarray:
    vecs: List[np.ndarray] = []
    for sub in windowed_subsequences(sequence, 500, 400):
        kmer = seq_to_kmers(sub, k)
        if len(kmer.strip()) < 8:
            continue
        inputs = tokenizer(
            kmer,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        inputs = {kk: vv.to(device) for kk, vv in inputs.items()}
        outputs = model(**inputs, output_hidden_states=True)
        last = outputs.hidden_states[-1]
        pooled = _mean_pool_hidden(last, inputs["attention_mask"])
        vecs.append(pooled.squeeze(0).float().cpu().numpy())
    if not vecs:
        raise ValueError("Sequence too short to embed after k-mer tokenization.")
    return np.stack(vecs, axis=0).mean(axis=0)


def embed_records_to_dataframe(
    records: List[Dict[str, Any]],
    model_dir: str,
    device: Optional[torch.device] = None,
):
    """
    records: dicts with keys strain_id, sequence (same shape as cleaned metadata JSON).
    Returns pandas DataFrame with strain_id + emb_0..emb_{d-1}.
    """
    import pandas as pd

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = load_mlm_model(model_dir, device)
    rows = []
    for i, rec in enumerate(records):
        sid = rec.get("strain_id")
        seq = rec.get("sequence")
        if not sid or not seq:
            continue
        try:
            emb = embed_sequence(model, tokenizer, str(seq), device)
        except Exception as exc:
            print(f"[skip] {sid}: {exc}")
            continue
        row = {"strain_id": sid, **{f"emb_{j}": float(emb[j]) for j in range(len(emb))}}
        rows.append(row)
        if (i + 1) % 10 == 0:
            print(f"Embedded {i + 1}/{len(records)} strains...")
    if not rows:
        raise RuntimeError("No embeddings produced; check JSON and sequences.")
    return pd.DataFrame(rows)
