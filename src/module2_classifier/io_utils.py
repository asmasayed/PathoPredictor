"""
Load tabular training data for the host-adaptation classifier.
Expected columns: label (0=avian-like, 1=human/mammal-adapted), emb_0, emb_1, ...
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from src.module2_common.features import sequence_motif_features


def list_embedding_columns(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if re.match(r"^emb_\d+$", str(c))]
    return sorted(cols, key=lambda x: int(x.split("_")[1]))


def load_classifier_training_arrays(csv_path: str | Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Classifier training CSV not found: {path}")

    df = pd.read_csv(path)
    if "label" not in df.columns:
        raise ValueError("CSV must include a 'label' column (0=avian, 1=human-adapted).")

    emb_cols = list_embedding_columns(df)
    if not emb_cols:
        raise ValueError("CSV must include embedding columns emb_0, emb_1, ...")

    X_emb = df[emb_cols].to_numpy(dtype=np.float64)
    if "sequence" in df.columns:
        motifs = np.stack(
            [sequence_motif_features(str(s) if pd.notna(s) else None) for s in df["sequence"]],
            axis=0,
        )
    else:
        motifs = np.zeros((len(df), 5), dtype=np.float64)

    X = np.hstack([X_emb, motifs])
    y = df["label"].to_numpy(dtype=np.int64)
    return X, y, emb_cols


def write_synthetic_classifier_csv(out_path: str | Path, n_samples: int = 160, embed_dim: int = 16, seed: int = 42) -> None:
    """Create a small demo dataset for local training smoke tests."""
    rng = np.random.default_rng(seed)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    X = rng.normal(size=(n_samples, embed_dim))
    # Weakly separate classes for a learnable toy problem
    labels = (X.sum(axis=1) + rng.normal(size=n_samples) * 0.5 > 0).astype(np.int64)
    cols = {f"emb_{i}": X[:, i] for i in range(embed_dim)}
    cols["label"] = labels
    pd.DataFrame(cols).to_csv(out_path, index=False)
