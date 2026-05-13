"""
Load tabular training data for epidemiological parameter regression.
Expected columns: beta, gamma, sigma, emb_0, emb_1, ...
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


def load_regressor_training_arrays(csv_path: str | Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Regressor training CSV not found: {path}")

    df = pd.read_csv(path)
    for col in ("beta", "gamma", "sigma"):
        if col not in df.columns:
            raise ValueError(f"CSV must include target column '{col}'.")

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
    y = df[["beta", "gamma", "sigma"]].to_numpy(dtype=np.float64)
    return X, y, emb_cols


def write_synthetic_regressor_csv(out_path: str | Path, n_samples: int = 200, embed_dim: int = 16, seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    X = rng.normal(size=(n_samples, embed_dim))
    # Toy linear mapping so the MLP has something to fit
    w = rng.normal(size=embed_dim)
    z = X @ w
    beta = np.clip(0.15 + 0.08 * np.tanh(z) + rng.normal(0, 0.02, n_samples), 0.05, 0.6)
    gamma = np.clip(0.08 + 0.04 * np.tanh(z * 0.5) + rng.normal(0, 0.01, n_samples), 0.03, 0.25)
    sigma = np.clip(0.12 + 0.05 * np.tanh(-z) + rng.normal(0, 0.015, n_samples), 0.05, 0.35)

    cols = {f"emb_{i}": X[:, i] for i in range(embed_dim)}
    cols["beta"] = beta
    cols["gamma"] = gamma
    cols["sigma"] = sigma
    pd.DataFrame(cols).to_csv(out_path, index=False)
