"""
Merge metadata, embedding table, and optional phenotype parameters into Module 2 training CSVs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.config.config import SEIR_DEFAULT_PARAMS
from src.preprocessing.phenotype_builder import host_label_binary


def _embedding_columns(df: pd.DataFrame) -> List[str]:
    return sorted([c for c in df.columns if c.startswith("emb_")], key=lambda x: int(x.split("_")[1]))


def load_metadata_records(path: Path) -> List[Dict]:
    text = path.read_text(encoding="utf-8")
    data = json.loads(text)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise ValueError("Metadata JSON must be a list of records or a single object.")


def build_classifier_csv(
    metadata_path: Path,
    embeddings_path: Path,
    out_path: Path,
) -> int:
    records = load_metadata_records(metadata_path)
    meta = pd.DataFrame(records)
    if "strain_id" not in meta.columns or "host" not in meta.columns:
        raise ValueError("Metadata must include strain_id and host columns.")
    emb = pd.read_csv(embeddings_path)
    if "strain_id" not in emb.columns:
        raise ValueError("Embeddings CSV must include strain_id.")
    df = meta.merge(emb, on="strain_id", how="inner")
    df["label"] = df["host"].map(host_label_binary)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    emb_cols = _embedding_columns(df)
    if not emb_cols:
        raise ValueError("No emb_* columns after merge.")
    out_cols = emb_cols + ["label"]
    if "sequence" in df.columns:
        out_cols = emb_cols + ["sequence", "label"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df[out_cols].to_csv(out_path, index=False)
    return len(df)


def build_regressor_csv(
    metadata_path: Path,
    embeddings_path: Path,
    out_path: Path,
    phenotype_path: Optional[Path] = None,
    fallback_seir: bool = False,
    seir_jitter: float = 0.02,
    seed: int = 42,
) -> int:
    records = load_metadata_records(metadata_path)
    meta = pd.DataFrame(records)
    emb = pd.read_csv(embeddings_path)
    df = meta.merge(emb, on="strain_id", how="inner")
    emb_cols = _embedding_columns(df)

    if phenotype_path and phenotype_path.is_file():
        pheno = pd.read_csv(phenotype_path)
        if "strain_id" not in pheno.columns:
            raise ValueError("Phenotype CSV must include strain_id.")
        for col in ("beta", "gamma", "sigma"):
            if col not in pheno.columns:
                raise ValueError(f"Phenotype CSV must include {col}.")
        df = df.merge(pheno[["strain_id", "beta", "gamma", "sigma"]], on="strain_id", how="inner")
    elif fallback_seir:
        rng = np.random.default_rng(seed)
        b0 = float(SEIR_DEFAULT_PARAMS["beta"])
        g0 = float(SEIR_DEFAULT_PARAMS["gamma"])
        s0 = float(SEIR_DEFAULT_PARAMS["sigma"])
        n = len(df)
        df["beta"] = np.clip(b0 + rng.normal(0, seir_jitter, n), 0.05, 0.8)
        df["gamma"] = np.clip(g0 + rng.normal(0, seir_jitter, n), 0.03, 0.4)
        df["sigma"] = np.clip(s0 + rng.normal(0, seir_jitter, n), 0.05, 0.5)
        print(
            "WARNING: Using fallback SEIR targets with jitter around SEIR_DEFAULT_PARAMS; "
            "replace with literature- or calibration-derived beta, gamma, sigma for real use."
        )
    else:
        raise FileNotFoundError(
            "Regressor training needs labeled parameters. Pass --phenotype-csv with columns "
            "strain_id,beta,gamma,sigma or use --fallback-seir for a placeholder demo only."
        )

    out_cols = emb_cols + ["beta", "gamma", "sigma"]
    if "sequence" in df.columns:
        out_cols = emb_cols + ["sequence", "beta", "gamma", "sigma"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df[out_cols].to_csv(out_path, index=False)
    return len(df)
