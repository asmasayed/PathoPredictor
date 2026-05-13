"""
Feature construction for Module 2: combine sequence-derived motifs with embeddings.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def sequence_motif_features(sequence: Optional[str]) -> np.ndarray:
    """
    Lightweight HA-oriented scalar features (no Module 1 dependency).
    Polybasic motifs near the cleavage site are associated with high pathogenicity in H5.
    """
    if not sequence:
        return np.zeros(5, dtype=np.float64)

    s = "".join(sequence.upper().split())
    n = max(len(s), 1)
    gc = (s.count("G") + s.count("C")) / n
    ambig = s.count("N") / n

    motifs = ("RRRK", "RRRKR", "GERRRKR", "PQRETR", "GKRA")
    motif_hits = float(sum(s.count(m) for m in motifs))

    # Normalize length to typical HA ~1.7kb
    len_norm = min(len(s) / 1700.0, 3.0)

    return np.array([gc, ambig, motif_hits, len_norm, float(len(s))], dtype=np.float64)


def compose_feature_vector(embedding: np.ndarray, sequence: Optional[str] = None) -> np.ndarray:
    """Concatenate embedding (1D) with motif features."""
    emb = np.asarray(embedding, dtype=np.float64).ravel()
    motifs = sequence_motif_features(sequence)
    return np.concatenate([emb, motifs], axis=0)
