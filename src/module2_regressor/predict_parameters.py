"""
Predict beta, gamma, sigma from embeddings (+ optional sequence motifs).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch

from src.module2_common.features import compose_feature_vector
from src.module2_regressor.model import ParameterRegressor


def load_regressor_checkpoint(
    checkpoint_path: Union[str, Path], meta_path: Union[str, Path]
) -> Tuple[ParameterRegressor, Dict]:
    ckpt_p = Path(checkpoint_path)
    meta_p = Path(meta_path)
    if not ckpt_p.is_file():
        raise FileNotFoundError(f"Regressor checkpoint not found: {ckpt_p}")
    if not meta_p.is_file():
        raise FileNotFoundError(f"Regressor meta JSON not found: {meta_p}")

    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    model = ParameterRegressor(
        input_dim=int(meta["input_dim"]),
        hidden_dim=int(meta["hidden_dim"]),
        output_dim=int(meta["output_dim"]),
    )
    ckpt = torch.load(ckpt_p, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, meta


def predict_parameters(
    model_or_paths: Union[ParameterRegressor, Tuple[Union[str, Path], Union[str, Path]]],
    features_or_embedding,
    sequence: Optional[str] = None,
) -> Dict:
    """
    Args:
        model_or_paths: Loaded ParameterRegressor or (checkpoint_path, meta_path).
        features_or_embedding: Full feature vector matching training dim, or 1D embedding if paths bundle meta used.
        sequence: Optional HA sequence for motif features when passing raw embedding only.

    Returns:
        beta, gamma, sigma plus derived summaries (approximate R0, latent period scale).
    """
    if isinstance(model_or_paths, ParameterRegressor):
        model = model_or_paths
        meta = {}
        x = np.asarray(features_or_embedding, dtype=np.float64).ravel()
    else:
        ckpt_path, meta_path = model_or_paths
        model, meta = load_regressor_checkpoint(ckpt_path, meta_path)
        emb = np.asarray(features_or_embedding, dtype=np.float64).ravel()
        x = compose_feature_vector(emb, sequence).ravel()
        if x.shape[0] != meta.get("feature_dim", meta["input_dim"]):
            raise ValueError(
                f"Feature dimension mismatch: got {x.shape[0]}, expected {meta.get('feature_dim', meta['input_dim'])}."
            )

    device = next(model.parameters()).device
    xt = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        out = model(xt).squeeze(0).cpu().numpy()

    beta = float(max(out[0], 1e-8))
    gamma = float(max(out[1], 1e-8))
    sigma = float(max(out[2], 1e-8))

    gamma_safe = gamma if abs(gamma) > 1e-6 else 1e-6
    sigma_safe = sigma if abs(sigma) > 1e-6 else 1e-6
    r0_approx = beta / gamma_safe
    mean_latent_days_approx = 1.0 / sigma_safe

    return {
        "beta": beta,
        "gamma": gamma,
        "sigma": sigma,
        "basic_reproduction_number_approx": r0_approx,
        "mean_latent_period_days_approx": mean_latent_days_approx,
    }
