"""
Host adaptation inference using the trained Random Forest bundle.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

import joblib
import numpy as np

from src.module2_common.features import compose_feature_vector


def load_classifier_bundle(bundle_path: Union[str, Path]) -> Dict:
    path = Path(bundle_path)
    if not path.is_file():
        raise FileNotFoundError(f"Classifier bundle not found: {path}")
    return joblib.load(path)


def predict_host_adaptation(
    bundle_or_path: Union[Dict, str, Path],
    embedding: Union[List[float], np.ndarray],
    sequence: Optional[str] = None,
) -> Dict:
    """
    Args:
        bundle_or_path: Loaded bundle dict or path to classifier_bundle.joblib
        embedding: 1D feature vector matching training emb_* dimensionality.
        sequence: Optional HA nucleotide sequence for motif augmentation.

    Returns:
        human_adaptation_probability, predicted_label, risk_score_percent (0-100)
    """
    bundle = bundle_or_path if isinstance(bundle_or_path, dict) else load_classifier_bundle(bundle_or_path)
    model = bundle["model"]
    scaler = bundle["scaler"]

    x = compose_feature_vector(np.asarray(embedding, dtype=np.float64), sequence).reshape(1, -1)
    expected = bundle.get("feature_dim")
    if expected is not None and x.shape[1] != expected:
        raise ValueError(
            f"Feature dimension mismatch: got {x.shape[1]}, bundle expects {expected}. "
            "Ensure embedding length matches training emb_* columns."
        )

    xs = scaler.transform(x)
    proba = model.predict_proba(xs)[0]
    classes = list(model.classes_)
    try:
        human_idx = classes.index(1)
    except ValueError:
        human_idx = int(np.argmax(classes))

    human_p = float(proba[human_idx])
    pred_label = int(model.predict(xs)[0])

    return {
        "human_adaptation_probability": human_p,
        "predicted_host_label": pred_label,
        "risk_score_percent": round(100.0 * human_p, 2),
        "class_probabilities": {str(c): float(p) for c, p in zip(classes, proba)},
    }
