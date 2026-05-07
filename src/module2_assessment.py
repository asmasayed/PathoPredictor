"""
Single entrypoint for Module 2: host adaptation + epidemiological phenotype from embeddings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

from src.config.config import MODULE2_CLASSIFIER_CONFIG, MODULE2_REGRESSOR_CONFIG
from src.module2_classifier.predict_host_risk import predict_host_adaptation
from src.module2_regressor.predict_parameters import predict_parameters


def _clinical_indicators(human_p: float, r0_approx: float, latent_days: float) -> Dict:
    """Qualitative summaries for dashboards (not diagnostic)."""
    transmission = (
        "elevated_transmissibility_potential" if r0_approx > 1.5 else "moderate_transmissibility_potential"
    )
    severity_hint = (
        "higher_relative_public_health_attention_warranted"
        if human_p > 0.65 and r0_approx > 1.2
        else "routine_monitoring"
    )
    incubation_hint = (
        "longer_latent_period"
        if latent_days > 2.5
        else "shorter_latent_period"
        if latent_days < 1.5
        else "moderate_latent_period"
    )
    return {
        "transmission_summary": transmission,
        "severity_attention": severity_hint,
        "latent_period_summary": incubation_hint,
    }


def run_phenotype_assessment(
    embedding: List[float],
    sequence: Optional[str] = None,
    classifier_bundle_path: Optional[Union[str, Path]] = None,
    regressor_ckpt_path: Optional[Union[str, Path]] = None,
    regressor_meta_path: Optional[Union[str, Path]] = None,
) -> Dict:
    """
    Load trained Module 2 artifacts (defaults from config) and return unified JSON-ready dict.
    """
    clf_path = Path(classifier_bundle_path or MODULE2_CLASSIFIER_CONFIG["model_bundle_path"])
    ckpt_path = Path(regressor_ckpt_path or MODULE2_REGRESSOR_CONFIG["checkpoint_path"])
    meta_path = Path(regressor_meta_path or MODULE2_REGRESSOR_CONFIG["meta_path"])

    host = predict_host_adaptation(clf_path, embedding, sequence=sequence)
    seir = predict_parameters((ckpt_path, meta_path), embedding, sequence=sequence)

    human_p = host["human_adaptation_probability"]
    r0_a = float(seir["basic_reproduction_number_approx"])
    latent = float(seir["mean_latent_period_days_approx"])

    return {
        "human_adaptation": host,
        "epidemiological_parameters": {
            "beta": seir["beta"],
            "gamma": seir["gamma"],
            "sigma": seir["sigma"],
            "basic_reproduction_number_approx": r0_a,
            "mean_latent_period_days_approx": latent,
        },
        "clinical_indicators": _clinical_indicators(human_p, r0_a, latent),
    }
