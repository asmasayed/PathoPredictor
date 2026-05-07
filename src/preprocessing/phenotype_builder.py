"""
Phenotype data builder for constructing phenotype features.
"""


from typing import Optional

import pandas as pd


def host_label_binary(host: Optional[str]) -> Optional[int]:
    """
    Map free-text host metadata to a binary training label for Module 2 classifiers.
    Returns 0 avian-like, 1 mammal/human-like, or None if unknown.
    """
    if host is None or (isinstance(host, float) and pd.isna(host)):
        return None
    h = str(host).strip().lower()

    avian_keywords = ("avian", "bird", "chicken", "duck", "goose", "turkey", "quail", "wild bird")
    mammal_keywords = (
        "human",
        "pig",
        "swine",
        "cat",
        "dog",
        "mink",
        "seal",
        "ferret",
        "cow",
        "bovine",
        "mouse",
        "rat",
        "mammal",
    )

    if any(k in h for k in avian_keywords):
        return 0
    if any(k in h for k in mammal_keywords):
        return 1
    return None


def build_phenotype_features(phenotype_data_path):
    """
    Build phenotype features from raw phenotype data.
    
    Args:
        phenotype_data_path: Path to phenotype data file
        
    Returns:
        DataFrame with phenotype features
    """
    df = pd.read_csv(phenotype_data_path)
    # Feature engineering logic here
    return df