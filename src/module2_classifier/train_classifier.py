"""
Train Module 2 host-adaptation classifier (Random Forest on embeddings + motif features).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config.config import MODULE2_CLASSIFIER_CONFIG
from src.module2_classifier.io_utils import load_classifier_training_arrays, write_synthetic_classifier_csv


def train_classifier(config: Optional[Dict] = None) -> Dict:
    """
    Fit RandomForestClassifier and persist scaler + model bundle.

    CSV format:
      - label: 0 avian-like, 1 human/mammal-adapted
      - emb_0 ... emb_{d-1}: embedding dimensions from Module 1 (or surrogate vectors)
      - optional sequence column for motif-augmented training
    """
    cfg = config or MODULE2_CLASSIFIER_CONFIG
    csv_path = Path(cfg["train_csv"])
    bundle_path = Path(cfg["model_bundle_path"])

    if not csv_path.is_file():
        print(f"No training CSV at {csv_path}; writing synthetic demo data.")
        write_synthetic_classifier_csv(csv_path, embed_dim=int(cfg.get("synthetic_embed_dim", 16)))

    X, y, emb_cols = load_classifier_training_arrays(csv_path)

    stratify = y if len(np.unique(y)) > 1 else None
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=float(cfg.get("val_fraction", 0.2)),
        random_state=int(cfg.get("seed", 42)),
        stratify=stratify,
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    clf = RandomForestClassifier(
        n_estimators=int(cfg.get("n_estimators", 200)),
        max_depth=cfg.get("max_depth"),
        class_weight="balanced_subsample",
        random_state=int(cfg.get("seed", 42)),
        n_jobs=-1,
    )
    clf.fit(X_train_s, y_train)

    val_pred = clf.predict(X_val_s)
    metrics = {"val_accuracy": float(accuracy_score(y_val, val_pred))}
    if len(np.unique(y_val)) > 1:
        try:
            metrics["val_roc_auc"] = float(roc_auc_score(y_val, clf.predict_proba(X_val_s)[:, 1]))
        except ValueError:
            metrics["val_roc_auc"] = None
    print(classification_report(y_val, val_pred, digits=3))
    print("Validation metrics:", metrics)

    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": clf,
        "scaler": scaler,
        "embedding_columns": emb_cols,
        "motif_feature_dim": 5,
        "feature_dim": X.shape[1],
    }
    joblib.dump(bundle, bundle_path)
    print(f"Saved classifier bundle to {bundle_path}")
    return metrics


if __name__ == "__main__":
    train_classifier(MODULE2_CLASSIFIER_CONFIG)
