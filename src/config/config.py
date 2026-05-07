"""
Configuration file for PathoPredictor project.
Contains all configuration parameters and settings.
"""

from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data paths
DATA_RAW_PATH = "data/raw"
DATA_PROCESSED_PATH = "data/processed"
DATA_EXTERNAL_PATH = "data/external"

# Model paths
MODELS_PATH = "models"

# Module-specific configurations
MODULE1_CONFIG = {
    "model_name": "dnbert",
    "batch_size": 32,
    "learning_rate": 1e-4,
}

MODULE2_CLASSIFIER_CONFIG = {
    "model_name": "host_risk_classifier",
    "batch_size": 64,
    "learning_rate": 1e-3,
    "train_csv": str(_PROJECT_ROOT / "data/processed/module2/classifier_train.csv"),
    "model_bundle_path": str(_PROJECT_ROOT / "models/module2_classifier/classifier_bundle.joblib"),
    "val_fraction": 0.2,
    "seed": 42,
    "n_estimators": 200,
    "max_depth": None,
    "synthetic_embed_dim": 16,
}

MODULE2_REGRESSOR_CONFIG = {
    "model_name": "parameter_regressor",
    "batch_size": 64,
    "learning_rate": 1e-3,
    "train_csv": str(_PROJECT_ROOT / "data/processed/module2/regressor_train.csv"),
    "checkpoint_path": str(_PROJECT_ROOT / "models/module2_regressor/regressor.pt"),
    "meta_path": str(_PROJECT_ROOT / "models/module2_regressor/regressor_meta.json"),
    "hidden_dim": 128,
    "epochs": 80,
    "val_fraction": 0.2,
    "seed": 7,
    "synthetic_embed_dim": 16,
}

MODULE3_CONFIG = {
    "model_name": "lstm_time_series",
    "batch_size": 32,
    "learning_rate": 1e-3,
    "sequence_length": 100,
}

# SEIR Model parameters
SEIR_DEFAULT_PARAMS = {
    "beta": 0.3,
    "gamma": 0.1,
    "sigma": 0.2,
}
