"""
Configuration file for PathoPredictor project.
Contains all configuration parameters and settings.
"""

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
}

MODULE2_REGRESSOR_CONFIG = {
    "model_name": "parameter_regressor",
    "batch_size": 64,
    "learning_rate": 1e-3,
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
