"""
Main pipeline script for PathoPredictor.
"""

import sys
import os
from initialize_vocab import initialize_genomic_vocabulary

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.config.config import (
    MODULE1_CONFIG,
    MODULE2_CLASSIFIER_CONFIG,
    MODULE2_REGRESSOR_CONFIG,
    MODULE3_CONFIG
)

def run_pipeline():
    """Run the complete PathoPredictor pipeline."""
    print("Starting PathoPredictor pipeline...")
    
    # Step 1: Initialize Vocabulary
    # This prepares the vocab.json that all subsequent modules will use
    # print("\n=== Step 1: Initializing Genomic Vocabulary ===")
    # initialize_genomic_vocabulary() 

    # Module 1: Train genomic LLM
    print("\n=== Module 1: Genomic LLM ===")
    from src.module1_genomic_llm.train_dnbert import train_dnbert
    train_dnbert(MODULE1_CONFIG)
    
    # Module 2: Train classifier and regressor
    print("\n=== Module 2: Classifier ===")
    from src.module2_classifier.train_classifier import train_classifier
    train_classifier(MODULE2_CLASSIFIER_CONFIG)
    
    print("\n=== Module 2: Regressor ===")
    from src.module2_regressor.train_regressor import train_regressor
    train_regressor(MODULE2_REGRESSOR_CONFIG)
    
    # Module 3: Train LSTM
    print("\n=== Module 3: LSTM ===")
    from src.module3_lstm.train_lstm import train_lstm
    recent_memory, scaler, last_real_cases = train_lstm(MODULE3_CONFIG)
    
    # MODULE 4: SEIR Simulation (YOUR HYBRID ENGINE)
    print("\n=== Module 4: SEIR Simulation ===")
    from src.simulation.seir_sim import run_simulation
    run_simulation(recent_memory, scaler, last_real_cases)
    
    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    run_pipeline()
