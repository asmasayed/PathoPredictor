"""
Main pipeline script for PathoPredictor.
"""

import sys
import os
import json
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
    print("Starting PathoPredictor Global Pipeline...")
    
    # Step 1: Initialize Vocabulary
    # print("\n=== Step 1: Initializing Genomic Vocabulary ===")
    # initialize_genomic_vocabulary() 

    # Module 1: Train genomic LLM
    print("\n=== Module 1: Genomic LLM ===")
    #from src.module1_genomic_llm.train_dnbert import train_dnbert
    #train_dnbert(MODULE1_CONFIG)
    
    # Module 2: Train classifier and regressor
    print("\n=== Module 2: Classifier ===")
    #from src.module2_classifier.train_classifier import train_classifier
    #train_classifier(MODULE2_CLASSIFIER_CONFIG)
    
    print("\n=== Module 2: Regressor ===")
    #from src.module2_regressor.train_regressor import train_regressor
    #train_regressor(MODULE2_REGRESSOR_CONFIG)
    
    # ==========================================
    # MULTI-CONTINENT AI & MATH ENGINE (Modules 3 & 4)
    # ==========================================
    from src.module3_lstm.train_lstm import train_lstm
    from src.simulation.seir_sim import run_simulation
    
    # Mathematically defined population sizes for accurate relative scaling
    regions = {
        "us": {"N": 600000},
        "africa": {"N": 800000},
        "asia": {"N": 1000000}
    }
    
    global_forecasts = {}

    for region, params in regions.items():
        print(f"\n==========================================")
        print(f"🌍 PROCESSING REGION: {region.upper()}")
        print(f"==========================================")
        
        # Step 1: Train the specialized AI brain for this specific climate
        print(f"\n[1/2] Training Neural Network...")
        recent_memory, scaler, last_real_cases = train_lstm(MODULE3_CONFIG, region=region)
        
        # Step 2: Run the differential calculus simulation using the trained brain
        print(f"\n[2/2] Running Differential SEIR Calculus...")
        seir_results = run_simulation(
            recent_memory=recent_memory, 
            scaler=scaler, 
            last_real_cases=last_real_cases, 
            region=region, 
            N=params["N"]
        )
        
        # Step 3: Store the mathematically proven curves in memory
        global_forecasts[region.upper()] = seir_results
        
    # Step 4: Save the master data to a JSON file so the Dashboard can read it instantly
    print("\n💾 Saving master global forecast data...")
    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/global_seir_forecasts.json", "w") as f:
        json.dump(global_forecasts, f)
        
    print("\n✅ PathoPredictor Pipeline completed successfully!")

if __name__ == "__main__":
    run_pipeline()
    
    # The final step will be triggering the new dashboard!
    try:
        import build_dashboard
    except Exception as e:
        print(f"Note: Dashboard script encountered an issue (we will fix this next!): {e}")
