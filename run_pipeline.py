"""
Main pipeline script for PathoPredictor.
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

#from src.module1_genomic_llm.module1_pipeline import run_module1
from src.module3_lstm.module3_pipeline import run_module3_seir

def run_pipeline():
    print("Starting PathoPredictor Global Pipeline...")

    print("\n=== Module 1: Genomic LLM ===")
    #run_module1()
    
    # Module 2: Train classifier and regressor
    print("\n=== Module 2: A. Classifier ===")
    #from src.module2_classifier.train_classifier import train_classifier
    #train_classifier(MODULE2_CLASSIFIER_CONFIG)
    
    print("\n=== Module 2: B. Regressor ===")
    #from src.module2_regressor.train_regressor import train_regressor
    #train_regressor(MODULE2_REGRESSOR_CONFIG)
    
    print("\n=== Module 3 + SEIR Simulation ===")
    run_module3_seir()

    print("\nPipeline execution finished!")

if __name__ == "__main__":
    run_pipeline()

    try:
        import build_dashboard
    except Exception as e:
        print(f"Dashboard script issue: {e}")
