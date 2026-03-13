"""
Module 1 Master Controller: Genomic LLM Pipeline
Handles data cleaning, Hugging Face tokenization, and VRAM-optimized DNABERT training.
"""
from pathlib import Path
from src.preprocessing.data_cleaner import clean_and_structure
# We will import tokenization and training here in Phase 3 & 4

def run_module1():
    print("\n========================================")
    print("   INITIALIZING MODULE 1: GENOMIC LLM   ")
    print("========================================")
    
    project_root = Path(__file__).resolve().parents[2]
    
    # Define standard paths
    raw_fasta = project_root / "data" / "raw" / "genomic_fasta" / "sequences.fasta"
    cleaned_fasta = project_root / "data" / "processed" / "module1" / "cleaned_h5n1_ha.fasta"
    cleaned_json = project_root / "data" / "processed" / "module1" / "h5n1_metadata.json"
    
    # Step 1: Clean and Structure
    print("\n--- Step 1: Cleaning Raw NCBI Data ---")
    if not raw_fasta.exists():
        raise FileNotFoundError(f"Missing input data at: {raw_fasta}")
        
    clean_and_structure(str(raw_fasta), str(cleaned_fasta), str(cleaned_json))
    print(f"Saved structured JSON to: {cleaned_json}")

    # Step 2: Tokenization (Placeholder for Phase 3)
    print("\n--- Step 2: Hugging Face Tokenization ---")
    print("[Pending Phase 3 Integration]")

    # Step 3: Model Training (Placeholder for Phase 4)
    print("\n--- Step 3: VRAM-Optimized Model Training ---")
    print("[Pending Phase 4 Integration]")

if __name__ == "__main__":
    run_module1()