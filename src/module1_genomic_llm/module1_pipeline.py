"""
Module 1 Master Controller: Genomic LLM Pipeline
Handles data cleaning, tokenization, training, embeddings, and API-ready mutation generation.
"""
from pathlib import Path
import json

# Import Phase 1 Core Functions
from src.preprocessing.data_cleaner import clean_and_structure
from src.module1_genomic_llm.tokenize_sequences import build_hf_dataset
from src.module1_genomic_llm.train_dnbert import train_dnbert
from src.module1_genomic_llm.extract_embeddings import extract_and_save_embeddings

# Import Phase 2 API Functions (The New Additions)
from src.module1_genomic_llm.process_uploaded_fasta import clean_uploaded_fasta
from src.module1_genomic_llm.generate_mutations import process_file

def run_module1():
    print("\n========================================")
    print("   INITIALIZING MODULE 1: GENOMIC LLM   ")
    print("========================================")
    
    project_root = Path(__file__).resolve().parents[2]
    
    # --- PHASE 1 PATHS (Core Engine) ---
    raw_fasta = project_root / "data" / "raw" / "genomic_fasta" / "sequences.fasta"
    cleaned_fasta = project_root / "data" / "processed" / "module1" / "cleaned_h5n1_ha.fasta"
    cleaned_json = project_root / "data" / "processed" / "module1" / "h5n1_metadata.json"
    hf_dataset_dir = project_root / "data" / "processed" / "module1" / "hf_tokenized_dataset"
    model_output_dir = project_root / "models" / "module1_dnbert"
    embeddings_output = project_root / "data" / "processed" / "module1" / "h5n1_embeddings.pt"
    
    # --- PHASE 2 PATHS (User Interaction) ---
    sample_upload_fasta = project_root / "data" / "raw" / "sample_upload.fasta"
    cleaned_upload_json = project_root / "data" / "processed" / "module1" / "cleaned_upload.json"


    # ==========================================
    # PHASE 1: CORE TRAINING PIPELINE
    # ==========================================

    # Step 1: Clean and Structure
    print("\n--- Step 1: Cleaning Raw NCBI Data ---")
    # clean_and_structure(str(raw_fasta), str(cleaned_fasta), str(cleaned_json))
    print("[Skipping: Already completed]")
    
    # Step 2: Tokenization (Arrow Dataset Upgrade)
    print("\n--- Step 2: Hugging Face Tokenization ---")
    # build_hf_dataset(str(cleaned_json), str(hf_dataset_dir))
    print("[Skipping: Already completed]")

    # Step 3: Model Training
    print("\n--- Step 3: VRAM-Optimized Model Training ---")
    # train_dnbert(str(hf_dataset_dir), str(model_output_dir))
    print("[Skipping: Already completed]")

    # Step 4: Extract Embeddings
    print("\n--- Step 4: Extracting Learned Embeddings ---")
    # extract_and_save_embeddings(str(cleaned_json), str(model_output_dir), str(embeddings_output))
    print("[Skipping: Already completed]")


    # ==========================================
    # PHASE 2: USER INFERENCE PIPELINE
    # ==========================================

    # Step 5: Process an Uploaded FASTA file
    print("\n--- Step 5: Processing User-Uploaded FASTA ---")
    
    # Auto-generate a dummy user upload file for testing if it doesn't exist
    if not sample_upload_fasta.exists():
        print(f"Creating a mock user upload file at {sample_upload_fasta}...")
        sample_upload_fasta.parent.mkdir(parents=True, exist_ok=True)
        with open(sample_upload_fasta, "w") as f:
            f.write(">Test_User_Upload_Strain\nATGGAGAAAATAGTGCTTCTTCTTGCAATAGTCAGTCTTGTTAAAAGTGATCAGATTTGCATTGGTTACCATGCAAACAATTCAACAGAGCAGGTTGACACAATCATGGAAAAGAACGTTACTGTTACACATGCC")

    # Run the cleaner
    cleaned_json_string = clean_uploaded_fasta(str(sample_upload_fasta))
    
    # Save it to disk so the next step can read it (simulating how the backend will handle files)
    with open(cleaned_upload_json, "w") as f:
        f.write(cleaned_json_string)
    print(f"Cleaned upload saved to {cleaned_upload_json}")


    # Step 6: Generate Mutations
    print("\n--- Step 6: Predicting Biologically Plausible Mutations ---")
    # Feed the cleaned JSON and the model directory to the mutation generator
    mutation_results_json = process_file(str(cleaned_upload_json), str(model_output_dir))
    
    print("\n========================================")
    print("      FINAL OUTPUT FOR REACT FRONTEND     ")
    print("========================================")
    print(mutation_results_json)

if __name__ == "__main__":
    run_module1()