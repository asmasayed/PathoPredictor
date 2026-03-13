"""
Module 1 Master Controller: Genomic LLM Pipeline
Handles data cleaning, Hugging Face tokenization, VRAM-optimized DNABERT training, and Embedding Extraction.
"""
from pathlib import Path
from src.preprocessing.data_cleaner import clean_and_structure
from src.module1_genomic_llm.tokenize_sequences import build_hf_dataset
from src.module1_genomic_llm.train_dnbert import train_dnbert
from src.module1_genomic_llm.extract_embeddings import extract_and_save_embeddings 

def run_module1():
    print("\n========================================")
    print("   INITIALIZING MODULE 1: GENOMIC LLM   ")
    print("========================================")
    
    project_root = Path(__file__).resolve().parents[2]
    
    # Define standard paths
    raw_fasta = project_root / "data" / "raw" / "genomic_fasta" / "sequences.fasta"
    cleaned_fasta = project_root / "data" / "processed" / "module1" / "cleaned_h5n1_ha.fasta"
    cleaned_json = project_root / "data" / "processed" / "module1" / "h5n1_metadata.json"
    hf_dataset_dir = project_root / "data" / "processed" / "module1" / "hf_tokenized_dataset"
    model_output_dir = project_root / "models" / "module1_dnbert"
    embeddings_output = project_root / "data" / "processed" / "module1" / "h5n1_embeddings.pt"
    
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
    #train_dnbert(str(hf_dataset_dir), str(model_output_dir))
    print("[Skipping: Already completed]")

    # Step 4: Extract Embeddings
    print("\n--- Step 4: Extracting Learned Embeddings ---")
    extract_and_save_embeddings(str(cleaned_json), str(model_output_dir), str(embeddings_output))

if __name__ == "__main__":
    run_module1()