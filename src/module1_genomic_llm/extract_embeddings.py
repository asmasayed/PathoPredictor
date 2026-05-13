"""
Module 1 - Step 5: Extract Sequence Embeddings
Runs the cleaned H5N1 sequences through the fine-tuned DNABERT model
to generate numerical embeddings for downstream tasks (Modules 2 & 3).
"""
import os
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

def seq_to_kmers(seq: str, k: int = 6) -> str:
    """Convert a nucleotide sequence into a space-separated string of k-mers."""
    return " ".join([seq[i : i + k] for i in range(len(seq) - k + 1)])

def extract_and_save_embeddings(
    json_path: str, 
    model_dir: str, 
    output_path: str, 
    max_length: int = 512
) -> None:
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Initializing Embedding Extraction on {device} ---")

    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Fine-tuned model not found at {model_dir}. Wait for training to finish!")

    # 1. Load the fine-tuned model and tokenizer
    # We use AutoModel here (not AutoModelForMaskedLM) because we want the raw hidden states, not the MLM predictions
    print("Loading fine-tuned model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir).to(device)
    model.eval() # Put model in evaluation mode (turns off dropout)

    # 2. Load the cleaned sequence data
    print(f"Loading cleaned H5N1 data from {json_path}...")
    with open(json_path, "r") as f:
        data = json.load(f)

    embeddings_dict = {}

    # 3. Extract embeddings
    print("Extracting embeddings per sequence...")
    # torch.no_grad() is crucial here! It tells PyTorch not to store gradients, saving massive amounts of VRAM
    with torch.no_grad():
        for item in tqdm(data, desc="Extracting"):
            strain_id = item["strain_id"]
            kmer_string = seq_to_kmers(item["sequence"])

            # Tokenize and automatically chunk the long sequence
            inputs = tokenizer(
                kmer_string,
                truncation=True,
                max_length=max_length,
                stride=50,
                return_overflowing_tokens=True,
                padding="max_length",
                return_tensors="pt"
            )
            
            # Remove the mapping column so the model doesn't complain
            inputs.pop("overflow_to_sample_mapping", None)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Pass the chunks through the model
            outputs = model(**inputs)
            
            # Extract the [CLS] token representation for each chunk (the 0th token)
            # Shape: (num_chunks, hidden_size)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            
            # Average the chunk embeddings to get a single vector for the whole sequence
            # Shape: (hidden_size)
            sequence_embedding = torch.mean(cls_embeddings, dim=0)
            
            # Move back to CPU and convert to standard python/numpy format for easy saving
            embeddings_dict[strain_id] = sequence_embedding.cpu()

    # 4. Save the embeddings
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"\nSaving embeddings to {output_path}...")
    torch.save(embeddings_dict, output_path)
    print(f"Success! Saved {len(embeddings_dict)} sequence embeddings.")

if __name__ == "__main__":
    from pathlib import Path
    project_root = Path(__file__).resolve().parents[2]
    
    in_json = project_root / "data" / "processed" / "module1" / "h5n1_metadata.json"
    model_directory = project_root / "models" / "module1_dnbert"
    out_file = project_root / "data" / "processed" / "module1" / "h5n1_embeddings.pt"
    
    extract_and_save_embeddings(str(in_json), str(model_directory), str(out_file))