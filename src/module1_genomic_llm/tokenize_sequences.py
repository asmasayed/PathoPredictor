"""
Module 1 - Step 2: Tokenize cleaned sequences and build a Hugging Face Arrow Dataset.
"""
import os
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer

def seq_to_kmers(seq: str, k: int = 6) -> str:
    """Convert a nucleotide sequence into a space-separated string of k-mers."""
    return " ".join([seq[i : i + k] for i in range(len(seq) - k + 1)])

def build_hf_dataset(json_path: str, output_dir: str, model_name: str = "zhihan1996/DNA_bert_6", max_length: int = 512) -> None:
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Cleaned JSON not found at {json_path}. Run data cleaner first.")

    print(f"Loading raw dataset from {json_path}...")
    # Load the JSON directly into a Hugging Face Dataset
    raw_dataset = load_dataset("json", data_files=json_path, split="train")
    
    print(f"Loading tokenizer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_and_chunk(examples):
        # 1. Convert biological sequence to k-mer strings
        kmer_strings = [seq_to_kmers(seq) for seq in examples["sequence"]]
        
        # 2. Tokenize and automatically chunk into windows of max_length
        # stride=50 gives a 50-token overlap between chunks so motifs aren't cut blindly
        tokenized = tokenizer(
            kmer_strings,
            truncation=True,
            max_length=max_length,
            stride=50, 
            return_overflowing_tokens=True, 
            padding="max_length"
        )
        
        # We don't need the mapping that points chunks back to their parent sequence for MLM
        tokenized.pop("overflow_to_sample_mapping")
        return tokenized

    print("Tokenizing and chunking sequences (this is heavily optimized in C++ under the hood)...")
    # map() applies this to the whole dataset efficiently in batches
    # remove_columns strips out the raw text/metadata since the model only wants input_ids and attention_mask
    tokenized_dataset = raw_dataset.map(
        tokenize_and_chunk,
        batched=True,
        remove_columns=raw_dataset.column_names, 
        desc="Tokenizing & Chunking"
    )

    # Set the format strictly to PyTorch tensors so the DataLoader can consume it natively later
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    print(f"Saving Arrow dataset to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    tokenized_dataset.save_to_disk(output_dir)
    print(f"Success! Final dataset contains {len(tokenized_dataset)} token windows.")

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parents[1]
    
    in_json = project_root / "data" / "processed" / "module1" / "h5n1_metadata.json"
    out_hf_dir = project_root / "data" / "processed" / "module1" / "hf_tokenized_dataset"
    
    build_hf_dataset(str(in_json), str(out_hf_dir))