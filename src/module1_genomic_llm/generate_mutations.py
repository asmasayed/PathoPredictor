"""
Module 1 - Mutation Generator (API-Ready)
Reads a cleaned JSON sequence, uses Variant Scoring (Perplexity) to rank 
biologically plausible mutations, and returns JSON for frontend consumption.
"""
import random
import json
import argparse
import sys
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForMaskedLM

#from src.module1_genomic_llm.process_uploaded_fasta import clean_uploaded_fasta
#json_string = clean_uploaded_fasta("path/to/uploaded.fasta")

def seq_to_kmers(seq: str, k: int = 6) -> str:
    """Convert a nucleotide sequence into a space-separated string of k-mers."""
    return " ".join([seq[i : i + k] for i in range(len(seq) - k + 1)])

def evaluate_sequence(model, tokenizer, sequence: str, target_index: int, device: torch.device) -> float:
    """
    Passes a windowed sequence through the model to calculate its 'Loss'.
    Centers a 500-bp window around the mutation so it doesn't get truncated.
    """
    # Create a 500 base-pair window centered around the mutation
    window_start = max(0, target_index - 250)
    window_end = min(len(sequence), target_index + 250)
    
    # Extract just that window
    windowed_seq = sequence[window_start:window_end]
    kmer_string = seq_to_kmers(windowed_seq)
    
    inputs = tokenizer(
        kmer_string,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding="max_length"
    )
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
    inputs["labels"] = inputs["input_ids"].clone()
    
    with torch.no_grad():
        outputs = model(**inputs)
        loss = outputs.loss.item()
        
    return loss

def predict_plausible_mutations(model_dir: str, sequence: str, strain_id: str = "Unknown", target_index: int = None) -> dict:
    """
    Mutates a specific base pair, ranks plausibility, and returns a structured dictionary.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForMaskedLM.from_pretrained(model_dir).to(device)
    model.eval()

    if target_index is None:
        target_index = random.randint(10, len(sequence) - 10)
        
    original_nucleotide = sequence[target_index]
    possible_nucleotides = ['A', 'T', 'C', 'G']
    
    mutations_data = []
    
    for nuc in possible_nucleotides:
        mutated_seq = sequence[:target_index] + nuc + sequence[target_index + 1:]
        loss_score = evaluate_sequence(model, tokenizer, mutated_seq, target_index, device)
        
        # Generate a predicted name format (e.g., C121G means Cytosine at 121 became Guanine)
        predicted_name = f"{original_nucleotide}{target_index}{nuc}" if nuc != original_nucleotide else f"WildType ({original_nucleotide}{target_index})"
        
        mutations_data.append({
            "predicted_name": predicted_name,
            "nucleotide": nuc,
            "loss_score": round(loss_score, 4),
            "is_original": nuc == original_nucleotide,
            "sequence_snippet": mutated_seq[:100] + "..." # Send a snippet to keep JSON lightweight
        })

    # Sort lowest loss first
    mutations_data.sort(key=lambda x: x["loss_score"])
    
    # Structure the final response payload for the frontend
    response_payload = {
        "strain_id": strain_id,
        "target_index": target_index,
        "original_nucleotide": original_nucleotide,
        "predictions": mutations_data
    }
    
    return response_payload

def process_file(json_filepath: str, model_dir: str) -> str:
    """Reads the JSON file from the cleaner and processes the sequence."""
    try:
        with open(json_filepath, 'r') as f:
            data = json.load(f)
            
        # Assuming the cleaner outputs a list of dicts. We take the first one for this tool.
        if isinstance(data, list) and len(data) > 0:
            sequence_data = data[0]
        elif isinstance(data, dict):
            sequence_data = data
        else:
            raise ValueError("Invalid JSON format. Expected a list or dictionary.")
            
        sequence = sequence_data.get("sequence")
        strain_id = sequence_data.get("strain_id", "Uploaded_Strain")
        
        if not sequence:
            raise ValueError("No 'sequence' key found in the JSON data.")
            
        # Run the predictor
        results = predict_plausible_mutations(model_dir, sequence, strain_id)
        
        # Return as a JSON string so it can be printed or sent over an API
        return json.dumps(results, indent=2)
        
    except Exception as e:
        error_response = {"error": str(e)}
        return json.dumps(error_response, indent=2)

if __name__ == "__main__":
    # Set up argument parsing so it can be called cleanly from the backend
    parser = argparse.ArgumentParser(description="Generate H5N1 Mutations from a JSON file.")
    parser.add_argument("--input", type=str, required=True, help="Path to the cleaned JSON file containing the sequence.")
    parser.add_argument("--model", type=str, required=True, help="Path to the fine-tuned DNABERT model directory.")
    
    args = parser.parse_args()
    
    # Process and print the JSON output to the terminal (which the backend will capture)
    json_output = process_file(args.input, args.model)
    print(json_output)