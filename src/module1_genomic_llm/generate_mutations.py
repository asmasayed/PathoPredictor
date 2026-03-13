"""
Module 1 - Bonus: Generate mutations using the fine-tuned Genomic LLM.
Uses the Masked Language Modeling (MLM) head to predict plausible future variants.
"""
import random
from transformers import pipeline, AutoTokenizer, AutoModelForMaskedLM

def seq_to_kmers(seq: str, k: int = 6) -> list:
    """Convert a nucleotide sequence into a list of k-mers."""
    return [seq[i : i + k] for i in range(len(seq) - k + 1)]

def generate_mutations(model_dir: str, sequence: str, num_mutations: int = 5, mask_position: int = None) -> list:
    """
    Generate mutations for a given sequence by masking a k-mer and predicting replacements.
    
    Args:
        model_dir: Path to the fine-tuned DNABERT model.
        sequence: The raw DNA sequence string.
        num_mutations: How many probable variants to return.
        mask_position: Specific base pair index to mutate (random if None).
        
    Returns:
        List of dictionaries containing the mutated sequence and the model's confidence score.
    """
    print("Loading fine-tuned model for mutation generation...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForMaskedLM.from_pretrained(model_dir)
    
    # Initialize the Hugging Face fill-mask pipeline
    unmasker = pipeline("fill-mask", model=model, tokenizer=tokenizer)

    # 1. Convert to k-mers
    kmer_list = seq_to_kmers(sequence)
    
    # 2. Pick a spot to mutate
    if mask_position is None:
        # Pick a random k-mer somewhere in the middle of the sequence
        mask_position = random.randint(10, len(kmer_list) - 10)
    
    original_kmer = kmer_list[mask_position]
    print(f"Targeting base pair index {mask_position}. Original 6-mer: {original_kmer}")

    # 3. Apply the mask
    kmer_list[mask_position] = tokenizer.mask_token
    masked_string = " ".join(kmer_list)
    
    # Ensure it's not too long for the model (truncate around the mask if necessary)
    # For a full implementation on ~1700bp, you'd chunk this. For testing, we'll assume it fits.

    # 4. Predict the mutations
    print(f"Predicting top {num_mutations} biologically probable mutations...")
    predictions = unmasker(masked_string, top_k=num_mutations)
    
    results = []
    for pred in predictions:
        predicted_kmer = pred["token_str"].replace(" ", "")
        score = pred["score"]
        
        # 5. Reconstruct the new full sequence
        # We replace the exact 6 base pairs at the mask_position
        mutated_seq = sequence[:mask_position] + predicted_kmer + sequence[mask_position + 6:]
        
        results.append({
            "predicted_kmer": predicted_kmer,
            "confidence_score": score,
            "mutated_sequence": mutated_seq
        })
        
    return results

if __name__ == "__main__":
    from pathlib import Path
    project_root = Path(__file__).resolve().parents[2]
    model_directory = project_root / "models" / "module1_dnbert"
    
    # Let's test it on a small dummy string of H5N1-like HA sequence
    test_sequence = "ATGGAGAAAATAGTGCTTCTTCTTGCAATAGTCAGTCTTGTTAAAAGTGATCAGATTTGCATTGGTTACCATGCAAACAATTCAACAGAGCAGGTTGACACAATCATGGAAAAGAACGTTACTGTTACACATGCC"
    
    print("\n--- Testing Mutation Generator ---")
    generated_variants = generate_mutations(str(model_directory), test_sequence, num_mutations=3)
    
    print("\n--- Results ---")
    for i, variant in enumerate(generated_variants, 1):
        print(f"\nVariant {i}:")
        print(f"New K-mer: {variant['predicted_kmer']} (Confidence: {variant['confidence_score']:.4f})")
        print(f"Sequence:  {variant['mutated_sequence'][:60]}...") # Printing just the first 60 chars