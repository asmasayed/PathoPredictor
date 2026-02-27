#preprocessing + artifact generation step
import json
import os
from src.preprocessing.fasta_parser import FastaParser
from src.preprocessing.sequence_tokenizer import SequenceTokenizer
from src.config.config import DATA_RAW_PATH, MODELS_PATH

def initialize_genomic_vocabulary():
    # 1. Setup paths
    fasta_path = os.path.join(DATA_RAW_PATH, "genomic_fasta/hiv1.fna")
    vocab_save_path = os.path.join(MODELS_PATH, "module1_dnbert/vocab.json")
    
    # Ensure the target directory exists(if folder "vocab_save_path" exists, do nothing,else create it)
    os.makedirs(os.path.dirname(vocab_save_path), exist_ok=True)

    print(f"--- Loading sequences from {fasta_path} ---")
    
    # 2. Parse the FASTA file
    parser = FastaParser(fasta_path)
    # Assuming fasta_parser.py has a method like parse() that returns a dict
    sequence_data = parser.parse(fasta_path) 
    sequences = list(sequence_data.values())
    
    print(f"Found {len(sequences)} sequences. Initializing Tokenizer...")
    # print(sequences)

    # 3. Fit the Tokenizer
    # Using k=3 (codons) is ideal for HIV-1 to capture protein-coding patterns
    tokenizer = SequenceTokenizer(k=3, vocab_size=1000)
    tokenizer.fit(sequences)

    # 4. Save the vocabulary
    vocab_data = {
        "k": tokenizer.k,
        "token_to_id": tokenizer.token_to_id,
        "id_to_token": tokenizer.id_to_token
    }

    with open(vocab_save_path, 'w') as f:
        json.dump(vocab_data, f, indent=4)

    print(f"--- Success! Vocabulary saved to {vocab_save_path} ---")
    print(f"Total unique tokens (k-mers): {len(tokenizer.token_to_id)}")

if __name__ == "__main__":
    initialize_genomic_vocabulary()