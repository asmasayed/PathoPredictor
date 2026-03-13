"""
Module 1 - Step 1: Clean raw H5N1 HA data into a structured JSON file.
"""
import os
import json
import re
from pathlib import Path
from Bio import SeqIO

# Regex to capture the strain name, stopping before the (H5N1) subtype label
# Example target: (A/Antarctic Fur Seal/Antarctica/029560:SG14/2024(H5N1))
STRAIN_BLOCK_REGEX = re.compile(r'\((A/[^\(]+)')

def extract_metadata(header: str) -> dict:
    """
    Extract structured metadata from NCBI Influenza header.
    Returns dictionary or None if parsing fails.
    """
    match = STRAIN_BLOCK_REGEX.search(header)
    if not match:
        return None

    # Extracted: A/Antarctic Fur Seal/Antarctica/029560:SG14/2024
    strain_full = match.group(1).strip()
    parts = strain_full.split('/')

    # We expect at least A / Host / Location / Strain / Year
    if len(parts) < 4:
        return None

    try:
        year = int(parts[-1])
    except ValueError:
        return None

    return {
        "strain_id": strain_full,
        "host": parts[1] if len(parts) > 4 else "Unknown",
        "region": parts[2] if len(parts) > 4 else parts[1],
        "year": year,
        "segment": "HA"
    }

def clean_and_structure(input_fasta: str, output_fasta: str, output_json: str) -> None:
    stats = {
        'total_processed': 0,
        'accepted': 0,
        'rejected_length': 0,
        'rejected_ambiguous': 0,
        'rejected_duplicate': 0,
        'rejected_metadata': 0
    }

    seen_sequences = set()
    structured_records = []

    os.makedirs(os.path.dirname(output_fasta), exist_ok=True)

    with open(output_fasta, "w") as out_f:
        for record in SeqIO.parse(input_fasta, "fasta"):
            stats['total_processed'] += 1

            seq = str(record.seq).upper().replace("-", "")
            seq_length = len(seq)

            # Length filter (HA genes are typically ~1700bp)
            if not (1600 <= seq_length <= 1900):
                stats['rejected_length'] += 1
                continue

            # Ambiguous filter (Reject if too many 'N's)
            if (seq.count("N") / seq_length) > 0.01:
                stats['rejected_ambiguous'] += 1
                continue

            # Remove non ATCGN characters
            seq = ''.join([b for b in seq if b in ['A', 'T', 'C', 'G', 'N']])

            # Deduplicate exact sequence matches
            if seq in seen_sequences:
                stats['rejected_duplicate'] += 1
                continue

            metadata = extract_metadata(record.description)
            if not metadata:
                stats['rejected_metadata'] += 1
                continue

            seen_sequences.add(seq)

            # Write normalized FASTA
            normalized_header = f">{metadata['strain_id']}|{metadata['year']}"
            out_f.write(f"{normalized_header}\n{seq}\n")

            # Build structured record for JSON
            metadata["sequence"] = seq
            structured_records.append(metadata)

            stats['accepted'] += 1

    # Save structured JSON for the LLM Tokenizer
    with open(output_json, "w") as jf:
        json.dump(structured_records, jf, indent=2)

    print("\n===== CLEANING + METADATA REPORT =====")
    for k, v in stats.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    # Test block for running this script directly
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    
    in_fasta = project_root / "data" / "raw" / "genomic_fasta" / "sequences.fasta"
    out_fasta = project_root / "data" / "processed" / "module1" / "cleaned_h5n1_ha.fasta"
    out_json = project_root / "data" / "processed" / "module1" / "h5n1_metadata.json"
    
    if in_fasta.exists():
        clean_and_structure(str(in_fasta), str(out_fasta), str(out_json))
    else:
        print(f"File not found: {in_fasta}. Please add your data to this directory.")