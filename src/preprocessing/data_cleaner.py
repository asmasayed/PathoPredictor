from Bio import SeqIO
from pathlib import Path
import os
import json
import re

STRAIN_BLOCK_REGEX = re.compile(r'\((A/[^)]+)\)')

def extract_metadata(header):
    """
    Extract structured metadata from NCBI Influenza header.
    Returns dictionary or None if parsing fails.
    """

    match = STRAIN_BLOCK_REGEX.search(header)
    if not match:
        return None

    strain_full = match.group(1)  # A/Host/Region/Strain/Year(H5N1 removed later)

    # Remove subtype like (H5N1) if still inside
    strain_full = strain_full.split('(')[0]

    parts = strain_full.split('/')

    if len(parts) < 5:
        return None

    try:
        year = int(parts[-1])
    except:
        return None

    return {
        "strain_id": strain_full,
        "host": parts[1],
        "region": parts[2],
        "year": year,
        "segment": "HA"
    }


def clean_and_structure(input_fasta, output_fasta, output_json):

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

            # Length filter
            if not (1600 <= seq_length <= 1900):
                stats['rejected_length'] += 1
                continue

            # Ambiguous filter
            if (seq.count("N") / seq_length) > 0.01:
                stats['rejected_ambiguous'] += 1
                continue

            # Remove non ATCGN
            seq = ''.join([b for b in seq if b in ['A','T','C','G','N']])

            # Deduplicate
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

            # Build structured record
            metadata["sequence"] = seq
            structured_records.append(metadata)

            stats['accepted'] += 1

    # Save JSON for ML modules
    with open(output_json, "w") as jf:
        json.dump(structured_records, jf, indent=2)

    print("\n===== CLEANING + METADATA REPORT =====")
    for k,v in stats.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    # Get the absolute path of the script's directory
    script_dir = Path(__file__).parent
    
    # Navigate up to the project root (adjust the .parent calls if your folder structure is different)
    project_root = script_dir.parent.parent
    
    # Define robust absolute paths for input and both outputs
    input_fasta = project_root / "data" / "raw" / "genomic_fasta" / "sequences.fasta"
    output_fasta = project_root / "data" / "processed" / "module1" / "cleaned_h5n1_ha.fasta"
    output_json = project_root / "data" / "processed" / "module1" / "h5n1_metadata.json"
    
    print(f"Initiating Biopython Data Cleaning Logic on: {input_fasta}")
    
    # Execute the function, converting Path objects to strings
    clean_and_structure(str(input_fasta), str(output_fasta), str(output_json))