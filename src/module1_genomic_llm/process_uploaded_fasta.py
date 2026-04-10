"""
Module 1 - API Helper: Process Uploaded FASTA
Takes a raw user-uploaded FASTA file from the frontend, cleans the sequence
(removes whitespace, validates characters), and outputs a structured JSON string.
"""

import json
import argparse
from pathlib import Path


def clean_uploaded_fasta(file_path: str) -> str:
    """
    Reads a FASTA file, extracts the ID and sequence, cleans the DNA,
    and returns a JSON string for the mutation generator.
    """
    try:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Uploaded file not found at: {file_path}")

        strain_id = "Unknown_Strain"
        raw_sequence_lines = []

        # Read the file
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if not lines:
            raise ValueError("The uploaded FASTA file is empty.")

        # Parse the FASTA (Forgiving approach for user uploads)
        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                # Grab the header as the ID (remove the '>' and any leading/trailing spaces)
                strain_id = line[1:].strip()
            else:
                # It's sequence data
                raw_sequence_lines.append(line.upper())

        # Join the sequence and clean it
        raw_sequence = "".join(raw_sequence_lines)

        # Remove anything that isn't A, T, C, G, or N
        cleaned_sequence = "".join([base for base in raw_sequence if base in ["A", "T", "C", "G", "N"]])

        if len(cleaned_sequence) < 100:
            raise ValueError(
                f"Sequence is too short ({len(cleaned_sequence)} bp). Please upload a valid HA gene sequence."
            )

        # Build the JSON payload expected by generate_mutations.py
        output_data = {
            "strain_id": strain_id,
            "sequence": cleaned_sequence,
            "length": len(cleaned_sequence),
        }

        return json.dumps(output_data, indent=2)

    except Exception as e:
        error_response = {"error": str(e)}
        return json.dumps(error_response, indent=2)


if __name__ == "__main__":
    # Command line argument parsing for the backend
    parser = argparse.ArgumentParser(description="Clean a user-uploaded FASTA file.")
    parser.add_argument("--file", type=str, required=True, help="Path to the uploaded .fasta file.")

    args = parser.parse_args()

    # Process and print the JSON output
    json_output = clean_uploaded_fasta(args.file)
    print(json_output)

