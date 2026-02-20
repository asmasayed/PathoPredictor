"""
FASTA file parser for genomic sequences.
"""

def parse_fasta(file_path):
    """
    Parse FASTA file and extract sequences.
    
    Args:
        file_path: Path to FASTA file
        
    Returns:
        Dictionary mapping sequence IDs to sequences
    """
    sequences = {}
    current_id = None
    current_seq = []
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id:
                    sequences[current_id] = ''.join(current_seq)
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
        
        if current_id:
            sequences[current_id] = ''.join(current_seq)
    
    return sequences
