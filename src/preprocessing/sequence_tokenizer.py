import collections

class SequenceTokenizer:
    """Tokenizes genomic sequences using a k-mer sliding window strategy."""
    
    def __init__(self, k=3, vocab_size=None):
        self.k = k
        self.vocab_size = vocab_size
        #encode
        self.token_to_id = {"<PAD>": 0, "<UNK>": 1, "<CLS>": 2, "<SEP>": 3} #tokens for bert 
        #decode
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        
        #create k mers clusters
    def _get_kmers(self, sequence):
        """Splits a sequence into overlapping k-mers."""
        return [sequence[i:i+self.k] for i in range(len(sequence) - self.k + 1)]
    
    def fit(self, sequences):
        """Builds the vocabulary from a list of genomic sequences."""
        all_kmers = []
        for seq in sequences:
            all_kmers.extend(self._get_kmers(seq))
            
        # Count frequencies to respect the vocab_size limit
        counts = collections.Counter(all_kmers)
        most_common = counts.most_common(self.vocab_size - len(self.token_to_id)) if self.vocab_size else counts.most_common()
        
        for kmer, _ in most_common:
            if kmer not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[kmer] = idx
                self.id_to_token[idx] = kmer

    def encode(self, sequence, add_special_tokens=True):
        """Converts a DNA string into a list of integer IDs."""
        kmers = self._get_kmers(sequence)
        tokens = [self.token_to_id.get(km, 1) for km in kmers] # 1 is <UNK>
        
        if add_special_tokens:
            tokens = [2] + tokens + [3] # Adding <CLS> and <SEP>
        return tokens
    
    def decode(self, token_ids):
        """Reconstructs the k-mer sequence (note: overlaps make full string reconstruction complex)."""
        return " ".join([self.id_to_token.get(tid, "<UNK>") for tid in token_ids])