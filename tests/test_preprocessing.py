"""
Tests for preprocessing modules.
"""

import unittest
from src.preprocessing.fasta_parser import parse_fasta
from src.preprocessing.host_label_encoder import HostLabelEncoder

class TestPreprocessing(unittest.TestCase):
    """Test cases for preprocessing functions."""
    
    def test_host_label_encoder(self):
        """Test host label encoder."""
        encoder = HostLabelEncoder()
        labels = ['host1', 'host2', 'host3']
        encoder.fit(labels)
        encoded = encoder.transform(labels)
        decoded = encoder.inverse_transform(encoded)
        self.assertEqual(labels, list(decoded))

if __name__ == '__main__':
    unittest.main()
