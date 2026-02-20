"""
Host label encoder for encoding host metadata.
"""

from sklearn.preprocessing import LabelEncoder

class HostLabelEncoder:
    """Encodes host metadata labels."""
    
    def __init__(self):
        self.encoder = LabelEncoder()
        self.fitted = False
        
    def fit(self, labels):
        """Fit encoder on labels."""
        self.encoder.fit(labels)
        self.fitted = True
        
    def transform(self, labels):
        """Transform labels to encoded values."""
        if not self.fitted:
            raise ValueError("Encoder must be fitted before transformation")
        return self.encoder.transform(labels)
    
    def inverse_transform(self, encoded_labels):
        """Inverse transform encoded labels back to original."""
        return self.encoder.inverse_transform(encoded_labels)
