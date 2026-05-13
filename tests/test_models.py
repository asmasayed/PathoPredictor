"""
Tests for model architectures.
"""

import unittest
import torch
from src.module2_classifier.model import HostRiskClassifier
from src.module2_regressor.model import ParameterRegressor
from src.module3_lstm.model import LSTMModel

class TestModels(unittest.TestCase):
    """Test cases for model architectures."""
    
    def test_host_risk_classifier(self):
        """Test host risk classifier model."""
        model = HostRiskClassifier(input_dim=10, hidden_dim=128, num_classes=3)
        x = torch.randn(32, 10)
        output = model(x)
        self.assertEqual(output.shape, (32, 3))
    
    def test_parameter_regressor(self):
        """Test parameter regressor model."""
        model = ParameterRegressor(input_dim=10, hidden_dim=128, output_dim=3)
        x = torch.randn(32, 10)
        output = model(x)
        self.assertEqual(output.shape, (32, 3))
    
    def test_lstm_model(self):
        """Test LSTM model."""
        model = LSTMModel(input_dim=5, hidden_dim=64, num_layers=2, output_dim=1)
        x = torch.randn(32, 100, 5)  # batch, sequence, features
        output = model(x)
        self.assertEqual(output.shape, (32, 1))

if __name__ == '__main__':
    unittest.main()
