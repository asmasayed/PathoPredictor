"""
Model architecture for Module 2: Parameter Regressor.
"""

import torch
import torch.nn as nn

class ParameterRegressor(nn.Module):
    """Neural network regressor for SEIR parameter prediction."""
    
    def __init__(self, input_dim, hidden_dim=128, output_dim=3):
        super(ParameterRegressor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
