import torch
import torch.nn as nn

class BetaAdjustmentLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=16, num_layers=1):
        """
        The core PyTorch architecture for the PathoPredictor mobility engine.
        """
        super(BetaAdjustmentLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 1. The Memory Core (LSTM)
        # Takes in the daily mobility index and remembers the trend.
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # 2. The Output Layer (Fully Connected)
        # Compresses the complex LSTM memory down into a single number (the multiplier)
        self.fc = nn.Linear(hidden_size, 1)
        
        # 3. The Biological Safeguard (ReLU)
        # Ensures our transmission multiplier never drops below 0 
        # (because negative transmission is biologically impossible).
        self.relu = nn.ReLU()

    def forward(self, x):
        # Initialize the hidden state and cell state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Pass the data through the LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # We only care about the AI's conclusion at the very LAST time step
        out = self.fc(out[:, -1, :])
        
        # Apply the safeguard
        out = self.relu(out)
        
        return out