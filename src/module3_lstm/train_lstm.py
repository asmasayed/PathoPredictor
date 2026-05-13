import torch
import torch.nn as nn
import torch.optim as optim
import csv
from pathlib import Path

# Import the brain we just built
from model import BetaAdjustmentLSTM

_MODULE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _MODULE_DIR.parent.parent


def load_data(filepath="city_mobility_100days.csv"):
    """Reads the CSV file and extracts just the mobility numbers."""
    mobility_data = []
    p = Path(filepath)
    if not p.is_file():
        p = _MODULE_DIR / "city_mobility_100days.csv"
    if not p.is_file():
        p = _PROJECT_ROOT / "city_mobility_100days.csv"
    if not p.is_file():
        raise FileNotFoundError(f"Mobility CSV not found (tried {filepath}, module dir, project root).")

    with open(p, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader) # Skip the header row
        for row in reader:
            mobility_data.append(float(row[1]))
    return mobility_data

def train_model():
    print("Loading mobility data...")
    data = load_data("city_mobility_100days.csv")
    
    # 1. Prepare the Data (The 7-Day Window)
    # We teach the AI to look at 7 days of history to predict the current day's multiplier
    seq_length = 7
    inputs = []
    targets = []
    
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        target = data[i+seq_length] 
        # Convert arrays to the specific matrix format PyTorch requires
        inputs.append([[x] for x in seq])
        targets.append([target])
        
    inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
    targets_tensor = torch.tensor(targets, dtype=torch.float32)
    
    # 2. Initialize the Model
    model = BetaAdjustmentLSTM(input_size=1, hidden_size=16, num_layers=1)
    
    # Mean Squared Error (calculates how wrong the AI's guess is)
    criterion = nn.MSELoss()
    # Adam Optimizer (the algorithm that tweaks the neurons to fix the error)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # 3. The Training Loop
    epochs = 150
    print("Starting neural network training...")
    
    for epoch in range(epochs):
        optimizer.zero_grad()           # Clear old memory
        outputs = model(inputs_tensor)  # AI makes a guess
        loss = criterion(outputs, targets_tensor) # Grade the guess
        loss.backward()                 # Figure out what went wrong
        optimizer.step()                # Update the brain weights
        
        # Print progress every 30 loops
        if (epoch + 1) % 30 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}] | AI Error (Loss): {loss.item():.4f}')
            
    # 4. Save next to this script (works from any cwd)
    save_path = _MODULE_DIR / "lstm_weights.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Training complete. Weights saved to: {save_path}")

if __name__ == "__main__":
    train_model()