import torch

def predict_beta_adjustment(model, time_series_data):
    """Predict beta parameter adjustment from time series."""
    memory_tensor = torch.tensor([time_series_data], dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        prediction = model(memory_tensor).item()
        
    # Keep the transmission rate biologically realistic
    beta = max(0.05, min(0.6, prediction))
    return beta