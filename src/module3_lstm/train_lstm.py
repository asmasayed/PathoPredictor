import os
import glob
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from src.module3_lstm.model import LSTMModel

def train_lstm(config):
    print("Loading Time-Series Dataset...")
    # It looks in your friend's specified data folder
    csv_files = glob.glob("data/raw/time_series/*.csv")
    
    if len(csv_files) == 0:
        raise FileNotFoundError("No CSV file found in data/raw/time_series/! Please add your dataset.")
        
    df = pd.read_csv(csv_files[0])
    
    # Assuming your dataset has columns named 'Date' and 'Cases' (or similar). 
    # Update these string names if your CSV uses different column headers!
    date_col = df.columns[0] # Grabs the first column as Date
    case_col = df.columns[1] # Grabs the second column as Cases
    
    df['Date'] = pd.to_datetime(df[date_col])
    daily_cases = df.groupby('Date')[case_col].sum().reset_index()
    
    # Fill in missing days with 0 to make the timeline continuous
    idx = pd.date_range(daily_cases['Date'].min(), daily_cases['Date'].max())
    daily_cases = daily_cases.set_index('Date').reindex(idx, fill_value=0).reset_index()
    
    scaler = MinMaxScaler(feature_range=(0.01, 1))
    scaled_data = scaler.fit_transform(daily_cases[case_col].values.reshape(-1, 1))

    SEQ_LENGTH = config.get("sequence_length", 5) 
    
    X, y = [], []
    for i in range(len(scaled_data) - SEQ_LENGTH):
        X.append(scaled_data[i : i + SEQ_LENGTH])
        y.append(scaled_data[i + SEQ_LENGTH])
        
    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.float32)

    model = LSTMModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("learning_rate", 0.001))
    criterion = nn.MSELoss()

    print("Training AI Brain... Please wait.")
    for epoch in range(150): 
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()

    # Save exactly where your friend's setup instructions requested
    os.makedirs("models/module3_lstm", exist_ok=True)
    torch.save(model.state_dict(), "models/module3_lstm/lstm_brain.pth")
    print("LSTM Training Complete! Brain saved.")
    
    recent_memory = scaled_data[-SEQ_LENGTH:].tolist()
    last_real_cases = int(daily_cases[case_col].iloc[-1])
    return recent_memory, scaler, last_real_cases

if __name__ == "__main__":
    from src.config.config import MODULE3_CONFIG
    train_lstm(MODULE3_CONFIG)
