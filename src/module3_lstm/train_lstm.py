import sys
import os
# --- PATH FIX ---
# This mathematically calculates the absolute path to your main PathoPredictor folder 
# and adds it to Python's brain, so it always knows where 'src' is.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.insert(0, project_root)
# ----------------

import glob
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from src.module3_lstm.model import LSTMModel

# 1. ADD REGION PARAMETER: We pass 'region' so the AI knows which climate it is studying
def train_lstm(config, region="us"):
    print(f"\n--- Loading Time-Series Dataset for {region.upper()} ---")
    
    # 2. DYNAMIC LOADING: It reads the specific file we downloaded in Phase 1
    target_file = f"data/raw/time_series/h5n1_{region}_outbreaks.csv"
    
    if not os.path.exists(target_file):
        raise FileNotFoundError(f"Could not find {target_file}! Did you run get_data.py?")
        
    df = pd.read_csv(target_file)
    
    date_col = df.columns[0] # Grabs the first column as Date
    case_col = df.columns[1] # Grabs the second column as Cases
    
    df['Date'] = pd.to_datetime(df[date_col])
    daily_cases = df.groupby('Date')[case_col].sum().reset_index()
    
    # Fill in missing days with 0 to make the timeline mathematically continuous
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

    print(f"Training AI Brain for {region.upper()}... Please wait.")
    for epoch in range(150): 
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()

    # 3. DYNAMIC SAVING: We save a separate brain for every continent so they don't overwrite!
    os.makedirs("models/module3_lstm", exist_ok=True)
    brain_path = f"models/module3_lstm/lstm_brain_{region}.pth"
    torch.save(model.state_dict(), brain_path)
    print(f"✅ LSTM Training Complete! Specialized AI Brain saved to: {brain_path}")
    
    recent_memory = scaled_data[-SEQ_LENGTH:].tolist()
    last_real_cases = int(daily_cases[case_col].iloc[-1])
    return recent_memory, scaler, last_real_cases

if __name__ == "__main__":
    from src.config.config import MODULE3_CONFIG
    
    # 4. SAFETY TEST LOOP: If you run this file directly, it trains all three brains sequentially
    for reg in ["us", "africa", "asia"]:
        try:
            train_lstm(MODULE3_CONFIG, region=reg)
        except Exception as e:
            print(f"⚠️ Skipping {reg} due to error: {e}")