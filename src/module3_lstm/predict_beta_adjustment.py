import csv
from pathlib import Path
from typing import Optional

import torch

from src.module3_lstm.model import BetaAdjustmentLSTM

_MODULE3_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _MODULE3_DIR.parent.parent


def _resolve_weights_path(model_path: Optional[str]) -> Path:
    if model_path:
        p = Path(model_path)
        if p.is_file():
            return p
    cand = _MODULE3_DIR / "lstm_weights.pth"
    if cand.is_file():
        return cand
    root = _PROJECT_ROOT / "lstm_weights.pth"
    if root.is_file():
        return root
    raise FileNotFoundError(
        f"Missing LSTM weights. Expected {_MODULE3_DIR / 'lstm_weights.pth'} "
        f"or project root lstm_weights.pth"
    )


def _resolve_mobility_csv(mobility_csv: Optional[str]) -> Path:
    if mobility_csv:
        p = Path(mobility_csv)
        if p.is_file():
            return p
        p = _MODULE3_DIR / mobility_csv
        if p.is_file():
            return p
        p = _PROJECT_ROOT / mobility_csv
        if p.is_file():
            return p
    for name in ("city_mobility_100days.csv",):
        p = _MODULE3_DIR / name
        if p.is_file():
            return p
        p = _PROJECT_ROOT / name
        if p.is_file():
            return p
    raise FileNotFoundError(
        f"Missing mobility CSV. Place city_mobility_100days.csv under {_MODULE3_DIR} or project root."
    )


def load_trained_model(model_path: Optional[str] = None):
    """Loads the pre-trained LSTM from disk (inference only; no training)."""
    path = _resolve_weights_path(model_path)
    model = BetaAdjustmentLSTM(input_size=1, hidden_size=16, num_layers=1)
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def predict_dynamic_beta(base_beta: float, mobility_csv: Optional[str] = None):
    """Reads mobility, runs the LSTM, and returns the daily beta array."""
    model = load_trained_model()
    csv_path = _resolve_mobility_csv(mobility_csv)

    mobility_data: list[float] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) > 1:
                mobility_data.append(float(row[1]))

    dynamic_betas: list[float] = []
    seq_length = 7

    for _ in range(seq_length):
        dynamic_betas.append(float(base_beta))

    with torch.no_grad():
        for i in range(len(mobility_data) - seq_length):
            seq = mobility_data[i : i + seq_length]
            input_tensor = torch.tensor([[[x] for x in seq]], dtype=torch.float32)
            multiplier = model(input_tensor).item()
            dynamic_betas.append(float(base_beta) * float(multiplier))

    return dynamic_betas


if __name__ == "__main__":
    test_betas = predict_dynamic_beta(base_beta=0.299)
    print(f"Generated {len(test_betas)} days of dynamic Beta values.")
