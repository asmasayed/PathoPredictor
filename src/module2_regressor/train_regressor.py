"""
Train Module 2 epidemiological parameter regressor (PyTorch MLP).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.config.config import MODULE2_REGRESSOR_CONFIG
from src.module2_regressor.io_utils import load_regressor_training_arrays, write_synthetic_regressor_csv
from src.module2_regressor.model import ParameterRegressor


def train_regressor(config: Optional[Dict] = None) -> Dict:
    cfg = config or MODULE2_REGRESSOR_CONFIG
    csv_path = Path(cfg["train_csv"])
    ckpt_path = Path(cfg["checkpoint_path"])
    meta_path = Path(cfg["meta_path"])

    if not csv_path.is_file():
        print(f"No training CSV at {csv_path}; writing synthetic demo data.")
        write_synthetic_regressor_csv(csv_path, embed_dim=int(cfg.get("synthetic_embed_dim", 16)))

    X, y, emb_cols = load_regressor_training_arrays(csv_path)

    seed = int(cfg.get("seed", 7))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)
    split = int(n * (1 - float(cfg.get("val_fraction", 0.2))))
    tr, va = idx[:split], idx[split:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train_t = torch.tensor(X[tr], dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y[tr], dtype=torch.float32, device=device)
    X_val_t = torch.tensor(X[va], dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y[va], dtype=torch.float32, device=device)

    ds = TensorDataset(X_train_t, y_train_t)
    dl = DataLoader(ds, batch_size=int(cfg["batch_size"]), shuffle=True)

    input_dim = X.shape[1]
    hidden_dim = int(cfg.get("hidden_dim", 128))
    model = ParameterRegressor(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]))
    loss_fn = nn.MSELoss()

    epochs = int(cfg.get("epochs", 80))
    best_val = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        for xb, yb in dl:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            vpred = model(X_val_t)
            vloss = float(loss_fn(vpred, y_val_t).cpu())
        if vloss < best_val:
            best_val = vloss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % max(epochs // 8, 1) == 0:
            print(f"epoch {epoch + 1}/{epochs} val_mse={vloss:.6f}")

    model.cpu()
    if best_state is not None:
        model.load_state_dict(best_state)

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, ckpt_path)

    meta = {
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "output_dim": 3,
        "embedding_columns": emb_cols,
        "motif_feature_dim": 5,
        "feature_dim": input_dim,
        "best_val_mse": best_val,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved regressor checkpoint to {ckpt_path}")
    print(f"Saved meta to {meta_path}")
    return {"best_val_mse": best_val}


if __name__ == "__main__":
    train_regressor(MODULE2_REGRESSOR_CONFIG)
