"""
Classic SEIR trajectory using epidemiological rates from Module 2.

Uses only `src.simulation.seir_model` (ODE integrator). Does not import or call
`module3_lstm`; the hybrid LSTM+SEIR path remains separate unless you extend it elsewhere.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.config.config import MODULE3_CONFIG
from src.simulation.seir_model import simulate_seir


def bootstrap_lstm_context(region: str) -> Tuple[List, Any, int]:
    """Match `train_lstm` scaling so inference matches trained brains."""
    project_root = Path(__file__).resolve().parents[2]
    target_file = project_root / "data" / "raw" / "time_series" / f"h5n1_{region}_outbreaks.csv"
    seq_len = int(MODULE3_CONFIG.get("sequence_length", 5))
    scaler = MinMaxScaler(feature_range=(0.01, 1.0))

    if not target_file.is_file():
        scaler.fit(np.array([[1.0], [500.0]]))
        v = float(scaler.transform([[15.0]])[0, 0])
        recent_memory = [[v]] * seq_len
        return recent_memory, scaler, 15

    df = pd.read_csv(target_file)
    date_col = df.columns[0]
    case_col = df.columns[1]
    df["Date"] = pd.to_datetime(df[date_col])
    daily_cases = df.groupby("Date")[case_col].sum().reset_index()
    idx = pd.date_range(daily_cases["Date"].min(), daily_cases["Date"].max())
    daily_cases = daily_cases.set_index("Date").reindex(idx, fill_value=0).reset_index()
    scaled_data = scaler.fit_transform(daily_cases[case_col].values.reshape(-1, 1))
    recent_memory = scaled_data[-seq_len:].tolist()
    last_real_cases = int(daily_cases[case_col].iloc[-1])
    return recent_memory, scaler, last_real_cases


def run_hybrid_seir_with_module2_rates(
    beta: float,
    gamma: float,
    sigma: float,
    *,
    N: int = 600_000,
    days: int = 60,
    region: str = "us",
) -> Dict:
    """Module 3 hybrid path (`seir_sim.run_simulation`) fed with Module 2 β/γ/σ."""
    from src.simulation.seir_sim import run_simulation

    recent_memory, scaler, last_real = bootstrap_lstm_context(region)
    series = run_simulation(
        recent_memory,
        scaler,
        last_real,
        region=region,
        N=N,
        beta=float(beta),
        gamma=float(gamma),
        sigma=float(sigma),
        sim_days=int(days),
    )
    return {
        "model": "hybrid_module3_lstm_seir",
        "note": (
            "γ and σ stay from Module 2 for all days; timestep 0 uses Module 2 β, "
            "then LSTM updates β."
        ),
        "region": region,
        "population_N": N,
        "days": days,
        "parameters_seeded_from_module2": {"beta": float(beta), "gamma": float(gamma), "sigma": float(sigma)},
        **series,
    }


def project_seir_using_module2_parameters(
    beta: float,
    gamma: float,
    sigma: float,
    *,
    N: int = 600_000,
    days: int = 60,
    initial_infectious: Optional[int] = None,
) -> Dict:
    """
    Integrate SEIR once with constant beta, gamma, sigma from Module 2 regressor.
    Returns JSON-serializable time series (daily samples).
    """
    if N <= 0 or days <= 0:
        raise ValueError("N and days must be positive.")

    beta = float(max(beta, 1e-8))
    gamma = float(max(gamma, 1e-8))
    sigma = float(max(sigma, 1e-8))

    i0 = int(initial_infectious if initial_infectious is not None else max(5, round(N * 1e-5)))
    i0 = min(max(i0, 1), N // 50)
    e0 = min(i0 * 2, max(N - i0 - 1, 0))
    r_init = 0
    s0 = N - e0 - i0 - r_init
    if s0 <= 0:
        raise ValueError("Initial compartments invalid for given N and I0; lower I0.")

    y0 = [float(s0), float(e0), float(i0), float(r_init)]
    t = np.linspace(0.0, float(days), days + 1)
    sol = simulate_seir(y0, t, beta, gamma, sigma, N)

    return {
        "model": "classical_seir_constant_parameters",
        "note": "Uses Module 2 beta/gamma/sigma only; does not load module3_lstm LSTM.",
        "population_N": N,
        "days": days,
        "parameters_used": {"beta": beta, "gamma": gamma, "sigma": sigma},
        "initial_state": {"S": s0, "E": e0, "I": i0, "R": r_init},
        "S": [int(round(x)) for x in sol[:, 0]],
        "E": [int(round(x)) for x in sol[:, 1]],
        "I": [int(round(x)) for x in sol[:, 2]],
        "R": [int(round(x)) for x in sol[:, 3]],
        "time_days": list(range(days + 1)),
    }
