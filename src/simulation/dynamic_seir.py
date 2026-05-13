import math
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.integrate import odeint

# Tell Python where to find your AI Module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.module3_lstm.predict_beta_adjustment import predict_dynamic_beta


def seir_derivatives(y, t, N, beta, alpha, gamma):
    """The core ODEs for the SEIR model."""
    S, E, I, R = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - alpha * E
    dIdt = alpha * E - gamma * I
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt


def _initial_exposed_from_risk(risk_score_percent: float, N: int) -> int:
    """
    Larger host-adaptation risk => larger initial spillover seed.
    alpha/beta/gamma stay fixed; only initial E0 changes.
    """
    n = float(N)
    scale = max(5.0, min(n ** 0.5 / 25.0, n / 100.0))
    raw = 1.0 + (float(risk_score_percent) / 100.0) * scale
    e0 = int(round(raw))
    return max(1, min(e0, max(1, N - 4)))


def run_dynamic_seir(
    N=10000,
    base_beta=0.2993,
    alpha=0.2008,
    gamma=0.1003,
    initial_infected=1,
    risk_score_percent: Optional[float] = None,
):
    """Runs the SEIR simulation using the AI's daily adjusted transmission rates."""
    if risk_score_percent is not None:
        initial_infected = _initial_exposed_from_risk(risk_score_percent, int(N))

    dynamic_betas = predict_dynamic_beta(base_beta)
    total_days = len(dynamic_betas)

    E0 = initial_infected
    I0 = 0
    R0 = 0
    S0 = N - E0 - I0 - R0
    y0 = [S0, E0, I0, R0]

    results = {"days": [], "susceptible": [], "exposed": [], "infected": [], "recovered": []}

    for day in range(total_days):
        today_beta = dynamic_betas[day]
        t = [0, 1]

        solution = odeint(seir_derivatives, y0, t, args=(N, today_beta, alpha, gamma))

        y0 = solution[1]

        results["days"].append(day)
        results["susceptible"].append(float(y0[0]))
        results["exposed"].append(float(y0[1]))
        results["infected"].append(float(y0[2]))
        results["recovered"].append(float(y0[3]))

    results["dynamic_betas"] = [float(b) for b in dynamic_betas]
    return results


def _milestones_from_series(
    S: List[float],
    E: List[float],
    I: List[float],
    R: List[float],
    base_beta: float,
    gamma: float,
    r_initial: float,
) -> Dict[str, Any]:
    i_arr = np.asarray(I, dtype=float)
    peak_infection_day = int(np.argmax(i_arr))
    max_concurrent_infections = int(math.ceil(float(np.max(i_arr))))

    below = np.where(i_arr < 1.0)[0]
    if below.size:
        total_outbreak_duration = int(below[0])
    else:
        total_outbreak_duration = len(I) - 1

    r_final = float(R[-1]) if R else 0.0
    total_infected = max(0.0, r_final - float(r_initial))

    return {
        "r_zero": float(base_beta / gamma) if gamma else 0.0,
        "peak_infection_day": peak_infection_day,
        "max_concurrent_infections": max_concurrent_infections,
        "total_outbreak_duration": total_outbreak_duration,
        "total_infected": round(total_infected, 4),
    }


def build_module3_seir_payload(
    *,
    N: int,
    base_beta: float,
    alpha: float,
    gamma: float,
    risk_score_percent: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Run ``run_dynamic_seir`` and return a JSON-serializable dashboard payload.
    """
    initial_r = 0.0
    traj = run_dynamic_seir(
        N=int(N),
        base_beta=float(base_beta),
        alpha=float(alpha),
        gamma=float(gamma),
        risk_score_percent=risk_score_percent,
    )

    S = traj["susceptible"]
    E = traj["exposed"]
    I = traj["infected"]
    R = traj["recovered"]
    betas = traj["dynamic_betas"]

    milestones = _milestones_from_series(S, E, I, R, float(base_beta), float(gamma), initial_r)

    return {
        "time_series": {
            "days": [int(d) for d in traj["days"]],
            "susceptible": S,
            "exposed": E,
            "infected": I,
            "recovered": R,
            "dynamic_betas": betas,
        },
        "milestones": milestones,
        "parameters": {
            "population_n": int(N),
            "base_beta": float(base_beta),
            "alpha": float(alpha),
            "gamma": float(gamma),
            "sigma": float(alpha),
            "risk_score_percent": float(risk_score_percent) if risk_score_percent is not None else None,
        },
    }


if __name__ == "__main__":
    print("Initializing AI-Driven Epidemiological Simulation...")
    payload = build_module3_seir_payload(
        N=10000,
        base_beta=0.2993,
        alpha=0.2008,
        gamma=0.1003,
        risk_score_percent=13.5,
    )
    print(payload["milestones"])
