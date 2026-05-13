import numpy as np
from scipy.integrate import odeint

def seir_equations(y, t, N, beta, alpha, gamma):
    """Standard SEIR differential equations."""
    S, E, I, R = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - alpha * E
    dIdt = alpha * E - gamma * I
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt

def simulate_seir(N, beta, alpha, gamma, days=100, initial_infected=1):
    """
    The function Module 2 and the API are looking for.
    It runs a standard simulation and returns the results.
    """
    # Initial conditions
    E0 = initial_infected
    I0 = 0
    R0 = 0
    S0 = N - E0 - I0 - R0
    y0 = [S0, E0, I0, R0]
    
    # Time points
    t = np.linspace(0, days, days)
    
    # Integrate the equations
    ret = odeint(seir_equations, y0, t, args=(N, beta, alpha, gamma))
    S, E, I, R = ret.T
    
    return {
        "days": t.tolist(),
        "susceptible": S.tolist(),
        "exposed": E.tolist(),
        "infected": I.tolist(),
        "recovered": R.tolist()
    }

if __name__ == "__main__":
    # Quick test to ensure the file works standalone
    res = simulate_seir(10000, 0.3, 0.2, 0.1)
    print("Standard SEIR Engine: Ready")