"""
SEIR model implementation for epidemic simulation.
"""

import numpy as np
from scipy.integrate import odeint

def seir_model(y, t, beta, gamma, sigma, N):
    """
    SEIR model differential equations.
    
    Args:
        y: State vector [S, E, I, R]
        t: Time
        beta: Transmission rate
        gamma: Recovery rate
        sigma: Incubation rate
        N: Total population
        
    Returns:
        Derivatives [dS/dt, dE/dt, dI/dt, dR/dt]
    """
    S, E, I, R = y
    
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    
    return [dSdt, dEdt, dIdt, dRdt]

def simulate_seir(initial_conditions, t, beta, gamma, sigma, N):
    """
    Simulate SEIR model.
    
    Args:
        initial_conditions: [S0, E0, I0, R0]
        t: Time points
        beta: Transmission rate
        gamma: Recovery rate
        sigma: Incubation rate
        N: Total population
        
    Returns:
        Array of state vectors over time
    """
    solution = odeint(seir_model, initial_conditions, t, args=(beta, gamma, sigma, N))
    return solution
