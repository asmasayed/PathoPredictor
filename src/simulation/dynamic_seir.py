"""
Dynamic SEIR model with time-varying parameters.
"""

import numpy as np
from src.simulation.seir_model import seir_model
from scipy.integrate import odeint

def dynamic_seir_model(y, t, beta_func, gamma, sigma, N):
    """
    Dynamic SEIR model with time-varying beta.
    
    Args:
        y: State vector [S, E, I, R]
        t: Time
        beta_func: Function returning beta at time t
        gamma: Recovery rate
        sigma: Incubation rate
        N: Total population
        
    Returns:
        Derivatives [dS/dt, dE/dt, dI/dt, dR/dt]
    """
    S, E, I, R = y
    beta = beta_func(t)
    
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    
    return [dSdt, dEdt, dIdt, dRdt]

def simulate_dynamic_seir(initial_conditions, t, beta_func, gamma, sigma, N):
    """
    Simulate dynamic SEIR model.
    
    Args:
        initial_conditions: [S0, E0, I0, R0]
        t: Time points
        beta_func: Function returning beta at time t
        gamma: Recovery rate
        sigma: Incubation rate
        N: Total population
        
    Returns:
        Array of state vectors over time
    """
    solution = odeint(dynamic_seir_model, initial_conditions, t, 
                     args=(beta_func, gamma, sigma, N))
    return solution
