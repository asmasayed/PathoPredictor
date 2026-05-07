import os
import sys
# --- PATH FIX ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.insert(0, project_root)
# ----------------

import torch
import numpy as np
import matplotlib.pyplot as plt
from src.module3_lstm.model import LSTMModel
from src.module3_lstm.predict_beta_adjustment import predict_beta_adjustment
from src.simulation.parameters import SEIRParameters
from src.simulation.seir_model import simulate_seir
from src.config.config import SEIR_DEFAULT_PARAMS

# 1. ADD REGION AND POPULATION VARIABLES
def run_simulation(
    recent_memory,
    scaler,
    last_real_cases,
    region="us",
    N=600000,
    beta=None,
    gamma=None,
    sigma=None,
    sim_days=60,
):
    """
    Hybrid SEIR: LSTM proposes beta each day after the first timestep.
    If ``beta`` (from Module 2) is supplied, timestep 0 uses that beta together with
    ``gamma``/``sigma``; later days keep Module 2 γ/σ while β is updated by the LSTM.
    """
    print(f"Initializing Hybrid SEIR Engine for {region.upper()}...")
    
    model = LSTMModel()
    
    # 2. DYNAMIC BRAIN LOADING: Load the specific AI trained for this climate
    brain_path = f"models/module3_lstm/lstm_brain_{region}.pth"
    if not os.path.exists(brain_path):
        raise FileNotFoundError(f"Could not find AI brain: {brain_path}. Did you train it?")
    try:
        state = torch.load(brain_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(brain_path, map_location="cpu")
    model.load_state_dict(state)
    
    params = SEIRParameters(
        beta=SEIR_DEFAULT_PARAMS.get("beta", 0.3) if beta is None else beta,
        gamma=SEIR_DEFAULT_PARAMS.get("gamma", 0.1) if gamma is None else gamma,
        sigma=SEIR_DEFAULT_PARAMS.get("sigma", 0.2) if sigma is None else sigma,
    )
    
    module2_beta_supplied = beta is not None
    
    I0 = max(5, last_real_cases)
    E0, R0 = I0 * 2, 0
    S0 = N - I0 - E0 - R0
    current_state = [S0, E0, I0, R0]
    
    hist_S, hist_E, hist_I, hist_R = [S0], [E0], [I0], [R0]
    
    for day in range(int(sim_days)):
        # Timestep 0 can keep Module-2-supplied beta; afterward LSTM drives beta.
        if (not module2_beta_supplied) or day > 0:
            pred_beta = predict_beta_adjustment(model, recent_memory)
            params.update_beta(pred_beta)

        t_span = np.array([0, 1])
        # The Calculus Engine calculates exactly how many animals get sick/recover
        result = simulate_seir(current_state, t_span, params.beta, params.gamma, params.sigma, N)
        current_state = result[1]
        
        hist_S.append(int(current_state[0]))
        hist_E.append(int(current_state[1]))
        hist_I.append(int(current_state[2]))
        hist_R.append(int(current_state[3]))

        # Update the memory tensor for the next day's prediction
        new_infection_scaled = scaler.transform([[current_state[2]]])[0].tolist()
        recent_memory.pop(0)
        recent_memory.append(new_infection_scaled)

    # 3. FREE THE DATA: Instead of trapping the data in plt.show(), we return it!
    # (We can still print a success message)
    print(f"✅ {int(sim_days)}-Day Forecast calculated for {region.upper()}.")
    
    return {
        "S": hist_S,
        "E": hist_E,
        "I": hist_I,
        "R": hist_R
    }

# Safe test block if you run the file directly
if __name__ == "__main__":
    print("This is the math engine module. It should be called from run_pipeline.py.")