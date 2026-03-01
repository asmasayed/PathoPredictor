import torch
import numpy as np
import matplotlib.pyplot as plt
from src.module3_lstm.model import LSTMModel
from src.module3_lstm.predict_beta_adjustment import predict_beta_adjustment
from src.simulation.parameters import SEIRParameters
from src.simulation.seir_model import simulate_seir
from src.config.config import SEIR_DEFAULT_PARAMS

def run_simulation(recent_memory, scaler, last_real_cases):
    print("Initializing Hybrid SEIR Engine...")
    
    model = LSTMModel()
    model.load_state_dict(torch.load("models/module3_lstm/lstm_brain.pth", weights_only=True))
    
    params = SEIRParameters(
        beta=SEIR_DEFAULT_PARAMS.get("beta", 0.3),
        gamma=SEIR_DEFAULT_PARAMS.get("gamma", 0.1),
        sigma=SEIR_DEFAULT_PARAMS.get("sigma", 0.2)
    )
    
    N = 600000 
    I0 = max(5, last_real_cases)
    E0, R0 = I0 * 2, 0
    S0 = N - I0 - E0 - R0
    current_state = [S0, E0, I0, R0]
    
    hist_S, hist_E, hist_I, hist_R = [S0], [E0], [I0], [R0]
    SIM_DAYS = 60
    
    for day in range(SIM_DAYS):
        pred_beta = predict_beta_adjustment(model, recent_memory)
        params.update_beta(pred_beta)
        
        t_span = np.array([0, 1])
        result = simulate_seir(current_state, t_span, params.beta, params.gamma, params.sigma, N)
        current_state = result[1]
        
        hist_S.append(current_state[0])
        hist_E.append(current_state[1])
        hist_I.append(current_state[2])
        hist_R.append(current_state[3])

        new_infection_scaled = scaler.transform([[current_state[2]]])[0].tolist()
        recent_memory.pop(0)
        recent_memory.append(new_infection_scaled)

    plt.figure(figsize=(12, 6))
    plt.plot(hist_S, label='Susceptible', color='blue', linestyle='--')
    plt.plot(hist_E, label='Exposed (Incubating)', color='orange')
    plt.plot(hist_I, label='Infectious (Spreading)', color='red', linewidth=3)
    plt.plot(hist_R, label='Recovered', color='green')
    plt.title("PathoPredictor: AI-Augmented Epidemic Forecast", fontsize=14, fontweight='bold')
    plt.xlabel("Days Forecasted")
    plt.ylabel("Population Count")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()