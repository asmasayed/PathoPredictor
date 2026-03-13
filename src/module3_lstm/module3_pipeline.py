import os
import json

from src.module3_lstm.train_lstm import train_lstm
from src.simulation.seir_sim import run_simulation
from src.config.config import MODULE3_CONFIG


def run_module3_seir():
    """
    Runs the full Module 3 + SEIR simulation pipeline.
    """

    regions = {
        "us": {"N": 600000},
        "africa": {"N": 800000},
        "asia": {"N": 1000000}
    }

    global_forecasts = {}

    for region, params in regions.items():

        print("\n==========================================")
        print(f"PROCESSING REGION: {region.upper()}")
        print("==========================================")

        print("\n[1/2] Training Neural Network...")
        recent_memory, scaler, last_real_cases = train_lstm(
            MODULE3_CONFIG,
            region=region
        )

        print("\n[2/2] Running Differential SEIR Calculus...")
        seir_results = run_simulation(
            recent_memory=recent_memory,
            scaler=scaler,
            last_real_cases=last_real_cases,
            region=region,
            N=params["N"]
        )

        global_forecasts[region.upper()] = seir_results

    print("\nSaving master global forecast data...")

    os.makedirs("data/processed", exist_ok=True)

    with open("data/processed/global_seir_forecasts.json", "w") as f:
        json.dump(global_forecasts, f)

    print("Module 3 + SEIR simulation completed!")

    return global_forecasts