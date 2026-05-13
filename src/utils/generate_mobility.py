import csv
import random
import math

def generate_lockdown_scenario(filename="city_mobility_100days.csv", days=100):
    """
    Generates a realistic 100-day mobility dataset.
    - Days 0 to 20: Normal city movement (baseline 1.0)
    - Days 21 to 60: Strict Lockdown (drops to 0.3)
    - Days 61 to 100: Gradual Reopening (slowly climbs back to 0.8)
    """
    data = [["Day", "Mobility_Index", "Lockdown_Active"]]
    
    for day in range(days):
        if day < 20:
            # Normal days: Mobility hovers around 1.0 (100%)
            mobility = 1.0 + random.uniform(-0.05, 0.05)
            lockdown = 0
            
        elif day < 60:
            # Sudden Lockdown: Mobility plummets to ~0.3 (30%)
            mobility = 0.3 + random.uniform(-0.02, 0.02)
            lockdown = 1
            
        else:
            # Gradual reopening: Logarithmic/slow climb back up
            climb = (day - 60) * 0.012
            mobility = min(0.3 + climb + random.uniform(-0.02, 0.02), 0.85)
            lockdown = 0
            
        # Round the mobility index for clean data
        data.append([day, round(mobility, 3), lockdown])
        
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)
        
    print(f"✅ Successfully generated {filename} with {days} days of realistic mobility data!")

if __name__ == "__main__":
    generate_lockdown_scenario()