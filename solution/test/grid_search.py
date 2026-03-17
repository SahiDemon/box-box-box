import json
import numpy as np
import glob

# Load a small clean subset
races = []
for file in glob.glob('data/historical_races/races_000*.json')[:2]:
    with open(file) as f:
        races.extend(json.load(f))
        
# 2000 races
print(f"Loaded {len(races)} races.")

# We want to find:
# soft_base, med_base, hard_base
# soft_cliff, med_cliff, hard_cliff
# soft_deg, med_deg, hard_deg
# soft_temp_deg, med_temp_deg, hard_temp_deg

# For each lap:
# time = base_lap_time + pit
#        + base[tire] 
#        + (deg[tire] + temp_deg[tire]*T) * max(0, age - cliff[tire])

from scipy.optimize import minimize

X_data = [] # We'll store: (tire, age, T, pit, base_lap_time)
            # Actually, to make it fast, we can aggregate laps for each driver
def run_search():
    pass
