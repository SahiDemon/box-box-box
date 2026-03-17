import json, glob, numpy as np
from scipy.optimize import linprog

def load_races(num_files=1):
    races = []
    for file in glob.glob("data/historical_races/races_000*.json")[:num_files]: 
        with open(file) as f:
            races.extend(json.load(f))
    return races

races = load_races(1)[:40] 

# evaluate feasibility of just generic histograms!
X_diff = []
for r in races:
    res = r["finishing_positions"]
    base = r["race_config"]["base_lap_time"]
    T = r["race_config"]["track_temp"] - 30.0 
    pit_lane_time = r["race_config"]["pit_lane_time"]
    total_laps = int(r["race_config"]["total_laps"])
    
    feats = {}
    for pos_key, strat in r["strategies"].items():
        pit_map = {int(stop["lap"]): stop["to_tire"] for stop in strat.get("pit_stops", [])}
        
        # let's just make a huge feature vector
        # for each tire, the number of laps it was at age 1, 2, 3 ... 80
        hist_S = np.zeros(80)
        hist_M = np.zeros(80)
        hist_H = np.zeros(80)
        
        tire = strat["starting_tire"]
        age = 0
        pit_count = len(strat.get("pit_stops", []))
        for lap in range(1, total_laps+1):
            age += 1
            if tire == "SOFT": hist_S[age] += 1
            elif tire == "MEDIUM": hist_M[age] += 1
            elif tire == "HARD": hist_H[age] += 1
            if lap in pit_map:
                tire = pit_map[lap]
                age = 0
                
        x = [pit_count * pit_lane_time, pit_count]
        
        # For each age, we have an effect (additive). Also maybe a base multiplier. Also temp multiplier.
        # But wait, if we just use the histograms directly, it should perfectly capture ANY age-based degradation!
        x.extend(hist_S)
        x.extend(hist_M)
        x.extend(hist_H)
        
        x.extend(hist_S * base)
        x.extend(hist_M * base)
        x.extend(hist_H * base)
        
        x.extend(hist_S * T)
        x.extend(hist_M * T)
        x.extend(hist_H * T)
        
        x.extend(hist_S * base * T)
        x.extend(hist_M * base * T)
        x.extend(hist_H * base * T)
        
        # Drivers bias
        driver_id = strat["driver_id"]
        drivers = [f"D{i:03d}" for i in range(1, 21)]
        d_arr = np.zeros(20)
        d_arr[drivers.index(driver_id)] = 1.0
        
        feats[driver_id] = np.concatenate([np.array(x), d_arr])
        
    for i in range(19):
        for j in range(i+1, 20):
            X_diff.append(feats[res[i]] - feats[res[j]])

A_ub = np.array(X_diff)
b_ub = -np.ones(len(X_diff)) * 0.01  
c = np.zeros(A_ub.shape[1]) 

res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=(None, None), method='highs')
print("Status (Histograms):", res.message)
