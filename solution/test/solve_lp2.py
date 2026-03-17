import json, glob, numpy as np
from scipy.optimize import linprog

def load_races(num_files=1):
    races = []
    for file in glob.glob("data/historical_races/races_000*.json")[:num_files]: 
        with open(file) as f:
            data = json.load(f)
        for r in data:
            races.append(r)
    return races

races = load_races(1)[:40] 

X_diff = []
for r in races:
    res = r["finishing_positions"]
    base = r["race_config"]["base_lap_time"]
    T = r["race_config"]["track_temp"] - 30.0 
    pit_lane_time = r["race_config"]["pit_lane_time"]
    
    feats = {}
    for p in r["strategies"].values():
        strat = p
        pit_map = {}
        for stop in strat.get("pit_stops", []):
            pit_map[int(stop["lap"])] = stop["to_tire"]
            
        counts = {"SOFT": 0.0, "MEDIUM": 0.0, "HARD": 0.0}
        degs = {"SOFT": 0.0, "MEDIUM": 0.0, "HARD": 0.0}
        degs2 = {"SOFT": 0.0, "MEDIUM": 0.0, "HARD": 0.0}
        degs3 = {"SOFT": 0.0, "MEDIUM": 0.0, "HARD": 0.0}
        
        tire = strat["starting_tire"]
        age = 0
        pit_count = len(strat.get("pit_stops", []))
        for lap in range(1, int(r["race_config"]["total_laps"])+1):
            age += 1
            counts[tire] += 1
            degs[tire] += age
            degs2[tire] += age**2
            degs3[tire] += age**3
            if lap in pit_map:
                tire = pit_map[lap]
                age = 0
                
        x = [pit_count * pit_lane_time, pit_count]
        for c in ["SOFT", "MEDIUM", "HARD"]:
            x += [
                counts[c]*base,          
                degs[c]*base,            
                degs2[c]*base,           
                degs3[c]*base,           
                counts[c]*base*T,        
                degs[c]*base*T,          
                degs2[c]*base*T,
                degs3[c]*base*T          
            ]
        
        drivers = [f"D{i:03d}" for i in range(1, 21)]
        d_arr = np.zeros(20)
        d_arr[drivers.index(p["driver_id"])] = 1.0
        
        feats[p["driver_id"]] = np.concatenate([np.array(x), d_arr])
        
    for i in range(19):
        for j in range(i+1, 20):
            X_diff.append(feats[res[i]] - feats[res[j]])

A_ub = np.array(X_diff)
b_ub = -np.ones(len(X_diff)) * 0.1  # margin
A_eq = np.zeros((1, A_ub.shape[1]))
A_eq[0, 0] = 1.0
b_eq = np.array([1.0])
c = np.ones(A_ub.shape[1]) 

res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(None, None), method='highs')
if res.success:
    print("Success! No free laps needed.")
else:
    print("Failed without free laps too.")

