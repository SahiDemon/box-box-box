import json, glob, numpy as np
from scipy.optimize import linprog

def degradation_sums(n):
    if n <= 0: return 0.0, 0.0
    return n*(n+1)/2.0, n*(n+1)*(2*n+1)/6.0

for file in glob.glob("data/historical_races/races_000*.json")[:1]: 
    with open(file) as f:
        data = json.load(f)
races = data[:40] 

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
        
        tire = strat["starting_tire"]
        age = 0
        pit_count = len(strat.get("pit_stops", []))
        for lap in range(1, int(r["race_config"]["total_laps"])+1):
            age += 1
            counts[tire] += 1
            free = {"SOFT": 6, "MEDIUM": 9, "HARD": 12}[tire]
            s1, s2 = degradation_sums(age - free)
            degs[tire] += s1 if age > free else 0
            degs2[tire] += s2 if age > free else 0
            if lap in pit_map:
                tire = pit_map[lap]
                age = 0
                
        x = [pit_count * pit_lane_time, pit_count]
        for c in ["SOFT", "MEDIUM", "HARD"]:
            x += [
                counts[c]*base,          
                degs[c]*base,            
                degs2[c]*base,           
                counts[c]*base*T,        
                degs[c]*base*T,          
                degs2[c]*base*T          
            ]
        
        drivers = [f"D{i:03d}" for i in range(1, 21)]
        d_arr = np.zeros(20)
        d_arr[drivers.index(p["driver_id"])] = 1.0
        
        feats[p["driver_id"]] = np.concatenate([np.array(x), d_arr])
        
    for i in range(19):
        for j in range(i+1, 20):
            X_diff.append(feats[res[i]] - feats[res[j]])

A_ub = np.array(X_diff)
b_ub = -np.ones(len(X_diff)) * 0.01  
c = np.zeros(A_ub.shape[1]) 

res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=(None, None), method='highs')
print("Status (NO EQ):", res.message)
if res.success:
    print("w0 =", res.x[0])
    w = res.x / res.x[0]
    print("Scaled w0 =", w[0])
    print("Scaled w1 =", w[1])

