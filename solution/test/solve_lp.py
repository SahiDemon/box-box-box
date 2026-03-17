import json, glob, numpy as np
from scipy.optimize import linprog

def degradation_sums(n):
    if n <= 0: return 0.0, 0.0
    return n*(n+1)/2.0, n*(n+1)*(2*n+1)/6.0

def extract(race, strat):
    base = race["base_lap_time"]
    T = race["track_temp"] - 30.0 
    pit_lane_time = race["pit_lane_time"]
    
    pit_map = {}
    for stop in strat.get("pit_stops", []):
        pit_map[int(stop["lap"])] = stop["to_tire"]
        
    counts = {"SOFT": 0.0, "MEDIUM": 0.0, "HARD": 0.0}
    degs = {"SOFT": 0.0, "MEDIUM": 0.0, "HARD": 0.0}
    degs2 = {"SOFT": 0.0, "MEDIUM": 0.0, "HARD": 0.0}
    
    tire = strat["starting_tire"]
    age = 0
    pit_count = len(strat.get("pit_stops", []))
    for lap in range(1, int(race["total_laps"])+1):
        age += 1
        counts[tire] += 1
        
        # Guessed free laps => We know 6, 9, 12 from standard F1 mechanics if those were the best.
        # But wait, what if they are exactly 5, 8, 11? 
        # Let's parameterize it. I will hardcode for now 6, 9, 12.
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
    return np.array(x)

races = []
for file in glob.glob("data/historical_races/races_000*.json")[:1]: 
    with open(file) as f:
        data = json.load(f)
        races.extend(data[:20]) # Just 20 races! 20*190=3800 constraints, highly solvable.

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
            # res[i] is FASTER than res[j]. So T(i) < T(j) ==> T(i) - T(j) < 0
            # w^T (x_i - x_j) <= -1.0  (we want a positive margin)
            X_diff.append(feats[res[i]] - feats[res[j]])

A_ub = np.array(X_diff)
b_ub = -np.ones(len(X_diff)) * 0.1  # w^T X <= -0.1

# We know w[0] (pit_lane_time multiplier) = 1.0 exactly.
# So we can enforce this mechanically.
A_eq = np.zeros((1, A_ub.shape[1]))
A_eq[0, 0] = 1.0
b_eq = np.array([1.0])

c = np.zeros(A_ub.shape[1]) 
# we don't care about the objective, just feasibility. (Or we can minimize sum of w to keep them small)

print("Solving LP with", A_ub.shape[0], "constraints...")
res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(None, None), method='highs')

if res.success:
    print("Feasible!")
    w_scaled = res.x
    feat_names = ["pit_lane_time", "pit_count"]
    for c in ["SOFT", "MEDIUM", "HARD"]:
        feat_names += [f"lap_mult_{c}", f"deg_lin_{c}", f"deg_quad_{c}", f"temp_lap_{c}", f"temp_lin_{c}", f"temp_quad_{c}"]

    print("Scaled Weights:")
    for i, name in enumerate(feat_names):
        print(f"{name:20s}: {w_scaled[i]:.6f}")

    print("\nDriver Biases (in seconds):")
    for i in range(20):
        print(f"D{i+1:03d}: {w_scaled[20+i]:.6f}")
else:
    print("Infeasible! Features might be wrong or free laps are wrong.")

