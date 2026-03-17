import json, glob, numpy as np
from sklearn.svm import LinearSVC

# 1. Feature extraction
def degradation_sums(n):
    if n <= 0: return 0.0, 0.0
    return n*(n+1)/2.0, n*(n+1)*(2*n+1)/6.0

def extract(race, strat):
    base = race["base_lap_time"]
    T = race["track_temp"] - 30.0 # arbitrary centering
    
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
        degs[tire] += age
        degs2[tire] += age*age
        if lap in pit_map:
            tire = pit_map[lap]
            age = 0
            
    x = [pit_count * pit_lane_time, pit_count]
    for c in ["SOFT", "MEDIUM", "HARD"]:
        x += [
            counts[c]*base,          # C_mul
            degs[c]*base,            # linear
            degs2[c]*base,           # quadratic
            counts[c]*base*T,        # C_temp
            degs[c]*base*T,          # linear_temp
            degs2[c]*base*T          # quad_temp
        ]
    return np.array(x)

X_diff, Y = [], []
np.random.seed(42)
for file in glob.glob("data/historical_races/races_00*.json")[:5]: # just sample from 5 files
    with open(file) as f:
        data = json.load(f)
    for r in data:
        res = r["finishing_positions"]
        feats = {p["driver_id"]: extract(r["race_config"], p) for p in r["strategies"].values()}
        
        # Include driver identity as a one-hot feature to catch bias
        drivers = [f"D{i:03d}" for i in range(1, 21)]
        def driver_feat(d):
            arr = np.zeros(20)
            arr[drivers.index(d)] = 1.0
            return arr
            
        for i in range(15):
            for j in range(i+1, 20):
                d1 = driver_feat(res[i])
                d2 = driver_feat(res[j])
                
                f1 = np.concatenate([feats[res[i]], d1])
                f2 = np.concatenate([feats[res[j]], d2])
                
                if np.random.random() < 0.5:
                    X_diff.append(f1 - f2)
                    Y.append(-1) # res[i] is faster, so time_i - time_j < 0
                else:
                    X_diff.append(f2 - f1)
                    Y.append(1)  # res[j] is slower, so time_j - time_i > 0

X_diff = np.array(X_diff)
Y = np.array(Y)

print("Training LinearSVC...")
svc = LinearSVC(C=100.0, dual=False, fit_intercept=False, max_iter=20000)
svc.fit(X_diff, Y)
sc = svc.score(X_diff, Y)
print(f"Accuracy: {sc*100:.2f}%")

print("Weights:")
for i, w in enumerate(svc.coef_[0]):
    if i < 20: 
        print(f"w_{i}: {w:.6f}")
print("Driver biases:")
for i, w in enumerate(svc.coef_[0][20:]):
    print(f"D{i+1:03d}: {w:.6f}")

