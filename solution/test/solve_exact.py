import json, glob, numpy as np
from sklearn.svm import LinearSVC

def degradation_sums(n):
    if n <= 0: return 0.0, 0.0
    return n*(n+1)/2.0, n*(n+1)*(2*n+1)/6.0

def load_races(num_files=1):
    races = []
    for file in glob.glob("data/historical_races/races_00*.json")[:num_files]: 
        with open(file) as f:
            data = json.load(f)
            races.extend(data)
    return races

races = load_races(4)

X_diff, Y = [], []
race_feats = []

# just extract features
i = 0
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
        
    race_feats.append((res, feats))
    
    # take subset to train
    if i < len(races) * 0.8:
        for idx in range(15):
            for j in range(idx+1, 20):
                f1 = feats[res[idx]]
                f2 = feats[res[j]]
                if np.random.random() < 0.5:
                    X_diff.append(f1 - f2)
                    Y.append(-1)
                else:
                    X_diff.append(f2 - f1)
                    Y.append(1)
    i += 1

X_diff = np.array(X_diff)
Y = np.array(Y)

print("Training LinearSVC...")
svc = LinearSVC(C=1.0, dual=False, fit_intercept=False, max_iter=20000)
svc.fit(X_diff, Y)
print(f"Train Pairwise Accuracy: {svc.score(X_diff, Y)*100:.2f}%")

w = svc.coef_[0]

exact = 0
total = 0
for res, feats in race_feats[int(len(races)*0.8):]:
    pred = []
    for d in drivers:
        pred.append((np.dot(w, feats[d]), d))
    pred.sort()
    pred_res = [p[1] for p in pred]
    if pred_res == res:
        exact += 1
    total += 1

print(f"Val Exact Match: {exact / total * 100:.2f}%")

