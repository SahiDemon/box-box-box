import json, glob, numpy as np
from sklearn.svm import LinearSVC
import pickle

for file in glob.glob("data/historical_races/races_000*.json")[:1]: 
    with open(file) as f:
        data = json.load(f)
races = data[:400] 

X_diff, Y = [], []

print("Extracting age-histogram features...")
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
            
        # Age histograms (max 80 laps)
        hist_S = np.zeros(80)
        hist_M = np.zeros(80)
        hist_H = np.zeros(80)
        
        tire = strat["starting_tire"]
        age = 0
        pit_count = len(strat.get("pit_stops", []))
        for lap in range(1, int(r["race_config"]["total_laps"])+1):
            age += 1
            if tire == "SOFT": hist_S[age] += 1
            elif tire == "MEDIUM": hist_M[age] += 1
            elif tire == "HARD": hist_H[age] += 1
                
            if lap in pit_map:
                tire = pit_map[lap]
                age = 0
                
        # Features
        x = [pit_count * pit_lane_time, pit_count]
        
        # for each tire, add base*hist, T*base*hist, T*hist, hist
        x.extend(hist_S * base)
        x.extend(hist_M * base)
        x.extend(hist_H * base)
        
        x.extend(hist_S * base * T)
        x.extend(hist_M * base * T)
        x.extend(hist_H * base * T)
        
        x.extend(hist_S * T)
        x.extend(hist_M * T)
        x.extend(hist_H * T)

        x.extend(hist_S)
        x.extend(hist_M)
        x.extend(hist_H)
        
        drivers = [f"D{i:03d}" for i in range(1, 21)]
        d_arr = np.zeros(20)
        d_arr[drivers.index(p["driver_id"])] = 1.0
        
        feats[p["driver_id"]] = np.concatenate([np.array(x), d_arr])
        
    for i in range(19):
        for j in range(i+1, 20):
            if np.random.random() < 0.5:
                X_diff.append(feats[res[i]] - feats[res[j]])
                Y.append(-1)
            else:
                X_diff.append(feats[res[j]] - feats[res[i]])
                Y.append(1)

X_diff = np.array(X_diff)
Y = np.array(Y)

print("Training LinearSVC on histograms...")
svc = LinearSVC(C=1.0, dual=False, fit_intercept=False, max_iter=20000)
svc.fit(X_diff, Y)
print(f"Accuracy: {svc.score(X_diff, Y)*100:.2f}%")

