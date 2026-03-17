import json, glob, numpy as np
from sklearn.svm import LinearSVC

def degradation_sums(n):
    if n <= 0: return 0.0, 0.0
    return n*(n+1)/2.0, n*(n+1)*(2*n+1)/6.0

race = json.load(open('data/historical_races/races_00000-00999.json'))[0]

base = race["race_config"]["base_lap_time"]
T = race["race_config"]["track_temp"] - 30.0 
pit_lane_time = race["race_config"]["pit_lane_time"]

feats = {}
for pos_key, p in race["strategies"].items():
    strat = p
    pit_map = {int(s["lap"]): s["to_tire"] for s in strat.get("pit_stops", [])}
    counts = {"SOFT": 0.0, "MEDIUM": 0.0, "HARD": 0.0}
    degs = {"SOFT": 0.0, "MEDIUM": 0.0, "HARD": 0.0}
    degs2 = {"SOFT": 0.0, "MEDIUM": 0.0, "HARD": 0.0}
    
    tire = strat["starting_tire"]
    age = 0
    pit_count = len(strat.get("pit_stops", []))
    for lap in range(1, int(race["race_config"]["total_laps"])+1):
        age += 1
        counts[tire] += 1
        free = {"SOFT": 6, "MEDIUM": 9, "HARD": 12}[tire]
        s1, s2 = degradation_sums(age - free)
        degs[tire] += s1 if age > free else 0
        degs2[tire] += s2 if age > free else 0
        if lap in pit_map:
            tire = pit_map[lap]
            age = 0
            
    x = [pit_count * pit_lane_time, pit_count, int(pos_key[3:])]
    for c in ["SOFT", "MEDIUM", "HARD"]:
        x += [counts[c]*base, degs[c]*base, degs2[c]*base, counts[c]*base*T, degs[c]*base*T, degs2[c]*base*T]
    
    feats[p["driver_id"]] = np.array(x)

res = race["finishing_positions"]
X_diff, Y = [], []
for i in range(19):
    for j in range(i+1, 20):
        X_diff.append(feats[res[i]] - feats[res[j]])
        Y.append(-1)
        X_diff.append(feats[res[j]] - feats[res[i]])
        Y.append(1)

svc = LinearSVC(C=10.0, dual=False, fit_intercept=False)
svc.fit(X_diff, Y)
print("Single race SVc accuracy with pos:", svc.score(X_diff, Y))
