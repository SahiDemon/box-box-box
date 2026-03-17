import json, glob, numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def load_races(num_files=1):
    races = []
    for file in glob.glob("data/historical_races/races_000*.json")[:num_files]: 
        with open(file) as f:
            races.extend(json.load(f))
    return races

races = load_races(2) # 2000 races

X_diff = []
y_diff = []

for r in races:
    res = r["finishing_positions"]
    base = r["race_config"]["base_lap_time"]
    T = r["race_config"]["track_temp"] - 30.0 
    pit_lane_time = r["race_config"]["pit_lane_time"]
    total_laps = int(r["race_config"]["total_laps"])
    track_id = {"Monaco":0,"Bahrain":1,"Spa":2,"Monza":3,"Silverstone":4,"COTA":5,"Suzuka":6}[r["race_config"]["track"]]
    
    feats = {}
    for pos_key, strat in r["strategies"].items():
        pit_map = {int(stop["lap"]): stop["to_tire"] for stop in strat.get("pit_stops", [])}
        S_laps, M_laps, H_laps = 0, 0, 0
        S_wear, M_wear, H_wear = 0, 0, 0
        
        tire = strat["starting_tire"]
        age = 0
        pit_count = len(strat.get("pit_stops", []))
        for lap in range(1, total_laps+1):
            age += 1
            if tire == "SOFT": S_laps += 1; S_wear += age
            elif tire == "MEDIUM": M_laps += 1; M_wear += age
            elif tire == "HARD": H_laps += 1; H_wear += age
                
            if lap in pit_map:
                tire = pit_map[lap]
                age = 0
                
        # Driver id
        drv_idx = int(strat["driver_id"][1:])
        
        feats[strat["driver_id"]] = np.array([
            pit_count, pit_count * pit_lane_time, 
            S_laps, M_laps, H_laps,
            S_wear, M_wear, H_wear,
            drv_idx,
            track_id, base, T, total_laps
        ])
        
    for i in range(19):
        for j in range(i+1, 20):
            # Normal order: i is faster
            diff = feats[res[i]] - feats[res[j]]
            X_diff.append(diff)
            y_diff.append(1)
            
            # Flipped order: j is faster
            diff = feats[res[j]] - feats[res[i]]
            X_diff.append(diff)
            y_diff.append(0)

X_diff = np.array(X_diff)
y_diff = np.array(y_diff)

clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
clf.fit(X_diff, y_diff)
print("RF Training Accuracy:", clf.score(X_diff, y_diff))

# Test on test_001
import json
test_data = json.load(open("data/test_cases/inputs/test_001.json"))
track_id = {"Monaco":0,"Bahrain":1,"Spa":2,"Monza":3,"Silverstone":4,"COTA":5,"Suzuka":6}[test_data["race_config"]["track"]]

pred_feats = {}
base = test_data["race_config"]["base_lap_time"]
T = test_data["race_config"]["track_temp"] - 30.0
pit_lane_time = test_data["race_config"]["pit_lane_time"]
total_laps = int(test_data["race_config"]["total_laps"])

for pos_key, strat in test_data["strategies"].items():
    pit_map = {int(stop["lap"]): stop["to_tire"] for stop in strat.get("pit_stops", [])}
    S_laps, M_laps, H_laps = 0, 0, 0
    S_wear, M_wear, H_wear = 0, 0, 0
    
    tire = strat["starting_tire"]
    age = 0
    pit_count = len(strat.get("pit_stops", []))
    for lap in range(1, total_laps+1):
        age += 1
        if tire == "SOFT": S_laps += 1; S_wear += age
        elif tire == "MEDIUM": M_laps += 1; M_wear += age
        elif tire == "HARD": H_laps += 1; H_wear += age
            
        if lap in pit_map:
            tire = pit_map[lap]
            age = 0
            
    drv_idx = int(strat["driver_id"][1:])
    pred_feats[strat["driver_id"]] = np.array([
        pit_count, pit_count * pit_lane_time, 
        S_laps, M_laps, H_laps,
        S_wear, M_wear, H_wear,
        drv_idx,
        track_id, base, T, total_laps
    ])

drivers = list(pred_feats.keys())
scores = {d: 0 for d in drivers}
for i in range(20):
    for j in range(i+1, 20):
        d1 = drivers[i]
        d2 = drivers[j]
        diff = pred_feats[d1] - pred_feats[d2]
        pred = clf.predict([diff])[0]
        if pred == 1:
            scores[d1] += 1
        else:
            scores[d2] += 1

predicted_order = sorted(drivers, key=lambda d: -scores[d])
expected = json.load(open("data/test_cases/expected_outputs/test_001.json"))["finishing_positions"]

matches = sum([1 for p, e in zip(predicted_order, expected) if p == e])
print(f"Test 001 Matches: {matches} / 20")
