import json, glob, numpy as np
from sklearn.neural_network import MLPClassifier

def load_races(num_files=3):
    races = []
    for file in glob.glob("data/historical_races/races_000*.json")[:num_files]: 
        with open(file) as f:
            races.extend(json.load(f))
    return races

races = load_races(3) # 3000 races

X_diff = []
y_diff = []

for r in races:
    res = r["finishing_positions"]
    base = r["race_config"]["base_lap_time"]
    T = r["race_config"]["track_temp"] - 30.0 
    pit_lane_time = r["race_config"]["pit_lane_time"]
    total_laps = int(r["race_config"]["total_laps"])
    
    feats = {}
    for pos_key, strat in r["strategies"].items():
        pit_map = {int(stop["lap"]): stop["to_tire"] for stop in strat.get("pit_stops", [])}
        S_laps, M_laps, H_laps = 0, 0, 0
        S_wear, M_wear, H_wear = 0, 0, 0
        S_wear2, M_wear2, H_wear2 = 0, 0, 0
        
        tire = strat["starting_tire"]
        age = 0
        pit_count = len(strat.get("pit_stops", []))
        for lap in range(1, total_laps+1):
            age += 1
            if tire == "SOFT": S_laps += 1; S_wear += age; S_wear2 += age**2
            elif tire == "MEDIUM": M_laps += 1; M_wear += age; M_wear2 += age**2
            elif tire == "HARD": H_laps += 1; H_wear += age; H_wear2 += age**2
                
            if lap in pit_map:
                tire = pit_map[lap]
                age = 0
                
        drv_idx = int(strat["driver_id"][1:])
        
        # provide the raw aggregates 
        feats[strat["driver_id"]] = np.array([
            pit_count * pit_lane_time, 
            S_laps, M_laps, H_laps,
            S_wear, M_wear, H_wear,
            S_wear2, M_wear2, H_wear2,
            S_laps * T, M_laps * T, H_laps * T,
            drv_idx,
            total_laps,
            base
        ])
        
    for i in range(19):
        for j in range(i+1, 20):
            X_diff.append(feats[res[i]] - feats[res[j]])
            y_diff.append(1)
            X_diff.append(feats[res[j]] - feats[res[i]])
            y_diff.append(0)

X_diff = np.array(X_diff)
y_diff = np.array(y_diff)

clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
clf.fit(X_diff, y_diff)
print("MLP Training Accuracy:", clf.score(X_diff, y_diff))

test_data = json.load(open("data/test_cases/inputs/test_001.json"))
pred_feats = {}
base = test_data["race_config"]["base_lap_time"]
T = test_data["race_config"]["track_temp"] - 30.0
pit_lane_time = test_data["race_config"]["pit_lane_time"]
total_laps = int(test_data["race_config"]["total_laps"])

for pos_key, strat in test_data["strategies"].items():
    pit_map = {int(stop["lap"]): stop["to_tire"] for stop in strat.get("pit_stops", [])}
    S_laps, M_laps, H_laps = 0, 0, 0
    S_wear, M_wear, H_wear = 0, 0, 0
    S_wear2, M_wear2, H_wear2 = 0, 0, 0
    
    tire = strat["starting_tire"]
    age = 0
    pit_count = len(strat.get("pit_stops", []))
    for lap in range(1, total_laps+1):
        age += 1
        if tire == "SOFT": S_laps += 1; S_wear += age; S_wear2 += age**2
        elif tire == "MEDIUM": M_laps += 1; M_wear += age; M_wear2 += age**2
        elif tire == "HARD": H_laps += 1; H_wear += age; H_wear2 += age**2
            
        if lap in pit_map:
            tire = pit_map[lap]
            age = 0
            
    drv_idx = int(strat["driver_id"][1:])
    pred_feats[strat["driver_id"]] = np.array([
            pit_count * pit_lane_time, 
            S_laps, M_laps, H_laps,
            S_wear, M_wear, H_wear,
            S_wear2, M_wear2, H_wear2,
            S_laps * T, M_laps * T, H_laps * T,
            drv_idx,
            total_laps,
            base
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
