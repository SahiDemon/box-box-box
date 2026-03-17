import json, glob, numpy as np
import xgboost as xgb

def load_races(num_files=10):
    races = []
    for file in glob.glob("data/historical_races/races_000*.json")[:num_files]: 
        with open(file) as f:
            races.extend(json.load(f))
    return races

races = load_races(5) # 5000 races

X = []
y = []
groups = []

for r_idx, r in enumerate(races):
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
        
        feats[strat["driver_id"]] = [
            drv_idx, # Driver index is likely monotonic
            pit_count * pit_lane_time, 
            S_laps, M_laps, H_laps,
            S_wear, M_wear, H_wear,
            S_wear2, M_wear2, H_wear2,
            S_laps * T, M_laps * T, H_laps * T,
            total_laps,
            base,
            T
        ]
        
    for driver in res:
        # target: higher score = faster (since we want ranker to push it up)
        # res[0] is fastest -> target 19
        # res[19] is slowest -> target 0
        score = 19 - res.index(driver)
        X.append(feats[driver])
        y.append(score)
        
    groups.append(20)

X = np.array(X)
y = np.array(y)

ranker = xgb.XGBRanker(
    tree_method="hist",
    eval_metric="ndcg",
    objective="rank:pairwise",
    learning_rate=0.03,
    n_estimators=3000,
    max_depth=6,
    subsample=0.8,
    random_state=42
)

ranker.fit(X, y, group=groups)
print("XGBRanker trained.")

def test_xgb(test_file):
    test_data = json.load(open(test_file))
    pred_feats = {}
    base = test_data["race_config"]["base_lap_time"]
    T = test_data["race_config"]["track_temp"] - 30.0
    pit_lane_time = test_data["race_config"]["pit_lane_time"]
    total_laps = int(test_data["race_config"]["total_laps"])

    X_test = []
    drivers = []
    
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
        pred_feats[strat["driver_id"]] = [
            drv_idx, pit_count * pit_lane_time, 
            S_laps, M_laps, H_laps,
            S_wear, M_wear, H_wear,
            S_wear2, M_wear2, H_wear2,
            S_laps * T, M_laps * T, H_laps * T,
            total_laps, base, T
        ]
        drivers.append(strat["driver_id"])

    X_test = np.array([pred_feats[d] for d in drivers])
    preds = ranker.predict(X_test)
    
    # higher pred = higher rank
    pred_order = sorted(range(20), key=lambda i: -preds[i])
    ordered_drivers = [drivers[i] for i in pred_order]
    return ordered_drivers

for tid in range(1, 10):
    test_file = f"data/test_cases/inputs/test_{tid:03d}.json"
    exp_file = f"data/test_cases/expected_outputs/test_{tid:03d}.json"
    expected = json.load(open(exp_file))["finishing_positions"]
    pred = test_xgb(test_file)
    matches = sum([1 for p, e in zip(pred, expected) if p == e])
    print(f"Test {tid:03d}: {matches}/20 exact matches.")
