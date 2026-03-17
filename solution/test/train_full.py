import json, glob, numpy as np
import xgboost as xgb
import os

print("Loading 30,000 races...")
races = []
for file in glob.glob("data/historical_races/races_*.json"): 
    with open(file) as f:
        races.extend(json.load(f))
print(f"Loaded {len(races)} races.")

X, y, groups = [], [], []

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
        
        tire = strat["starting_tire"]; age = 0
        pit_count = len(strat.get("pit_stops", []))
        for lap in range(1, total_laps+1):
            age += 1
            if tire == "SOFT": S_laps += 1; S_wear += age; S_wear2 += age**2
            elif tire == "MEDIUM": M_laps += 1; M_wear += age; M_wear2 += age**2
            elif tire == "HARD": H_laps += 1; H_wear += age; H_wear2 += age**2
                
            if lap in pit_map:
                tire = pit_map[lap]; age = 0
                
        drv_idx = int(strat["driver_id"][1:])
        
        feats[strat["driver_id"]] = [
            drv_idx, pit_count * pit_lane_time, 
            S_laps, M_laps, H_laps,
            S_wear, M_wear, H_wear,
            S_wear2, M_wear2, H_wear2,
            S_laps * T, M_laps * T, H_laps * T,
            total_laps, base, T,
            pit_count
        ]
        
    for driver in res:
        score = 19 - res.index(driver)
        X.append(feats[driver])
        y.append(score)
    groups.append(20)

X = np.array(X)
y = np.array(y)

print("Training XGBRanker...")
ranker = xgb.XGBRanker(
    tree_method='hist',
    eval_metric='ndcg',
    objective='rank:pairwise',
    learning_rate=0.1,
    n_estimators=1000,
    max_depth=8,
    random_state=42
)
ranker.fit(X, y, group=groups)
ranker.save_model("xgb_model.json")
print("Saved model to xgb_model.json")
