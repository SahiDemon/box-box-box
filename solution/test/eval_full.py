import json, glob, numpy as np, sys
import xgboost as xgb

ranker = xgb.XGBRanker()
ranker.load_model("xgb_model.json")

test_files = glob.glob("data/test_cases/inputs/test_*.json")
test_files.sort()

def extract_features(test_data):
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
        X_test.append([
            drv_idx, pit_count * pit_lane_time, 
            S_laps, M_laps, H_laps,
            S_wear, M_wear, H_wear,
            S_wear2, M_wear2, H_wear2,
            S_laps * T, M_laps * T, H_laps * T,
            total_laps, base, T, pit_count
        ])
        drivers.append(strat["driver_id"])
    return np.array(X_test), drivers

if len(sys.argv) > 1 and sys.argv[1] == "evaluate":
    total_perfect = 0
    total_tested = 0
    for test_file in test_files:
        exp_file = test_file.replace("inputs", "expected_outputs")
        if not glob.glob(exp_file): continue
        
        expected = json.load(open(exp_file))["finishing_positions"]
        test_data = json.load(open(test_file))
        X_test, drivers = extract_features(test_data)
        
        preds = ranker.predict(X_test)
        pred_order = [drivers[i] for i in sorted(range(20), key=lambda i: -preds[i])]
        
        matches = sum([1 for p, e in zip(pred_order, expected) if p == e])
        print(f"{test_file[-12:]}: {matches}/20")
        if matches == 20: total_perfect += 1
        total_tested += 1
    print(f"Total perfect: {total_perfect} / {total_tested}")
else:
    # Run loop over ALL test inputs to generate outputs... normally done by test_runner.sh
    # But wait, test_runner runs command in run_command.txt
    pass
