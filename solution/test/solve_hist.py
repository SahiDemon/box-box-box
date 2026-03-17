import json, glob, numpy as np
from sklearn.svm import LinearSVC
import sys

def get_feats(strat, laps, pit, base, T):
    pit_map = {int(p['lap']): p['to_tire'] for p in strat.get('pit_stops', [])}
    tire = strat['starting_tire']
    age = 0
    pc = len(strat.get('pit_stops', []))
    
    # 0 = SOFT, 1 = MEDIUM, 2 = HARD
    f_hist = np.zeros(3 * 60)
    f_hist_T = np.zeros(3 * 60)
    
    for l in range(1, laps+1):
        age += 1
        idx = 0 if tire == 'SOFT' else (1 if tire == 'MEDIUM' else 2)
        if age <= 60:
            f_hist[idx*60 + age - 1] += 1
            f_hist_T[idx*60 + age - 1] += T
        if l in pit_map: 
            tire = pit_map[l]
            age = 0
            
    return np.array([pc * pit, laps * base] + list(f_hist) + list(f_hist_T))

races = []
for file in glob.glob('data/historical_races/races_000*.json'): 
    with open(file) as f: races.extend(json.load(f))

print(f"Loaded {len(races)} races.")

X_diff = []
y_diff = []
for r in races:
    res = r['finishing_positions']
    base = r['race_config']['base_lap_time']
    T = r['race_config']['track_temp'] - 30.0 
    pit = r['race_config']['pit_lane_time']
    laps = int(r['race_config']['total_laps'])
    feats = {}
    for pos_key, strat in r['strategies'].items():
        feats[strat['driver_id']] = get_feats(strat, laps, pit, base, T)
        
    for i in range(19):
        for j in range(i+1, 20):
            d1 = res[i]; d2 = res[j]
            f_diff = feats[d1] - feats[d2]
            # Exclude exact ties to prevent contradictory labels
            if np.any(f_diff != 0):
                X_diff.append(f_diff)
                y_diff.append(1)
                X_diff.append(-f_diff)
                y_diff.append(0)

X_diff = np.array(X_diff)
y_diff = np.array(y_diff)

clf = LinearSVC(dual=False, random_state=42, C=10)
clf.fit(X_diff, y_diff)
print("Train Acc:", clf.score(X_diff, y_diff))

test_files = glob.glob("data/test_cases/inputs/test_*.json")
test_files.sort()
perf = 0
for test_file in test_files:
    exp_file = test_file.replace("inputs", "expected_outputs")
    if not glob.glob(exp_file): continue
    
    expected = json.load(open(exp_file))["finishing_positions"]
    test_data = json.load(open(test_file))
    base = test_data["race_config"]["base_lap_time"]
    T = test_data["race_config"]["track_temp"] - 30.0
    pit = test_data["race_config"]["pit_lane_time"]
    laps = int(test_data["race_config"]["total_laps"])

    X_test = []
    drivers = []
    for pos_key, strat in test_data["strategies"].items():
        X_test.append(get_feats(strat, laps, pit, base, T))
        drivers.append(strat["driver_id"])
    X_test = np.array(X_test)
    
    scores = {d: 0 for d in drivers}
    for i in range(20):
        for j in range(i+1, 20):
            d1 = drivers[i]
            d2 = drivers[j]
            diff = X_test[i] - X_test[j]
            pred = clf.predict([diff])[0]
            if pred == 1:
                scores[d1] += 1
            else:
                scores[d2] += 1
                
    # Sort purely by scores. In exact match, scores will be unique 19 to 0. 
    # If tie, fallback to native driver string sort as a tie-breaker. (which happens inherently with stable sort and string alphabetical)
    pred_order = sorted(drivers, key=lambda d: (-scores[d], d))
    
    matches = sum([1 for p, e in zip(pred_order, expected) if p == e])
    if matches == 20: perf += 1
    # print(f"{test_file[-12:]}: {matches}")
print(f"Total Perfect: {perf}/100")
