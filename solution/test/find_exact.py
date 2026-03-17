import json, numpy as np
from scipy.optimize import linprog
import glob

print("Loading data...")
races = []
for file in glob.glob('data/historical_races/races_000*.json'):
    with open(file) as f: 
        races.extend(json.load(f))
print(f"Loaded {len(races)} races.")

X_diff = []
b_rhs = [] 
labels = []

def get_feats(strat, laps, pit_time, base_time, T):
    pit_map = {int(p['lap']): p['to_tire'] for p in strat.get('pit_stops', [])}
    tire = strat['starting_tire']
    age = 0
    pc = len(strat.get('pit_stops', []))
    
    feats = np.zeros(15)
    
    for l in range(1, laps+1):
        age += 1
        idx = 0 if tire == 'SOFT' else (1 if tire == 'MEDIUM' else 2)
        feats[idx*5 + 0] += 1
        feats[idx*5 + 1] += age
        feats[idx*5 + 2] += age**2
        feats[idx*5 + 3] += T
        feats[idx*5 + 4] += T * age
        if l in pit_map: 
            tire = pit_map[l]
            age = 0
            
    return pc * pit_time, laps * base_time, feats

for r in races[:500]: 
    res = r['finishing_positions']
    base = r['race_config']['base_lap_time']
    T = r['race_config']['track_temp']
    pit = r['race_config']['pit_lane_time']
    laps = int(r['race_config']['total_laps'])
    
    driver_feats = {}
    for pos_key, strat in r['strategies'].items():
        pit_time_tot, base_time_tot, f_comp = get_feats(strat, laps, pit, base, T)
        driver_feats[strat['driver_id']] = (pit_time_tot, base_time_tot, f_comp)
        
    for i in range(19):
        for j in range(i+1, 20):
            d1 = res[i]; d2 = res[j]
            p1, b1, f1 = driver_feats[d1]
            p2, b2, f2 = driver_feats[d2]
            
            diff_f = f1 - f2
            diff_p = p1 - p2
            diff_b = b1 - b2
            
            if np.any(diff_f != 0) or diff_p != 0:
                X_diff.append(diff_f)
                b_rhs.append(-0.001 - diff_p - diff_b)
                labels.append((d1, d2))

num_vars = 15
num_cons = len(X_diff)

print(f"Solving with {num_cons} constraints...")

A_ub = np.hstack([np.array(X_diff), -np.eye(num_cons)])
b_ub = np.array(b_rhs)

c = np.zeros(num_vars + num_cons)
c[num_vars:] = 1.0 

bounds = [(None, None)] * num_vars + [(0, None)] * num_cons

result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

print('Status:', result.message)
if result.success:
    slacks = result.x[num_vars:]
    violated = np.sum(slacks > 0.001)
    print(f'Violated constraints: {violated} out of {num_cons}')
    
    W = result.x[:num_vars]
    labels_feat = ["S_const", "S_age", "S_age2", "S_T", "S_T_age",
                   "M_const", "M_age", "M_age2", "M_T", "M_T_age",
                   "H_const", "H_age", "H_age2", "H_T", "H_T_age"]
                   
    print("\nCoefficients:")
    for i, w in enumerate(W):
        print(f"{labels_feat[i]}: {w:.6f}")
