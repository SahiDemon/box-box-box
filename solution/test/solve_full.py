import json, glob, numpy as np
from sklearn.svm import LinearSVC
import pickle

def degradation_sums(n):
    if n <= 0: return 0.0, 0.0
    return n*(n+1)/2.0, n*(n+1)*(2*n+1)/6.0

def load_races(num_files=10):
    races = []
    files = sorted(glob.glob("data/historical_races/races_*.json"))[:num_files]
    for file in files:
        with open(file) as f:
            races.extend(json.load(f))
    return races

def eval_freelaps(races, free_soft, free_med, free_hard):
    X_diff, Y = [], []
    for r in races:
        base = r["race_config"]["base_lap_time"]
        T = r["race_config"]["track_temp"] - 30.0 
        pit_lane_time = r["race_config"]["pit_lane_time"]
        res = r["finishing_positions"]
        
        feats = {}
        for p in r["strategies"].values():
            strat = p
            pit_map = {int(s["lap"]): s["to_tire"] for s in strat.get("pit_stops", [])}
            
            counts = {"SOFT": 0.0, "MEDIUM": 0.0, "HARD": 0.0}
            degs = {"SOFT": 0.0, "MEDIUM": 0.0, "HARD": 0.0}
            degs2 = {"SOFT": 0.0, "MEDIUM": 0.0, "HARD": 0.0}
            
            tire = strat["starting_tire"]
            age = 0
            pit_count = len(strat.get("pit_stops", []))
            for lap in range(1, int(r["race_config"]["total_laps"])+1):
                age += 1
                counts[tire] += 1
                
                free = {"SOFT": free_soft, "MEDIUM": free_med, "HARD": free_hard}[tire]
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
            
        for i in range(19):
            for j in range(i+1, 20):
                d1 = feats[res[i]]
                d2 = feats[res[j]]
                if np.random.random() < 0.5:
                    X_diff.append(d1 - d2)
                    Y.append(-1)
                else:
                    X_diff.append(d2 - d1)
                    Y.append(1)
                    
    X_diff = np.array(X_diff)
    Y = np.array(Y)

    svc = LinearSVC(C=1.0, dual=False, fit_intercept=False, max_iter=20000, tol=1e-5)
    svc.fit(X_diff, Y)
    sc = svc.score(X_diff, Y)
    return sc, svc

print("Loading 5000 races...")
races = load_races(5)

print("Training SVM...")
fs, fm, fh = 6, 9, 12
sc, svc = eval_freelaps(races, fs, fm, fh)
print(f"Accuracy: {sc*100:.2f}%")

w = svc.coef_[0]
scale = 1.0 / w[0]
w_scaled = w * scale

params = {
    "pit_lane_multiplier": w_scaled[0],
    "pit_stop_fixed": w_scaled[1],
    "driver_bias": {f"D{i+1:03d}": w_scaled[20+i] for i in range(20)},
    "compound": {
        "SOFT": {
            "free_laps": fs,
            "lap_mult": w_scaled[2],
            "deg_linear_mult": w_scaled[3],
            "deg_quadratic_mult": w_scaled[4],
            "temp_lap_mult": w_scaled[5],
            "temp_deg_mult": w_scaled[6],
            "temp_quad_mult": w_scaled[7],
        },
        "MEDIUM": {
            "free_laps": fm,
            "lap_mult": w_scaled[8],
            "deg_linear_mult": w_scaled[9],
            "deg_quadratic_mult": w_scaled[10],
            "temp_lap_mult": w_scaled[11],
            "temp_deg_mult": w_scaled[12],
            "temp_quad_mult": w_scaled[13],
        },
        "HARD": {
            "free_laps": fh,
            "lap_mult": w_scaled[14],
            "deg_linear_mult": w_scaled[15],
            "deg_quadratic_mult": w_scaled[16],
            "temp_lap_mult": w_scaled[17],
            "temp_deg_mult": w_scaled[18],
            "temp_quad_mult": w_scaled[19],
        },
    }
}

with open("solution/model_params.json", "w") as f:
    json.dump(params, f, indent=2)

print("Saved model_params.json")
