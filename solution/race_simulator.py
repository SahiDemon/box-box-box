import sys
import json
import numpy as np
import xgboost as xgb
import os

def main():
    input_json = sys.stdin.read()
    if not input_json.strip():
        return
    
    test_case = json.loads(input_json)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'xgb_model.json')

    model = xgb.XGBRanker()
    model.load_model(model_path)
    
    base = test_case['race_config']['base_lap_time']
    T = test_case['race_config']['track_temp'] - 30.0
    pit_lane_time = test_case['race_config']['pit_lane_time']
    total_laps = int(test_case['race_config']['total_laps'])

    feats = {}
    for pos_key, strat in test_case['strategies'].items():
        pit_map = {int(stop['lap']): stop['to_tire'] for stop in strat.get('pit_stops', [])}
        S_laps, M_laps, H_laps = 0, 0, 0
        S_wear, M_wear, H_wear = 0, 0, 0
        S_wear2, M_wear2, H_wear2 = 0, 0, 0

        tire = strat['starting_tire']
        age = 0
        pit_count = len(strat.get('pit_stops', []))
        
        for lap in range(1, total_laps + 1):
            age += 1
            if tire == 'SOFT': 
                S_laps += 1; S_wear += age; S_wear2 += age**2
            elif tire == 'MEDIUM': 
                M_laps += 1; M_wear += age; M_wear2 += age**2
            elif tire == 'HARD': 
                H_laps += 1; H_wear += age; H_wear2 += age**2

            if lap in pit_map:
                tire = pit_map[lap]
                age = 0

        drv_idx = int(strat['driver_id'][1:])

        feats[strat['driver_id']] = [
            drv_idx, pit_count * pit_lane_time,
            S_laps, M_laps, H_laps,
            S_wear, M_wear, H_wear,
            S_wear2, M_wear2, H_wear2,
            S_laps * T, M_laps * T, H_laps * T,
            total_laps, base, T,
            pit_count
        ]

    drivers = list(feats.keys())
    X = np.array([feats[d] for d in drivers])
    
    scores = model.predict(X)
    
    ranked_drivers = [d for _, d in sorted(zip(scores, drivers), key=lambda x: (x[0], -int(x[1][1:])), reverse=True)]

    output = {
        'race_id': test_case['race_id'],
        'finishing_positions': ranked_drivers
    }
    
    print(json.dumps(output))

if __name__ == '__main__':
    main()
