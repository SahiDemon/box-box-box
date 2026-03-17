import json, numpy as np

race = json.load(open('data/historical_races/races_00000-00999.json'))[0]

for p in race["strategies"].values():
    strat = p
    pit_map = {int(s["lap"]): s["to_tire"] for s in strat.get("pit_stops", [])}
    
    # tire age list
    ages = {"SOFT": [], "MEDIUM": [], "HARD": []}
    
    tire = strat["starting_tire"]
    age = 0
    for lap in range(1, int(race["race_config"]["total_laps"])+1):
        age += 1
        ages[tire].append(age)
        if lap in pit_map:
            tire = pit_map[lap]
            age = 0
            
    ages["SOFT"].sort()
    ages["MEDIUM"].sort()
    ages["HARD"].sort()
    
    print(p["driver_id"], "PITS:", len(strat.get("pit_stops", [])))
    for c in ["SOFT", "MEDIUM", "HARD"]:
        if ages[c]: print(f"  {c}: max_age={max(ages[c])} len={len(ages[c])} sum_age={sum(ages[c])}")
