#!/usr/bin/env python3
import glob
import json
import math
import os
import random
from typing import Dict, List, Tuple

COMPOUNDS = ("SOFT", "MEDIUM", "HARD")
DRIVER_IDS = [f"D{i:03d}" for i in range(1, 21)]
DRIVER_INDEX = {d: i for i, d in enumerate(DRIVER_IDS)}

FEATURE_NAMES = [
    "pit_lane",
    "pit_count",
    "soft_base_laps",
    "soft_base_dsum",
    "soft_base_dsum2",
    "soft_base_temp_laps",
    "soft_base_temp_dsum",
    "med_base_laps",
    "med_base_dsum",
    "med_base_dsum2",
    "med_base_temp_laps",
    "med_base_temp_dsum",
    "hard_base_laps",
    "hard_base_dsum",
    "hard_base_dsum2",
    "hard_base_temp_laps",
    "hard_base_temp_dsum",
]


def stable_softplus(x: float) -> float:
    if x > 0:
        return x + math.log1p(math.exp(-x))
    return math.log1p(math.exp(x))


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def build_pit_map(pit_stops: List[Dict]) -> Dict[int, str]:
    pit_map: Dict[int, str] = {}
    for stop in pit_stops:
        pit_map[int(stop["lap"])] = stop["to_tire"]
    return pit_map


def degradation_sums(stint_len: int, free_laps: int) -> Tuple[float, float]:
    if stint_len <= free_laps:
        return 0.0, 0.0
    n = float(stint_len - free_laps)
    return n * (n + 1.0) / 2.0, n * (n + 1.0) * (2.0 * n + 1.0) / 6.0


def extract_features(race_config: Dict, strategy: Dict, free_laps: Dict[str, int]) -> List[float]:
    total_laps = int(race_config["total_laps"])
    base = float(race_config["base_lap_time"])
    pit_lane_time = float(race_config["pit_lane_time"])
    temp_delta = float(race_config["track_temp"]) - 30.0

    pit_map = build_pit_map(strategy.get("pit_stops", []))

    laps = {c: 0.0 for c in COMPOUNDS}
    dsum = {c: 0.0 for c in COMPOUNDS}
    dsum2 = {c: 0.0 for c in COMPOUNDS}

    tire = strategy["starting_tire"]
    pit_count = 0
    stint_len = 0

    for lap in range(1, total_laps + 1):
        stint_len += 1
        if lap in pit_map:
            laps[tire] += float(stint_len)
            s1, s2 = degradation_sums(stint_len, free_laps[tire])
            dsum[tire] += s1
            dsum2[tire] += s2
            pit_count += 1
            tire = pit_map[lap]
            stint_len = 0

    if stint_len > 0:
        laps[tire] += float(stint_len)
        s1, s2 = degradation_sums(stint_len, free_laps[tire])
        dsum[tire] += s1
        dsum2[tire] += s2

    return [
        pit_lane_time * pit_count,
        float(pit_count),
        base * laps["SOFT"],
        base * dsum["SOFT"],
        base * dsum2["SOFT"],
        base * temp_delta * laps["SOFT"],
        base * temp_delta * dsum["SOFT"],
        base * laps["MEDIUM"],
        base * dsum["MEDIUM"],
        base * dsum2["MEDIUM"],
        base * temp_delta * laps["MEDIUM"],
        base * temp_delta * dsum["MEDIUM"],
        base * laps["HARD"],
        base * dsum["HARD"],
        base * dsum2["HARD"],
        base * temp_delta * laps["HARD"],
        base * temp_delta * dsum["HARD"],
    ]


def iter_historical_races(repo_root: str):
    pattern = os.path.join(repo_root, "data", "historical_races", "races_*.json")
    for path in sorted(glob.glob(pattern)):
        with open(path, "r", encoding="utf-8") as file:
            races = json.load(file)
        for race in races:
            yield race


def build_dataset(
    repo_root: str,
    free_laps: Dict[str, int],
    max_races: int = 18000,
    seed: int = 123,
):
    data = []
    for race in iter_historical_races(repo_root):
        ordered = list(race["finishing_positions"])
        feats: Dict[str, List[float]] = {}
        for strategy in race["strategies"].values():
            driver = strategy["driver_id"]
            feats[driver] = extract_features(race["race_config"], strategy, free_laps)
        data.append((ordered, feats))
        if len(data) >= max_races:
            break

    random.seed(seed)
    random.shuffle(data)
    split = int(0.9 * len(data))
    return data[:split], data[split:]


def fit_standardizer(data, sample_pairs: int = 160000, seed: int = 11):
    random.seed(seed)
    mean = [0.0 for _ in FEATURE_NAMES]
    m2 = [0.0 for _ in FEATURE_NAMES]

    for n in range(1, sample_pairs + 1):
        ordered, feats = random.choice(data)
        i = random.randint(0, 18)
        j = random.randint(i + 1, 19)
        x = [a - b for a, b in zip(feats[ordered[i]], feats[ordered[j]])]
        for k, xk in enumerate(x):
            delta = xk - mean[k]
            mean[k] += delta / n
            m2[k] += delta * (xk - mean[k])

    std = [math.sqrt(max(m2[k] / max(1, sample_pairs - 1), 1e-12)) for k in range(len(mean))]
    return mean, std


def norm_diff(diff: List[float], mean: List[float], std: List[float]) -> List[float]:
    return [(diff[k] - mean[k]) / std[k] for k in range(len(diff))]


def train_pairwise(train_data, mean, std, epochs=12, steps_per_epoch=220000, lr=0.015, l2=1e-5, seed=7):
    random.seed(seed)
    w = [0.0 for _ in FEATURE_NAMES]
    b = [0.0 for _ in DRIVER_IDS]

    m = [0.0 for _ in FEATURE_NAMES]
    v = [0.0 for _ in FEATURE_NAMES]
    mb = [0.0 for _ in DRIVER_IDS]
    vb = [0.0 for _ in DRIVER_IDS]
    b1, b2, eps = 0.9, 0.999, 1e-8
    step = 0

    for epoch in range(epochs):
        epoch_lr = lr * (0.88 ** epoch)
        for _ in range(steps_per_epoch):
            ordered, feats = random.choice(train_data)
            i = random.randint(0, 18)
            j = random.randint(i + 1, 19)
            faster = ordered[i]
            slower = ordered[j]

            raw = [a - b for a, b in zip(feats[faster], feats[slower])]
            x = norm_diff(raw, mean, std)
            margin = sum(wk * xk for wk, xk in zip(w, x))
            margin += b[DRIVER_INDEX[faster]] - b[DRIVER_INDEX[slower]]
            gc = sigmoid(margin) - 1.0

            step += 1
            for k in range(len(w)):
                g = gc * x[k] + l2 * w[k]
                m[k] = b1 * m[k] + (1 - b1) * g
                v[k] = b2 * v[k] + (1 - b2) * (g * g)
                mhat = m[k] / (1 - b1 ** step)
                vhat = v[k] / (1 - b2 ** step)
                w[k] -= epoch_lr * mhat / (math.sqrt(vhat) + eps)

            fi = DRIVER_INDEX[faster]
            si = DRIVER_INDEX[slower]
            for idx, sign in ((fi, 1.0), (si, -1.0)):
                g = sign * gc + l2 * b[idx]
                mb[idx] = b1 * mb[idx] + (1 - b1) * g
                vb[idx] = b2 * vb[idx] + (1 - b2) * (g * g)
                mhat = mb[idx] / (1 - b1 ** step)
                vhat = vb[idx] / (1 - b2 ** step)
                b[idx] -= epoch_lr * mhat / (math.sqrt(vhat) + eps)

        probe = 0.0
        probe_n = 3000
        for _ in range(probe_n):
            ordered, feats = random.choice(train_data)
            i = random.randint(0, 18)
            j = random.randint(i + 1, 19)
            raw = [a - b for a, b in zip(feats[ordered[i]], feats[ordered[j]])]
            x = norm_diff(raw, mean, std)
            mrg = sum(wk * xk for wk, xk in zip(w, x))
            mrg += b[DRIVER_INDEX[ordered[i]]] - b[DRIVER_INDEX[ordered[j]]]
            probe += stable_softplus(mrg)
        print(f"epoch={epoch + 1} probe_loss={probe / probe_n:.6f}")

    return w, b


def score_driver(driver_id: str, feat: List[float], w: List[float], b: List[float], mean: List[float], std: List[float]) -> float:
    val = 0.0
    for k, fk in enumerate(feat):
        val += w[k] * ((fk - mean[k]) / std[k])
    val += b[DRIVER_INDEX[driver_id]]
    return val


def evaluate_exact(data, w, b, mean, std) -> float:
    correct = 0
    for ordered, feats in data:
        pred = sorted(ordered, key=lambda d: (score_driver(d, feats[d], w, b, mean, std), d))
        if pred == ordered:
            correct += 1
    return correct / len(data) if data else 0.0


def evaluate_pairwise(data, w, b, mean, std, pairs=60000, seed=99) -> float:
    random.seed(seed)
    ok = 0
    for _ in range(pairs):
        ordered, feats = random.choice(data)
        i = random.randint(0, 18)
        j = random.randint(i + 1, 19)
        f = ordered[i]
        s = ordered[j]
        if score_driver(f, feats[f], w, b, mean, std) < score_driver(s, feats[s], w, b, mean, std):
            ok += 1
    return ok / pairs


def to_params(free_laps: Dict[str, int], w: List[float], b: List[float], std: List[float]) -> Dict:
    unscaled = [w[k] / std[k] for k in range(len(w))]

    return {
        "pit_lane_multiplier": unscaled[0],
        "pit_stop_fixed": unscaled[1],
        "driver_bias": {driver_id: b[DRIVER_INDEX[driver_id]] for driver_id in DRIVER_IDS},
        "compound": {
            "SOFT": {
                "free_laps": free_laps["SOFT"],
                "lap_mult": unscaled[2],
                "deg_linear_mult": unscaled[3],
                "deg_quadratic_mult": unscaled[4],
                "temp_lap_mult": unscaled[5],
                "temp_deg_mult": unscaled[6],
            },
            "MEDIUM": {
                "free_laps": free_laps["MEDIUM"],
                "lap_mult": unscaled[7],
                "deg_linear_mult": unscaled[8],
                "deg_quadratic_mult": unscaled[9],
                "temp_lap_mult": unscaled[10],
                "temp_deg_mult": unscaled[11],
            },
            "HARD": {
                "free_laps": free_laps["HARD"],
                "lap_mult": unscaled[12],
                "deg_linear_mult": unscaled[13],
                "deg_quadratic_mult": unscaled[14],
                "temp_lap_mult": unscaled[15],
                "temp_deg_mult": unscaled[16],
            },
        },
    }


def main() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    candidates = [
        {"SOFT": 4, "MEDIUM": 7, "HARD": 10},
        {"SOFT": 5, "MEDIUM": 8, "HARD": 11},
        {"SOFT": 6, "MEDIUM": 9, "HARD": 12},
        {"SOFT": 7, "MEDIUM": 10, "HARD": 13},
    ]

    best = None
    for free_laps in candidates:
        print("-" * 72)
        print(f"Trying free_laps={free_laps}")
        train_data, val_data = build_dataset(repo_root, free_laps=free_laps, max_races=18000, seed=123)
        print(f"Train races: {len(train_data)}  Val races: {len(val_data)}")

        mean, std = fit_standardizer(train_data, sample_pairs=160000, seed=11)
        w, b = train_pairwise(train_data, mean, std, epochs=10, steps_per_epoch=170000, lr=0.012, l2=1e-5, seed=7)

        val_pair = evaluate_pairwise(val_data, w, b, mean, std, pairs=50000)
        val_exact = evaluate_exact(val_data, w, b, mean, std)
        print(f"Val pairwise: {val_pair * 100:.2f}%")
        print(f"Val exact:    {val_exact * 100:.2f}%")

        if best is None or (val_pair, val_exact) > (best["val_pair"], best["val_exact"]):
            best = {
                "free_laps": free_laps,
                "w": w,
                "b": b,
                "std": std,
                "val_pair": val_pair,
                "val_exact": val_exact,
            }

    print("=" * 72)
    print(f"Best free_laps: {best['free_laps']}")
    print(f"Best val pairwise: {best['val_pair'] * 100:.2f}%")
    print(f"Best val exact:    {best['val_exact'] * 100:.2f}%")

    params = to_params(best["free_laps"], best["w"], best["b"], best["std"])
    out_path = os.path.join(os.path.dirname(__file__), "model_params.json")
    with open(out_path, "w", encoding="utf-8") as file:
        json.dump(params, file, indent=2)
    print(f"Saved params: {out_path}")


if __name__ == "__main__":
    main()
