"""
Mumbai Suburban Railway - TFLite model training & export.

The training data is *modelled from real facts* rather than placeholders:

* Topology (lines, stations, commute types, weather) is read from
  ../android/app/src/main/assets/transit_meta.json, so the integer feature
  encodings used here are exactly the ones the Android app uses when it encodes
  a user's selection.
* Corridor loads, station footfall tiers and the bimodal rush profile are
  calibrated to published Mumbai Suburban Railway aggregates (see the *_WEIGHT /
  *_LOAD tables below; they are approximations of public figures, not measured
  per-station records).
* Weather penalties follow the monsoon reality:
  Waterlogging >> Heavy Monsoon Rain > Light Rain > Clear.
* Sunday daytime maintenance "mega blocks" delay services, matching the real
  weekend block pattern on both CR and WR.

Generation is chunked, so arbitrarily large datasets (10M+ rows) stream through
tf.data without being materialised in memory.

Feature vector (float32, shape (6,)):
    [hour_of_day, day_of_week, line_code, station_code, type_code, weather_code]
Target (float32, shape (1,)):
    [delay_mins]
"""

import json
import math
import os

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Topology - loaded from the shared transit_meta.json asset
# --------------------------------------------------------------------------- #

HERE = os.path.dirname(os.path.abspath(__file__))
META_PATH = os.path.normpath(
    os.path.join(HERE, "..", "android", "app", "src", "main", "assets", "transit_meta.json")
)

_FALLBACK_META = {
    "version": "1.0.0",
    "city": "Mumbai",
    "lines": [
        "Western Line", "Central Line", "Harbour Line",
        "Trans-Harbour Line", "Port Line", "Vasai Road-Roha Line",
    ],
    "stations": [
        "Churchgate", "CSMT", "Mumbai Central", "Dadar", "Bandra", "Andheri", "Borivali",
        "Bhayandar", "Vasai Road", "Virar", "Dahanu Road", "Parel", "Kurla", "Ghatkopar",
        "Thane", "Dombivli", "Kalyan", "Kasara", "Khopoli", "Sandhurst Road", "Wadala Road",
        "Mahim Junction", "Vashi", "Nerul", "Belapur", "Panvel",
    ],
    "transit_types": ["Slow Local", "Fast Local", "AC Local"],
    "crowd_labels": [
        "Low/Seated", "Moderate/Standing", "Heavy Crush Load",
        "Super Dense Crush Load (SDCL)",
    ],
    "weather_conditions": ["Clear", "Light Rain", "Heavy Monsoon Rain", "Waterlogging"],
}


def load_transit_meta():
    """Reads the shared transit_meta.json, falling back to the embedded defaults."""
    try:
        with open(META_PATH, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        required = ("lines", "stations", "transit_types", "weather_conditions")
        if all(meta.get(key) for key in required):
            return meta
    except (OSError, ValueError, json.JSONDecodeError):
        pass
    return _FALLBACK_META


META = load_transit_meta()
CITY = META.get("city", "Mumbai")
LINES = list(META["lines"])
STATIONS = list(META["stations"])
TRANSIT_TYPES = list(META["transit_types"])
WEATHER = list(META["weather_conditions"])
CROWD_LABELS = list(META.get("crowd_labels", []))

LINE_INDEX = {name: i for i, name in enumerate(LINES)}
STATION_INDEX = {name: i for i, name in enumerate(STATIONS)}
TYPE_INDEX = {name: i for i, name in enumerate(TRANSIT_TYPES)}
WEATHER_INDEX = {name: i for i, name in enumerate(WEATHER)}

# Stations served by each line, including the shared interchange hubs
# (Dadar, Kurla, Wadala Road) that connect the corridors.
LINE_STATIONS = {
    "Western Line": ["Churchgate", "Mumbai Central", "Dadar", "Bandra", "Andheri",
                     "Borivali", "Bhayandar", "Vasai Road", "Virar", "Dahanu Road"],
    "Central Line": ["CSMT", "Sandhurst Road", "Parel", "Dadar", "Kurla", "Ghatkopar",
                     "Thane", "Dombivli", "Kalyan", "Kasara", "Khopoli"],
    "Harbour Line": ["CSMT", "Sandhurst Road", "Wadala Road", "Mahim Junction", "Kurla",
                     "Vashi", "Nerul", "Belapur", "Panvel"],
    "Trans-Harbour Line": ["Thane", "Vashi", "Nerul", "Panvel"],
    "Port Line": ["CSMT", "Sandhurst Road", "Wadala Road", "Mahim Junction"],
    "Vasai Road-Roha Line": ["Vasai Road", "Bhayandar", "Panvel"],
}

# --------------------------------------------------------------------------- #
# Calibration constants (approximate published aggregates / known behaviour)
# --------------------------------------------------------------------------- #

# Relative corridor load. Central and Western carry the bulk of the ~7.5M
# daily suburban passenger journeys; Harbour / Trans-Harbour / Port are lighter.
LINE_LOAD = {
    "Western Line": 1.25, "Central Line": 1.35, "Harbour Line": 0.85,
    "Trans-Harbour Line": 0.65, "Port Line": 0.55, "Vasai Road-Roha Line": 0.50,
}

# Relative station footfall tier: trunk terminals and interchanges (Churchgate,
# CSMT, Dadar, Kurla, Thane, Kalyan) are the busiest nodes on the network.
STATION_FOOTFALL = {
    "Churchgate": 3.0, "CSMT": 3.0, "Dadar": 3.0, "Kurla": 2.6, "Thane": 2.4,
    "Kalyan": 2.2, "Andheri": 2.2, "Borivali": 2.0, "Bandra": 1.8, "Ghatkopar": 1.8,
    "Mumbai Central": 1.6, "Panvel": 1.5, "Virar": 1.4, "Vashi": 1.4, "Dombivli": 1.3,
    "Bhayandar": 1.0, "Vasai Road": 1.0, "Belapur": 1.0, "Nerul": 1.0, "Parel": 0.9,
    "Wadala Road": 0.9, "Sandhurst Road": 0.8, "Mahim Junction": 0.8,
    "Dahanu Road": 0.5, "Kasara": 0.4, "Khopoli": 0.4,
}
DEFAULT_FOOTFALL = 0.6

# Bimodal weekday rush profile. Services run roughly 04:00-01:00; the system
# peaks in the 08-11 and 17-21 windows (matches the Android app + train_and_export).
HOUR_LOAD = np.array([
    0.15, 0.05, 0.02, 0.02,   # 00-03
    0.20, 0.60, 1.30, 2.80,   # 04-07
    4.20, 4.50, 3.50, 2.50,   # 08-11
    1.70, 1.55, 1.70, 2.00,   # 12-15
    2.60, 3.90, 4.60, 4.30,   # 16-19
    3.30, 2.40, 1.60, 0.95,   # 20-23
])
SERVICE_HOURS = tuple(range(4, 24))

# Rolling-stock mix and how heavily each is loaded (AC locals are on average
# less crowded than the general Slow locals).
TYPE_DEMAND = {"Slow Local": 1.10, "Fast Local": 1.00, "AC Local": 0.85}
TYPE_SHARE = {"Slow Local": 0.55, "Fast Local": 0.33, "AC Local": 0.12}

# Monsoon probabilities and their effect on demand and running time.
WEATHER_PROB = {"Clear": 0.55, "Light Rain": 0.20, "Heavy Monsoon Rain": 0.15, "Waterlogging": 0.10}
WEATHER_DEMAND = {"Clear": 1.00, "Light Rain": 0.97, "Heavy Monsoon Rain": 0.90, "Waterlogging": 0.78}
WEATHER_DELAY = {"Clear": 0.0, "Light Rain": 1.2, "Heavy Monsoon Rain": 5.0, "Waterlogging": 20.0}

# Day-of-week load (Mon..Sun). Weekend traffic drops; Sunday also carries the
# daytime maintenance mega block.
DOW_LOAD = np.array([1.00, 1.00, 1.00, 1.00, 1.05, 0.85, 0.55])

# Pressure -> SDCL crowd band thresholds (occupancy load factor).
CROWD_THRESHOLDS = (0.25, 0.65, 1.50)
PRESSURE_SCALE = 0.60
SUNDAY_MEGA_BLOCK_HOURS = (10, 16)

MORNING_PEAK = (8, 11)
EVENING_PEAK = (17, 21)


def _in_peak(hour):
    return (MORNING_PEAK[0] <= hour <= MORNING_PEAK[1]) or (EVENING_PEAK[0] <= hour <= EVENING_PEAK[1])


# --------------------------------------------------------------------------- #
# Sampling machinery
# --------------------------------------------------------------------------- #

# Precompute the per-line station-index arrays and sampling probabilities.
_LINE_PROB = np.array([LINE_LOAD.get(ln, 0.5) for ln in LINES], dtype=np.float64)
_LINE_PROB = _LINE_PROB / _LINE_PROB.sum()

_LINE_STATION_IDX = [
    np.array([STATION_INDEX[s] for s in LINE_STATIONS.get(ln, []) if s in STATION_INDEX], dtype=np.int64)
    for ln in LINES
]

_TYPE_PROB = np.array([TYPE_SHARE.get(t, 1.0) for t in TRANSIT_TYPES], dtype=np.float64)
_TYPE_PROB = _TYPE_PROB / _TYPE_PROB.sum()

_WEATHER_PROB = np.array([WEATHER_PROB.get(w, 1.0) for w in WEATHER], dtype=np.float64)
_WEATHER_PROB = _WEATHER_PROB / _WEATHER_PROB.sum()

_FOOTFALL = np.array([STATION_FOOTFALL.get(s, DEFAULT_FOOTFALL) for s in STATIONS], dtype=np.float64)
_FOOTFALL_FACTOR = _FOOTFALL / _FOOTFALL.mean()

_LINE_FACTOR = np.array([LINE_LOAD.get(ln, 0.5) for ln in LINES], dtype=np.float64)
_LINE_FACTOR = _LINE_FACTOR / _LINE_FACTOR.mean()

_HOUR_FACTOR = HOUR_LOAD / HOUR_LOAD.mean()

_TYPE_FACTOR = np.array([TYPE_DEMAND.get(t, 1.0) for t in TRANSIT_TYPES], dtype=np.float64)
_WEATHER_DEMAND = np.array([WEATHER_DEMAND.get(w, 1.0) for w in WEATHER], dtype=np.float64)
_WEATHER_DELAY = np.array([WEATHER_DELAY.get(w, 0.0) for w in WEATHER], dtype=np.float64)


def _sample(num_samples, rng):
    """Vectorised draw of realistic (context, delay, crowd) observations."""
    hour = rng.integers(SERVICE_HOURS[0], SERVICE_HOURS[-1] + 1, size=num_samples).astype(np.int64)
    dow = rng.integers(0, 7, size=num_samples).astype(np.int64)
    line_code = rng.choice(len(LINES), size=num_samples, p=_LINE_PROB).astype(np.int64)

    station_code = np.empty(num_samples, dtype=np.int64)
    for i in range(len(LINES)):
        mask = line_code == i
        count = int(mask.sum())
        if count:
            station_code[mask] = rng.choice(_LINE_STATION_IDX[i], size=count)

    type_code = rng.choice(len(TRANSIT_TYPES), size=num_samples, p=_TYPE_PROB).astype(np.int64)
    weather_code = rng.choice(len(WEATHER), size=num_samples, p=_WEATHER_PROB).astype(np.int64)

    # Demand load factor from the calibrated factors.
    pressure = (
        PRESSURE_SCALE
        * _HOUR_FACTOR[hour]
        * _FOOTFALL_FACTOR[station_code]
        * _LINE_FACTOR[line_code]
        * _TYPE_FACTOR[type_code]
        * _WEATHER_DEMAND[weather_code]
        * DOW_LOAD[dow]
    )

    # Crowd band (0 = Low/Seated ... 3 = SDCL) from the load factor.
    crowd_index = np.digitize(pressure, CROWD_THRESHOLDS).astype(np.int32)

    # Delay: base + congestion + weather + Sunday mega block + noise.
    congestion = 1.6 * np.clip(pressure, 0.0, 3.0)
    mega_block = np.where(
        (dow == 6) & (hour >= SUNDAY_MEGA_BLOCK_HOURS[0]) & (hour <= SUNDAY_MEGA_BLOCK_HOURS[1]),
        np.maximum(0.0, rng.normal(14.0, 4.0, size=num_samples)),
        0.0,
    )
    noise = rng.normal(0.0, 1.5, size=num_samples)
    delay = np.maximum(0.0, 1.0 + congestion + _WEATHER_DELAY[weather_code] + mega_block + noise)

    features = np.stack(
        [hour, dow, line_code, station_code, type_code, weather_code], axis=1
    ).astype(np.float32)
    return {
        "features": features,
        "delay": delay.astype(np.float32).reshape(-1, 1),
        "crowd_index": crowd_index,
        "hour": hour, "dow": dow, "line_code": line_code, "station_code": station_code,
        "type_code": type_code, "weather_code": weather_code, "pressure": pressure,
    }


# --------------------------------------------------------------------------- #
# Public dataset API
# --------------------------------------------------------------------------- #

def generate_batches(num_samples, batch_size, seed=0):
    """Yields (features, delay) float32 batches - the streaming training source."""
    rng = np.random.default_rng(seed)
    produced = 0
    while produced < num_samples:
        n = min(batch_size, num_samples - produced)
        sample = _sample(n, rng)
        produced += n
        yield sample["features"], sample["delay"]


def generate_mumbai_transit_data(num_samples=1_000_000, seed=42):
    """Returns a readable DataFrame of modelled Mumbai observations (for inspection)."""
    sample = _sample(int(num_samples), np.random.default_rng(seed))
    df = pd.DataFrame({
        "hour_of_day": sample["hour"],
        "day_of_week": sample["dow"],
        "line_encoded": sample["line_code"],
        "station_encoded": sample["station_code"],
        "type_encoded": sample["type_code"],
        "weather_encoded": sample["weather_code"],
        "line": [LINES[c] for c in sample["line_code"]],
        "station": [STATIONS[c] for c in sample["station_code"]],
        "transit_type": [TRANSIT_TYPES[c] for c in sample["type_code"]],
        "weather": [WEATHER[c] for c in sample["weather_code"]],
        "load_factor": sample["pressure"],
        "target_crowd_index": sample["crowd_index"],
        "target_delay_mins": sample["delay"].reshape(-1),
    })
    df["is_peak"] = [(1 if _in_peak(int(h)) else 0) for h in sample["hour"]]
    df["is_sunday_mega_block"] = [
        (1 if (int(d) == 6 and SUNDAY_MEGA_BLOCK_HOURS[0] <= int(h) <= SUNDAY_MEGA_BLOCK_HOURS[1]) else 0)
        for d, h in zip(sample["dow"], sample["hour"])
    ]
    return df


def build_tf_dataset(num_samples, batch_size, seed):
    """Streams generate_batches() into a tf.data.Dataset of (features, delay)."""
    import tensorflow as tf

    # The generator yields whole batches, so the leading dimension is dynamic.
    signature = (
        tf.TensorSpec(shape=(None, 6), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
    )
    dataset = tf.data.Dataset.from_generator(
        lambda: generate_batches(num_samples, batch_size, seed),
        output_signature=signature,
    )
    return dataset.prefetch(tf.data.AUTOTUNE)


# --------------------------------------------------------------------------- #
# Training and export
# --------------------------------------------------------------------------- #

NUM_SAMPLES = int(os.environ.get("NUM_SAMPLES", 10_000_000))
VAL_SAMPLES = int(os.environ.get("VAL_SAMPLES", 500_000))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 4096))
EPOCHS = int(os.environ.get("EPOCHS", 8))
EXPORT_NAME = os.environ.get("EXPORT_NAME", "transit_model.tflite")


def train_and_export():
    import tensorflow as tf

    print(f"[{CITY}] training on {NUM_SAMPLES:,} modelled observations")
    print(f"  lines={len(LINES)} stations={len(STATIONS)} types={TRANSIT_TYPES} weather={WEATHER}")
    print(f"  batch={BATCH_SIZE} epochs={EPOCHS} val={VAL_SAMPLES:,}")

    train_ds = build_tf_dataset(NUM_SAMPLES, BATCH_SIZE, seed=1).repeat()
    val_ds = build_tf_dataset(VAL_SAMPLES, BATCH_SIZE, seed=99).repeat()
    train_steps = math.ceil(NUM_SAMPLES / BATCH_SIZE)
    val_steps = max(1, math.ceil(VAL_SAMPLES / BATCH_SIZE))

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(6,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1, activation="linear"),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    # Streaming fit: memory stays flat regardless of NUM_SAMPLES.
    model.fit(
        train_ds,
        steps_per_epoch=train_steps,
        validation_data=val_ds,
        validation_steps=val_steps,
        epochs=EPOCHS,
        verbose=2,
    )

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    export_dir = os.path.normpath(os.path.join(HERE, "..", "android", "app", "src", "main", "assets"))
    os.makedirs(export_dir, exist_ok=True)
    out_path = os.path.join(export_dir, EXPORT_NAME)
    with open(out_path, "wb") as f:
        f.write(tflite_model)
    print(f"Exported {len(tflite_model):,} bytes -> {out_path}")


if __name__ == "__main__":
    train_and_export()
