# AI Transit Demand & Overcrowding Intelligence Platform (Render-ready)

import os
import re
import io
import json
import math
import tempfile
import requests
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import gradio as gr
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

# Set random seed for reproducibility
np.random.seed(42)

print("Dependencies successfully imported!")

# ---------- Mumbai Suburban Railway metadata ----------
# transit_meta.json (shared with the Android asset) is the single source of truth
# for the network: this module loads it so the demo dataset, line/station dropdowns
# and weather options all reflect the real Western/Central/Harbour/Trans-Harbour/
# Port/Vasai Road-Roha lines and their rolling stock.
TRANSIT_META_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "proj", "android", "app", "src", "main", "assets", "transit_meta.json",
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
    "weather_conditions": ["Clear", "Light Rain", "Heavy Monsoon Rain", "Waterlogging"],
}

def load_transit_meta():
    """Reads the shared transit_meta.json, falling back to the embedded defaults."""
    try:
        with open(TRANSIT_META_PATH, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        required = ("lines", "stations", "transit_types", "weather_conditions")
        if all(meta.get(key) for key in required):
            return meta
    except (OSError, ValueError, json.JSONDecodeError):
        pass
    return _FALLBACK_META

TRANSIT_META = load_transit_meta()

# Stations served by each line, including the shared interchange hubs
# (Dadar, Kurla, Wadala Road) that are critical for network connectivity.
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

# Approximate per-rake capacity by rolling stock.
TYPE_CAPACITY = {"Slow Local": 500, "Fast Local": 560, "AC Local": 700}

WEATHER_PROB = {
    "Clear": 0.55, "Light Rain": 0.20, "Heavy Monsoon Rain": 0.15, "Waterlogging": 0.10,
}
WEATHER_FACTOR = {
    "Clear": 1.0, "Light Rain": 0.92, "Heavy Monsoon Rain": 0.80, "Waterlogging": 0.70,
}

# Mumbai suburban rush windows, matching proj/tflite_export/train_and_export.py.
MORNING_PEAK = (8, 11)
EVENING_PEAK = (17, 21)

def _in_peak(hour):
    return (MORNING_PEAK[0] <= hour <= MORNING_PEAK[1]) or (EVENING_PEAK[0] <= hour <= EVENING_PEAK[1])

class TransitDataEngine:
    """Manages dataset ingestion, automatic mapping, cleaning, and feature engineering."""
    def __init__(self):
        self.df_raw = None
        self.df_cleaned = None
        self.df_featured = None
        self.mapping = {}
        self.quality_report = {}
        self.is_demo = False
        self.warnings = []

    @staticmethod
    def generate_demo_dataset(num_days=30):
        """Generates a realistic Mumbai Suburban Railway dataset driven by transit_meta.json."""
        meta = TRANSIT_META
        station_roster = set(meta.get("stations", []))
        lines = [ln for ln in meta.get("lines", []) if LINE_STATIONS.get(ln)]
        transit_types = meta.get("transit_types") or ["Train"]
        weathers = meta.get("weather_conditions") or ["Clear"]
        weights = np.array([WEATHER_PROB.get(w, 1.0) for w in weathers], dtype=float)
        weights = weights / weights.sum()

        data = []
        start_date = datetime.now() - timedelta(days=num_days)

        for day in range(num_days):
            current_date = start_date + timedelta(days=day)
            is_weekend = 1 if current_date.weekday() >= 5 else 0
            weather = np.random.choice(weathers, p=weights)
            temp_hi = 28 if weather in ("Heavy Monsoon Rain", "Waterlogging") else 35
            temp = np.random.randint(22, temp_hi)

            for line in lines:
                route_stations = [s for s in LINE_STATIONS[line] if s in station_roster]
                for station in route_stations:
                    for commute_type in transit_types:
                        cap = TYPE_CAPACITY.get(commute_type, 500)
                        for hour in range(6, 24):  # Suburban services run 06:00-23:00
                            # Mumbai demand heuristics: weekday rush windows + weekend midday peak
                            if is_weekend:
                                base_demand = cap * (0.7 if 11 <= hour <= 16 else 0.25)
                            elif MORNING_PEAK[0] <= hour <= MORNING_PEAK[1]:
                                base_demand = cap * 1.15
                            elif EVENING_PEAK[0] <= hour <= EVENING_PEAK[1]:
                                base_demand = cap * 1.25
                            elif 11 <= hour <= 14:
                                base_demand = cap * 0.5
                            else:
                                base_demand = cap * 0.15

                            # Apply random noise and weather factor
                            noise = np.random.normal(0, cap * 0.08)
                            weather_factor = WEATHER_FACTOR.get(weather, 1.0)

                            ridership = max(0, int(base_demand * weather_factor + noise))

                            data.append({
                                "timestamp": current_date.replace(hour=hour, minute=0, second=0),
                                "service_route": line,
                                "station_stop": station,
                                "transport_mode": commute_type,
                                "passenger_count": ridership,
                                "vehicle_capacity": cap,
                                "weather_condition": weather,
                                "temperature_celsius": temp
                            })

        return pd.DataFrame(data)

    def detect_columns(self, df):
        """Uses heuristics to match variations of common transit dataset columns."""
        column_heuristics = {
            "ridership": ["ridership", "passengers", "passenger_count", "passengers_count", "demand", "boardings", "trips", "entries"],
            "route": ["route", "route_id", "line", "line_id", "service", "bus_route", "service_route"],
            "station": ["station", "station_name", "stop", "stop_name", "origin", "destination", "station_stop"],
            "capacity": ["capacity", "vehicle_capacity", "seats", "max_capacity"],
            "date": ["date", "timestamp", "datetime", "day"],
            "mode": ["mode", "transport_type", "vehicle_type", "transport_mode"],
            "weather": ["weather", "weather_condition", "weather_status", "outlook"]
        }

        detected = {}
        lower_cols = {col.lower(): col for col in df.columns}

        for target_key, patterns in column_heuristics.items():
            for pattern in patterns:
                if pattern in lower_cols:
                    detected[target_key] = lower_cols[pattern]
                    break
            if target_key not in detected:
                # Fallback to loose regex matching
                for col in df.columns:
                    if any(p in col.lower() for p in patterns):
                        detected[target_key] = col
                        break

        self.mapping = detected
        return detected

    def clean_data(self, df, custom_mapping=None):
        """Preprocesses and formats the target dataframe dynamically."""
        self.df_raw = df.copy()
        self.warnings = []
        if custom_mapping:
            self.mapping.update(custom_mapping)

        # Standardize mandatory structures
        m = self.mapping

        total_rows = len(df)
        duplicates = df.duplicated().sum()
        df = df.drop_duplicates().copy()

        # Handle Missing values in core prediction areas
        nulls = df.isnull().sum().sum()

        # Parse Datetime
        date_col = m.get("date")
        if date_col and date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df = df.dropna(subset=[date_col])
            if len(df) and df[date_col].dt.hour.nunique() <= 1:
                self.warnings.append("No time-of-day information found (dates only). Hourly and peak-hour features are not meaningful, so treat forecasts as day-level.")
        else:
            # Generate dummy timestamps if dates are fully absent
            df['inferred_timestamp'] = pd.date_range(start="2023-10-01 06:00:00", periods=len(df), freq='h')
            self.mapping["date"] = 'inferred_timestamp'
            self.warnings.append("No date/time column detected. Synthetic hourly timestamps were generated, so time-based patterns are not real.")

        # Parse / clean ridership
        rid_col = m.get("ridership")
        if rid_col and rid_col in df.columns:
            df[rid_col] = pd.to_numeric(df[rid_col], errors='coerce').fillna(0)
            # Correct outliers or negative elements
            df[rid_col] = df[rid_col].apply(lambda x: max(0, x))
        else:
            df['inferred_ridership'] = 100
            self.warnings.append("No ridership column detected. A constant placeholder value was used, so forecasts are meaningless.")
            self.mapping["ridership"] = 'inferred_ridership'

        # Handle missing capacity
        cap_col = m.get("capacity")
        if not cap_col or cap_col not in df.columns:
            df['assumed_capacity'] = 100
            self.mapping["capacity"] = 'assumed_capacity'
            self.warnings.append("No capacity column detected. Capacity was assumed to be 100 per vehicle.")
        else:
            df[cap_col] = pd.to_numeric(df[cap_col], errors='coerce').fillna(100).clip(lower=1)

        # Clean routes & stations
        route_col = m.get("route")
        if not route_col or route_col not in df.columns:
            df['assigned_route'] = 'All Network'
            self.mapping["route"] = 'assigned_route'

        station_col = m.get("station")
        if not station_col or station_col not in df.columns:
            df['assigned_station'] = 'Unified Node'
            self.mapping["station"] = 'assigned_station'

        # Categorical columns are always strings so dropdown values match the data
        for key in ("route", "station", "mode", "weather"):
            col = self.mapping.get(key)
            if col and col in df.columns:
                df[col] = df[col].fillna("Unknown").astype(str)

        # Quality calculations
        self.df_cleaned = df.reset_index(drop=True)
        self.quality_report = {
            "Total Records": total_rows,
            "Duplicates Dropped": duplicates,
            "Null Entries Imputed": nulls,
            "Valid Clean Records": len(self.df_cleaned)
        }
        return self.df_cleaned

    def engineer_features(self):
        """Generates lag metrics and time categories ensuring zero target-leak."""
        df = self.df_cleaned.copy()
        m = self.mapping

        date_col = m["date"]
        rid_col = m["ridership"]
        route_col = m["route"]

        df['year'] = df[date_col].dt.year
        df['month'] = df[date_col].dt.month
        df['day'] = df[date_col].dt.day
        df['day_of_week'] = df[date_col].dt.dayofweek
        df['hour'] = df[date_col].dt.hour
        df['weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

        # Categorize peak windows (Mumbai suburban rush)
        df['morning_peak'] = df['hour'].apply(lambda h: 1 if MORNING_PEAK[0] <= h <= MORNING_PEAK[1] else 0)
        df['evening_peak'] = df['hour'].apply(lambda h: 1 if EVENING_PEAK[0] <= h <= EVENING_PEAK[1] else 0)
        df['peak_hour'] = df['hour'].apply(lambda h: 1 if _in_peak(h) else 0)

        # Lag features are computed per route AND station, so a row only ever sees
        # the same stop's earlier hours (never a sibling station's same-hour value).
        station_col = m["station"]
        df = df.sort_values(by=date_col, kind="mergesort")
        grp = df.groupby([route_col, station_col])[rid_col]
        overall_mean = df[rid_col].mean()
        df['previous_hour_demand'] = grp.shift(1).fillna(overall_mean)
        df['rolling_mean_3h'] = grp.transform(
            lambda s: s.shift(1).rolling(3, min_periods=1).mean()
        ).fillna(overall_mean)

        self.df_featured = df.reset_index(drop=True)
        return self.df_featured

class TransitPredictor:
    """Trains and manages Scikit-Learn regressors for transport ridership forecasting."""
    def __init__(self):
        self.model = None
        self.feature_cols = []
        self.metrics = {}
        self.importances = {}
        self.route_categories = []
        self.weather_categories = []
        self.beats_baseline = True

    def train(self, df, mapping):
        """Fits an ensemble ML regressor and computes robust error performance metrics."""
        rid_col = mapping["ridership"]
        route_col = mapping["route"]

        # Standard ML features
        self.feature_cols = [
            'hour', 'day_of_week', 'weekend', 'morning_peak', 'evening_peak',
            'peak_hour', 'previous_hour_demand', 'rolling_mean_3h'
        ]

        # Factor in categorical metadata safely using pandas Categorical Codes
        df_ml = df.copy()
        route_cat = df_ml[route_col].astype('category')
        self.route_categories = list(route_cat.cat.categories)
        df_ml['route_code'] = route_cat.cat.codes
        self.feature_cols.append('route_code')

        # Handle optional elements
        weather_col = mapping.get("weather")
        if weather_col and weather_col in df_ml.columns:
            weather_cat = df_ml[weather_col].astype('category')
            self.weather_categories = list(weather_cat.cat.categories)
            df_ml['weather_code'] = weather_cat.cat.codes
            self.feature_cols.append('weather_code')

        X = df_ml[self.feature_cols]
        y = df_ml[rid_col]

        # Time-aware sequential split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Model Instantiation
        self.model = RandomForestRegressor(n_estimators=40, max_depth=8, random_state=42)
        self.model.fit(X_train, y_train)

        # Testing / Evaluation
        preds = self.model.predict(X_test)
        baseline_preds = np.full_like(y_test, y_train.mean())

        self.metrics = {
            "Model MAE": float(mean_absolute_error(y_test, preds)),
            "Model RMSE": float(root_mean_squared_error(y_test, preds)),
            "Model R2": float(r2_score(y_test, preds)),
            "Baseline MAE": float(mean_absolute_error(y_test, baseline_preds)),
            "Baseline RMSE": float(root_mean_squared_error(y_test, baseline_preds)),
            "Baseline R2": float(r2_score(y_test, baseline_preds))
        }

        self.beats_baseline = self.metrics["Model MAE"] < self.metrics["Baseline MAE"]

        # Feature importances
        importances = self.model.feature_importances_
        self.importances = dict(zip(self.feature_cols, importances))

    def predict_one(self, hour, day_of_week, route_val, prev_demand, weather_val, df_ref, mapping):
        """Infers single-instance forecasts utilizing custom user controls."""
        if self.model is None:
            return 0.0

        route_col = mapping["route"]
        weather_col = mapping.get("weather")

        if route_val not in self.route_categories:
            raise ValueError(f"Route '{route_val}' is not in the loaded dataset.")
        route_code = self.route_categories.index(route_val)
        weather_code = 0
        if 'weather_code' in self.feature_cols and weather_val in self.weather_categories:
            weather_code = self.weather_categories.index(weather_val)

        # Recalculate feature attributes
        is_weekend = 1 if day_of_week >= 5 else 0
        m_peak = 1 if MORNING_PEAK[0] <= hour <= MORNING_PEAK[1] else 0
        e_peak = 1 if EVENING_PEAK[0] <= hour <= EVENING_PEAK[1] else 0
        p_hour = 1 if (m_peak or e_peak) else 0

        feat_dict = {
            'hour': hour,
            'day_of_week': day_of_week,
            'weekend': is_weekend,
            'morning_peak': m_peak,
            'evening_peak': e_peak,
            'peak_hour': p_hour,
            'previous_hour_demand': prev_demand,
            'rolling_mean_3h': prev_demand,
            'route_code': route_code
        }
        if 'weather_code' in self.feature_cols:
            feat_dict['weather_code'] = weather_code

        input_df = pd.DataFrame([feat_dict])[self.feature_cols]
        return float(self.model.predict(input_df)[0])

class TransitDecisionCore:
    """Calculates occupancy stress metrics and optimizes route infrastructure allocation."""
    @staticmethod
    def calculate_occupancy(ridership, capacity):
        if capacity <= 0:
            return 0, "Low"
        rate = (ridership / capacity) * 100
        if rate < 50:
            risk = "Low"
        elif rate <= 75:
            risk = "Moderate"
        elif rate <= 90:
            risk = "High"
        elif rate <= 100:
            risk = "Critical"
        else:
            risk = "Overcapacity"
        return round(rate, 1), risk

    @staticmethod
    def optimize_allocation(demand, capacity, target_occupancy=85, is_peak_hour=False):
        """Determines refined service adjustments and additional fleet counts based on peak-hour thresholds."""
        if capacity <= 0:
            return {"status": "Operational", "add_vehicles": 0, "freq_adjustment_pct": 0, "notes": "No capacity defined."}

        # During peak hours, we use a higher target occupancy threshold (e.g. 90%) to tolerate higher loads
        effective_target = (target_occupancy + 5) if is_peak_hour else target_occupancy
        target_cap = capacity * (effective_target / 100.0)
        required_units = math.ceil(demand / target_cap) if demand > 0 else 1

        if demand > capacity:
            add_vehicles = max(1, required_units - 1)
            freq_inc = min(100, int(((demand - capacity) / capacity) * 100))
            peak_msg = " [PEAK WINDOW ACTIVE]" if is_peak_hour else ""
            return {
                "status": "Deploy Extra Fleet",
                "add_vehicles": add_vehicles,
                "freq_adjustment_pct": freq_inc,
                "notes": f"Demand exceeds standard limit{peak_msg}. Deploy approx {add_vehicles} units to stabilize flow."
            }
        elif is_peak_hour and demand > (capacity * 0.75):
            # Pre-emptive frequency boost for peak windows to prevent cascade overcrowding
            return {
                "status": "Pre-emptive Peak Frequency Boost",
                "add_vehicles": 0,
                "freq_adjustment_pct": 15,
                "notes": "Peak hour demand is approaching capacity threshold. Pre-emptively increasing frequency by 15%."
            }
        elif demand < (capacity * 0.35) and not is_peak_hour:
            return {
                "status": "Consolidate Fleet",
                "add_vehicles": 0,
                "freq_adjustment_pct": -20,
                "notes": "Low demand corridor. Recommend reducing standard frequency to conserve city energy."
            }
        else:
            return {
                "status": "Sufficient Service",
                "add_vehicles": 0,
                "freq_adjustment_pct": 0,
                "notes": "Normal status. Standard operational cycles adequate."
            }

MAX_ROWS = int(os.environ.get("MAX_ROWS", 200_000))
MAX_FILE_MB = int(os.environ.get("MAX_FILE_MB", 25))

def build_pipeline(df):
    """Builds an engine + trained predictor without touching global state,
    so a failed upload can never corrupt the dataset that is currently live."""
    if len(df) > MAX_ROWS:
        raise ValueError(f"Dataset has {len(df):,} rows; the limit is {MAX_ROWS:,}. Please trim or aggregate it.")
    eng, pred = TransitDataEngine(), TransitPredictor()
    eng.detect_columns(df)
    eng.clean_data(df)
    if len(eng.df_cleaned) < 30:
        raise ValueError("Need at least 30 valid rows to train a model.")
    eng.engineer_features()
    pred.train(eng.df_featured, eng.mapping)
    return eng, pred

engine, predictor = build_pipeline(TransitDataEngine.generate_demo_dataset())
engine.is_demo = True
print("Platform Engine is ready and populated with active demonstration metrics!")

# ---------- helpers that read whatever dataset is currently loaded ----------
def get_choices(key, limit=500):
    df, col = engine.df_featured, engine.mapping.get(key)
    if df is not None and col and col in df.columns:
        return sorted(df[col].astype(str).unique().tolist())[:limit]
    return []

def route_capacity(route):
    m, df = engine.mapping, engine.df_featured
    rows = df[df[m["route"]] == route]
    return float(rows[m["capacity"]].median()) if not rows.empty else 100.0

def forecast_demand(route, hour, dow, weather=None):
    """Runs the live ML model for a route/hour, using that route's typical previous-hour load."""
    m, df = engine.mapping, engine.df_featured
    rows = df[df[m["route"]] == route]
    prev = rows[rows["hour"] == (hour - 1) % 24][m["ridership"]]
    prev_demand = float(prev.mean()) if len(prev) else float(rows[m["ridership"]].mean())
    wcol = m.get("weather")
    if weather is None and wcol and wcol in df.columns and not rows.empty:
        weather = rows[wcol].mode().iloc[0]
    return max(0.0, predictor.predict_one(hour, dow, route, prev_demand, weather, None, m))

RISK_STYLE = {"Low": ("🟢", "#28a745"), "Moderate": ("🟡", "#b8860b"), "High": ("🟠", "#fd7e14"),
              "Critical": ("🔴", "#dc3545"), "Overcapacity": ("🔴", "#dc3545")}

def process_uploaded_dataset(file_obj, url_str):
    """Ingests a new dataset, retrains the model, and swaps it in only if everything succeeds."""
    global engine, predictor
    try:
        if file_obj is not None:
            path = file_obj if isinstance(file_obj, str) else file_obj.name
            if os.path.getsize(path) > MAX_FILE_MB * 1024 * 1024:
                raise ValueError(f"File is larger than {MAX_FILE_MB} MB.")
            low = path.lower()
            if low.endswith((".csv", ".txt")):
                df = pd.read_csv(path)
            elif low.endswith(".xlsx"):
                df = pd.read_excel(path)
            else:
                raise ValueError("Unsupported file type. Please upload a .csv or .xlsx file.")
        elif url_str and url_str.strip():
            url = url_str.strip()
            if not url.lower().startswith(("http://", "https://")):
                raise ValueError("URL must start with http:// or https://")
            df = pd.read_csv(url, nrows=MAX_ROWS + 1)
        else:
            return "Please upload a file or enter a valid CSV url.", None, None

        new_engine, new_predictor = build_pipeline(df)
        engine, predictor = new_engine, new_predictor

        lines = [f"✅ Loaded {len(engine.df_cleaned):,} rows and retrained the model."]
        lines += [f"⚠️ {w}" for w in engine.warnings]
        if not predictor.beats_baseline:
            lines.append(f"⚠️ The model does not beat the simple-average baseline on held-out data (R² = {predictor.metrics['Model R2']:.2f}). Treat forecasts with caution.")

        report = dict(engine.quality_report)
        report.update({
            "Model R2": round(predictor.metrics["Model R2"], 3),
            "Model MAE": round(predictor.metrics["Model MAE"], 2),
            "Baseline MAE": round(predictor.metrics["Baseline MAE"], 2),
            "Detected Columns": ", ".join(f"{k}→{v}" for k, v in engine.mapping.items()),
        })
        return "\n".join(lines), engine.df_cleaned.head(20), pd.DataFrame([report])
    except Exception as e:
        return f"❌ Dataset load error (previous dataset kept): {e}", None, None

def get_kpi_indicators():
    """Computes analytical KPIs for the primary platform dashboard."""
    df = engine.df_featured
    m = engine.mapping
    rid = m["ridership"]
    cap = m["capacity"]

    total_ridership = int(df[rid].sum())
    avg_ridership = int(df[rid].mean())
    peak_hour = int(df.groupby('hour')[rid].mean().idxmax())

    # Risk classification
    df['occupancy_rate'] = (df[rid] / df[cap]) * 100
    overcapacity_count = int((df['occupancy_rate'] > 100).sum())

    top_route = df.groupby(m["route"])[rid].sum().idxmax()
    worst_station = df.groupby(m["station"])[rid].mean().idxmax()

    return total_ridership, avg_ridership, f"{peak_hour:02d}:00", top_route, worst_station, overcapacity_count

# Visualization Builders
def render_ridership_trend():
    df = engine.df_featured
    m = engine.mapping
    fig = px.line(
        df.groupby(m["date"])[m["ridership"]].sum().reset_index(),
        x=m["date"], y=m["ridership"],
        title="Network-Wide Ridership Over Time",
        template="plotly_white"
    )
    fig.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def render_hourly_profile():
    df = engine.df_featured
    m = engine.mapping
    fig = px.bar(
        df.groupby('hour')[m["ridership"]].mean().reset_index(),
        x='hour', y=m["ridership"],
        labels={m["ridership"]: "Avg Passengers"},
        title="Average Hourly Ridership Profile (Peak Analysis)",
        template="plotly_white"
    )
    fig.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def render_pressure_heatmap():
    df = engine.df_featured
    m = engine.mapping
    # Cross tabulation of routes vs hour
    top_routes = df.groupby(m["route"])[m["ridership"]].sum().nlargest(30).index
    matrix = df[df[m["route"]].isin(top_routes)].pivot_table(index=m["route"], columns='hour', values=m["ridership"], aggfunc='mean').fillna(0)
    fig = px.imshow(
        matrix,
        labels=dict(x="Hour of Day", y="Route", color="Ridership"),
        title="Transit Route & Hour Heatmap Map",
        color_continuous_scale="Viridis",
        template="plotly_white"
    )
    fig.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def render_feature_importance():
    feats = list(predictor.importances.keys())
    scores = list(predictor.importances.values())
    fig = px.bar(
        x=scores, y=feats, orientation='h',
        labels={'x': 'Relative Importance', 'y': 'Feature'},
        title="Model Feature Predictors (Permutation Explainability)",
        template="plotly_white"
    )
    fig.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def run_live_predictions(route_name, day_of_week_str, hour, prev_demand, weather_status):
    """Runs the currently trained model and turns the forecast into fleet recommendations."""
    day_mapping = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
    day_val = day_mapping.get(day_of_week_str, 0)
    if route_name not in predictor.route_categories:
        raise gr.Error(f"Route '{route_name}' is not in the loaded dataset. Pick a route from the dropdown.")
    hour = int(hour)

    val_pred = max(0.0, predictor.predict_one(hour, day_val, route_name, prev_demand or 0, weather_status, None, engine.mapping))

    # RMSE approximates the error std-dev, so +/-1.96*RMSE is a rough 95% band
    res = max(5.0, predictor.metrics.get("Model RMSE", 15.0))
    low_bound = max(0, int(val_pred - 1.96 * res))
    high_bound = int(val_pred + 1.96 * res)
    expected = int(val_pred)

    capacity_val = route_capacity(route_name)
    is_peak = _in_peak(hour)
    occ_pct, risk_level = TransitDecisionCore.calculate_occupancy(expected, capacity_val)
    alloc = TransitDecisionCore.optimize_allocation(expected, capacity_val, is_peak_hour=is_peak)

    rec_str = f"**Status**: {alloc['status']}\n\n**Fleet Adjustment**: +{alloc['add_vehicles']} vehicles\n\n**Frequency Change**: {alloc['freq_adjustment_pct']}%\n\n**Actionable Advice**: {alloc['notes']}"
    if not predictor.beats_baseline:
        rec_str += "\n\n⚠️ The current model does not beat the simple-average baseline. Treat this forecast with caution."
    return expected, f"{low_bound} to {high_bound}", f"{occ_pct}% ({risk_level})", rec_str

def execute_what_if_analysis(cap, demand_mult, target_occ):
    """Executes responsive simulation based on what-if variables."""
    base_demand = int(engine.df_featured[engine.mapping["ridership"]].mean())
    simulated_demand = int(base_demand * (demand_mult / 100.0))

    occ_pct, risk_level = TransitDecisionCore.calculate_occupancy(simulated_demand, cap)
    target_capacity_limit = cap * (target_occ / 100.0)

    required_vehicles = math.ceil(simulated_demand / target_capacity_limit) if simulated_demand > 0 else 1
    add_req = max(0, required_vehicles - 1)

    status_markdown = f"""
    ### Simulated Scenario Results:
    * **Simulated Ridership**: {simulated_demand} passengers (Multiplier applied: {demand_mult}%)
    * **Simulated Fleet Capacity**: {cap} spaces
    * **Resulting Occupancy**: **{occ_pct}%**
    * **Classified Risk Rating**: **{risk_level}**
    * **Calculated Vehicle Requirement**: **{required_vehicles} standard fleet units** (Requires **+{add_req}** additional vehicles under target {target_occ}% occupancy limit)
    """
    return status_markdown

def export_recommendations_csv():
    """Generates download-ready mitigation matrix containing priorities and recommended fleet actions with peak-hour adjustments."""
    df = engine.df_featured
    m = engine.mapping
    route_col = m["route"]
    rid_col = m["ridership"]
    cap_col = m["capacity"]

    # Compile routes containing stress incidents
    agg_df = df.groupby([route_col, 'hour']).agg({rid_col: 'mean', cap_col: 'first'}).reset_index()
    recs = []
    for idx, row in agg_df.iterrows():
        hr = int(row['hour'])
        is_peak = _in_peak(hr)
        alloc = TransitDecisionCore.optimize_allocation(row[rid_col], row[cap_col], is_peak_hour=is_peak)
        if alloc["add_vehicles"] > 0 or alloc["freq_adjustment_pct"] != 0:
            recs.append({
                "Route": row[route_col],
                "Hour": f"{hr}:00",
                "Avg Demand": int(row[rid_col]),
                "Available Capacity": int(row[cap_col]),
                "Risk Status": alloc["status"],
                "Additional Fleet Required": alloc["add_vehicles"],
                "Frequency Change Required": f"{alloc['freq_adjustment_pct']}%",
                "System Recommendation": alloc["notes"]
            })

    rec_df = pd.DataFrame(recs)
    if rec_df.empty:
        rec_df = pd.DataFrame([{"Status": "No adjustments required across networks."}])

    filepath = os.path.join(tempfile.mkdtemp(), "Transit_Intelligence_Mitigation_Plan.csv")
    rec_df.to_csv(filepath, index=False)
    return filepath


import uuid
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import gradio as gr
from datetime import datetime

# global storage for reports in-memory
citizen_reports_db = pd.DataFrame(columns=[
    "Report ID", "Timestamp", "Location", "Route/Station", "Problem Type", "Severity", "Description"
])

def submit_citizen_report(location, route_station, problem_type, severity, description):
    global citizen_reports_db
    new_id = str(uuid.uuid4())[:8]
    new_row = {
        "Report ID": new_id,
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Location": location,
        "Route/Station": route_station,
        "Problem Type": problem_type,
        "Severity": severity,
        "Description": description
    }
    citizen_reports_db = pd.concat([citizen_reports_db, pd.DataFrame([new_row])], ignore_index=True)
    return f"🚨 Thank you! Your report (ID: {new_id}) has been recorded. Transit authorities have been notified."

def get_reports_summary():
    global citizen_reports_db
    if citizen_reports_db.empty:
        return "No reports submitted yet.", pd.DataFrame()

    total = len(citizen_reports_db)
    most_common = citizen_reports_db["Problem Type"].mode().iloc[0] if not citizen_reports_db["Problem Type"].empty else "N/A"
    high_sev = len(citizen_reports_db[citizen_reports_db["Severity"] == "🔴 Critical"])

    summary_md = f"**Total Reports Received**: {total} | **Most Common Issue**: {most_common} | **🔴 Critical Severity Reports**: {high_sev}"
    return summary_md, citizen_reports_db

def _dow_from(date_val):
    dt = pd.to_datetime(date_val, errors="coerce")
    return (dt if pd.notna(dt) else datetime.now()).weekday()

def citizen_find_journey(from_st, to_st, date_val, hour_val, mode_val):
    m, df = engine.mapping, engine.df_featured
    hour_val, dow = int(hour_val), _dow_from(date_val)

    sub = df
    mode_col = m.get("mode")
    if mode_col and mode_col in df.columns and mode_val and mode_val != "All modes":
        sub = df[df[mode_col] == mode_val]
    if sub.empty:
        return "<p style='color:#102a43;'>No services found for that transport type.</p>"

    # Prefer routes serving both stops, then either stop, then any route
    routes = list(sub[m["route"]].unique())
    served = sub.groupby(m["route"])[m["station"]].agg(lambda s: set(s))
    both = [r for r in routes if {from_st, to_st} <= served[r]]
    either = [r for r in routes if {from_st, to_st} & served[r]]
    routes = (both or either or routes)[:10]

    scored = []
    for r in routes:
        cap = route_capacity(r)
        exp = forecast_demand(r, hour_val, dow)
        scored.append((exp / cap if cap > 0 else 0, r, exp, cap))
    _, route_name, expected, cap = min(scored)

    data_hours = set(int(h) for h in df["hour"].unique())
    window = [h for h in range(hour_val - 3, hour_val + 4) if h in data_hours] or [hour_val]
    forecasts = {h: forecast_demand(route_name, h, dow) for h in window}
    best_h = min(forecasts, key=forecasts.get)

    sel_occ, sel_risk = TransitDecisionCore.calculate_occupancy(expected, cap)
    best_occ, best_risk = TransitDecisionCore.calculate_occupancy(forecasts[best_h], cap)
    se, sc = RISK_STYLE[sel_risk]
    be, bc = RISK_STYLE[best_risk]

    if best_h != hour_val and best_occ < sel_occ:
        rec = f"""
    <div style='background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 5px solid {bc}; margin-bottom: 15px;'>
        <h3 style='color: {bc}; margin: 0;'>⭐ RECOMMENDED OPTION</h3>
        <p style='margin: 5px 0; color: #333;'><strong style='color:#333;'>{route_name} at {best_h:02d}:00</strong></p>
        <p style='margin: 2px 0; color: #333;'>Expected crowd level: <strong style='color:#333;'>{be} {best_risk} ({best_occ}% occupancy)</strong></p>
        <p style='margin: 2px 0; color: #333;'>Estimated passengers: {int(forecasts[best_h])} | Capacity: {int(cap)}</p>
        <p style='margin: 5px 0 0 0; font-size: 13px; color: #555;'><em style='color:#555;'>Lowest predicted crowding within 3 hours of your chosen time.</em></p>
    </div>"""
    else:
        rec = """
    <div style='background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 5px solid #28a745; margin-bottom: 15px;'>
        <h3 style='color: #28a745; margin: 0;'>✅ YOUR TIME IS ALREADY THE BEST</h3>
        <p style='margin: 5px 0; color: #333;'>No less-crowded departure was predicted within 3 hours of your chosen time.</p>
    </div>"""

    note = "" if predictor.beats_baseline else "<p style='color:#b8860b; font-size:12px;'>⚠️ The model does not currently beat the simple-average baseline; treat forecasts with caution.</p>"
    return rec + f"""
    <div style='background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 5px solid {sc};'>
        <h3 style='color: {sc}; margin: 0;'>SELECTED DEPARTURE OPTION</h3>
        <p style='margin: 5px 0; color: #333;'><strong style='color:#333;'>{route_name} at {hour_val:02d}:00</strong></p>
        <p style='margin: 2px 0; color: #333;'>Expected crowd level: <strong style='color: {sc};'>{se} {sel_risk} ({sel_occ}% occupancy)</strong></p>
        <p style='margin: 2px 0; color: #333;'>Estimated passengers: {int(expected)} | Capacity: {int(cap)}</p>
    </div>{note}"""

def citizen_best_travel_times(route, date_val):
    df = engine.df_featured
    routes = get_choices("route")
    if not routes:
        return "No data loaded."
    route_name = route if route in routes else routes[0]
    dow, cap = _dow_from(date_val), route_capacity(route_name)

    rows = []
    for h in sorted(int(x) for x in df["hour"].unique()):
        occ, risk = TransitDecisionCore.calculate_occupancy(forecast_demand(route_name, h, dow), cap)
        rows.append((h, occ, risk))

    md = f"### Predicted crowd levels: {route_name}\n"
    for h, occ, risk in rows:
        md += f"* **{h:02d}:00** {RISK_STYLE[risk][0]} {risk} ({occ}% occupancy)\n"
    best = sorted(sorted(rows, key=lambda r: r[1])[:3])
    md += "\n⭐ **Least crowded hours**: " + ", ".join(f"{h:02d}:00 ({occ}%)" for h, occ, _ in best)
    if len(rows) == 1:
        md += "\n\n⚠️ The dataset has no time-of-day information, so hourly comparison isn't possible."
    return md

def render_peak_vs_offpeak_chart():
    cap = 100
    demand_range = np.arange(0, 160, 5)
    peak_records = []
    off_peak_records = []
    for d in demand_range:
        p_alloc = TransitDecisionCore.optimize_allocation(d, cap, is_peak_hour=True)
        op_alloc = TransitDecisionCore.optimize_allocation(d, cap, is_peak_hour=False)
        peak_records.append({
            "Demand": d,
            "Freq_Adjustment": p_alloc["freq_adjustment_pct"],
            "Notes": p_alloc["notes"]
        })
        off_peak_records.append({
            "Demand": d,
            "Freq_Adjustment": op_alloc["freq_adjustment_pct"],
            "Notes": op_alloc["notes"]
        })
    df_p = pd.DataFrame(peak_records)
    df_op = pd.DataFrame(off_peak_records)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_p["Demand"],
        y=df_p["Freq_Adjustment"],
        mode='lines+markers',
        name='Peak-Hour Freq Adjustment (%)',
        line=dict(color='#d9534f', width=3),
        hovertemplate="<b>Peak Hour</b><br>Demand: %{x}<br>Freq Mod: %{y}%<br>%{customdata}<extra></extra>",
        customdata=df_p["Notes"]
    ))
    fig.add_trace(go.Scatter(
        x=df_op["Demand"],
        y=df_op["Freq_Adjustment"],
        mode='lines+markers',
        name='Off-Peak Freq Adjustment (%)',
        line=dict(color='#5bc0de', width=2, dash='dash'),
        hovertemplate="<b>Off-Peak Hour</b><br>Demand: %{x}<br>Freq Mod: %{y}%<br>%{customdata}<extra></extra>",
        customdata=df_op["Notes"]
    ))
    fig.add_shape(
        type="line",
        x0=cap, y0=-30, x1=cap, y1=110,
        line=dict(color="#292b2c", width=2, dash="dot")
    )
    fig.add_vrect(
        x0=75, x1=100,
        fillcolor="#f0ad4e", opacity=0.15,
        layer="below", line_width=0,
        annotation_text="Pre-emptive Peak Boost Window (75%-100% Demand)",
        annotation_position="top left"
    )
    fig.update_layout(
        title="Interactive Optimization Impact: Peak vs Off-Peak Allocation Profiles",
        xaxis_title="Simulated Passenger Demand (Ridership)",
        yaxis_title="Target Frequency Adjustment (% of Baseline)",
        template="plotly_white",
        hovermode="x unified"
    )
    return fig

def load_demo_and_update():
    """Resets the platform to the synthetic demo dataset."""
    global engine, predictor
    new_engine, new_predictor = build_pipeline(TransitDataEngine.generate_demo_dataset())
    new_engine.is_demo = True
    engine, predictor = new_engine, new_predictor

def _dropdown_lists():
    modes = ["All modes"] + get_choices("mode")
    return get_choices("station"), get_choices("route"), modes, (get_choices("weather") or ["N/A"])

STATIONS, ROUTES, MODES, WEATHERS = _dropdown_lists()

def refresh_choices():
    """Re-populates every dropdown from the currently loaded dataset."""
    st, ro, mo, we = _dropdown_lists()
    st0 = st[0] if st else None
    st1 = st[1] if len(st) > 1 else st0
    ro0 = ro[0] if ro else None
    return (
        gr.update(choices=st, value=st0), gr.update(choices=st, value=st1),
        gr.update(choices=mo, value=mo[0]), gr.update(choices=ro, value=ro0),
        gr.update(choices=ro, value=ro0), gr.update(choices=we, value=we[0]),
        gr.update(value=st0 or ""), gr.update(value=ro0 or ""),
    )

with gr.Blocks() as demo:
    # Explicit styles with high contrast slate/navy palette and dark font configurations to guarantee readability
    gr.HTML("""
    <div style='background: linear-gradient(135deg, #102a43, #243b53); color: #ffffff; padding: 25px; border-radius: 12px; margin-bottom: 25px; border: 2px solid #102a43;'>
        <h1 style='margin: 0; font-size: 28px; color: #ffffff; text-shadow: 1px 1px 3px rgba(0,0,0,0.6);'>AI Transit Demand & Overcrowding Intelligence</h1>
        <p style='margin: 8px 0 0 0; color: #e1e8ed; font-size: 16px; font-weight: 500;'>State-of-the-art decision-support & predictive engine for municipal transit systems</p>
    </div>
    """)

    with gr.Tabs() as main_tabs:
        # ==================== CITIZEN MODE ====================
        with gr.Tab("🚲 CITIZEN MODE (Default Landing)"):
            gr.HTML("""
            <div style='border: 1px solid #bcccdc; padding: 20px; border-radius: 8px; background-color: #f0f4f8; margin-bottom: 15px;'>
                <h2 style='margin-top: 0; color: #102a43;'>🚍 SMART TRANSIT</h2>
                <p style='font-size: 16px; color: #102a43; font-weight: bold;'>Know the crowd before you travel.</p>
                <p style='font-size: 14px; color: #102a43;'>Use predicted passenger demand to find less crowded travel options and help identify overloaded routes.</p>
                <p style='color: #627d98; font-size: 12px;'>⚠️ <em style='color:#555;'>DEMO DATA — NOT LIVE TRANSPORT INFORMATION (Predictions are estimates based on available historical data and may differ from actual conditions.)</em></p>
            </div>
            """)

            with gr.Tabs():
                # 1. FIND MY BEST JOURNEY
                with gr.Tab("📍 Find My Best Journey"):
                    with gr.Row():
                        with gr.Column():
                            from_st = gr.Dropdown(choices=STATIONS, label="From (Select station / stop)", value=STATIONS[0] if STATIONS else None)
                            to_st = gr.Dropdown(choices=STATIONS, label="To (Select station / stop)", value=STATIONS[1] if len(STATIONS) > 1 else None)
                            j_date = gr.Textbox(label="Date", value=str(datetime.now().strftime("%Y-%m-%d")))
                            j_time = gr.Slider(minimum=0, maximum=23, step=1, label="Time of Departure (Hour)", value=18)
                            j_mode = gr.Dropdown(choices=MODES, label="Commute Type / Rolling Stock", value=MODES[0])
                            btn_journey = gr.Button("FIND BEST OPTION", variant="primary")
                        with gr.Column():
                            journey_output = gr.HTML(value="<p style='color:#102a43; font-weight: bold;'>Fill options and click Find Best Option to execute intelligence forecast.</p>")
                    btn_journey.click(citizen_find_journey, inputs=[from_st, to_st, j_date, j_time, j_mode], outputs=[journey_output])

                # 2. BEST TIME TO TRAVEL
                with gr.Tab("⏰ Best Time To Travel"):
                    with gr.Row():
                        with gr.Column():
                            b_route = gr.Dropdown(choices=ROUTES, label="Route / Line", value=ROUTES[0] if ROUTES else None)
                            b_date = gr.Textbox(label="Date", value=str(datetime.now().strftime("%Y-%m-%d")))
                            btn_best_time = gr.Button("ANALYSIS BEST TIME", variant="primary")
                        with gr.Column():
                            best_time_output = gr.Markdown("Click analysis to view historical demand profiles.")
                    btn_best_time.click(citizen_best_travel_times, inputs=[b_route, b_date], outputs=[best_time_output])

                # 3. CROWD LEVEL MAP
                with gr.Tab("🗺️ Check Crowd Levels"):
                    gr.Markdown("### Live Predicted Crowding Density (Heatmap Profile)\n*These crowd levels are predictions based on available transportation data and algorithms.*")
                    chart_heatmap_citizen = gr.Plot(label="Station x Hour Crowding Heatmap")
                    btn_heatmap_cit = gr.Button("Load / Refresh Crowd Heatmap Map", variant="secondary")
                    btn_heatmap_cit.click(render_pressure_heatmap, outputs=[chart_heatmap_citizen])

                # 4. CITIZEN REPORTING
                with gr.Tab("🚨 Report a Problem"):
                    gr.Markdown("### File a direct crowd or operational service report to transit authorities")
                    with gr.Row():
                        with gr.Column():
                            rep_loc = gr.Textbox(label="Location / Station Name", value="Dadar")
                            rep_route = gr.Textbox(label="Route/Station Involved", value="Central Line")
                            rep_type = gr.Dropdown(choices=["Overcrowding", "Bus unavailable", "Train overcrowded", "Long waiting time", "Service delay", "Service cancellation", "Poor station condition", "Accessibility issue", "Other"], label="Problem Type", value="Overcrowding")
                            rep_sev = gr.Dropdown(choices=["🟢 Low", "🟡 Moderate", "🟠 High", "🔴 Critical"], label="Severity Level", value="🔴 Critical")
                            rep_desc = gr.Textbox(label="Detailed Description", value="The platform is extremely busy during peak hour, leaving many unable to board.")
                            btn_submit_rep = gr.Button("SUBMIT REPORT", variant="primary")
                        with gr.Column():
                            report_status = gr.Textbox(label="Submission Status Log", value="Awaiting Submission...")
                    btn_submit_rep.click(submit_citizen_report, inputs=[rep_loc, rep_route, rep_type, rep_sev, rep_desc], outputs=[report_status])

        # ==================== AUTHORITY MODE ====================
        with gr.Tab("⚙️ AUTHORITY / ADMIN MODE"):
            gr.Markdown("### Advanced Platform Backend, Predictive Engine Diagnostics, and Fleet Operations")

            with gr.Tabs():
                # 1. EXECUTIVE OVERVIEW
                with gr.Tab("📊 Overview"):
                    with gr.Row():
                        kpi1 = gr.Number(label="Total Network Demand Volume", value=124000)
                        kpi2 = gr.Number(label="Avg Hourly Load", value=154)
                        kpi3 = gr.Textbox(label="Peak Hour System-Wide", value="18:00")
                    with gr.Row():
                        kpi4 = gr.Textbox(label="High-Demand Route Node", value="Central Line")
                        kpi5 = gr.Textbox(label="Maximum Risk Station", value="Dadar")
                        kpi6 = gr.Number(label="Overcapacity Incidents (Critical)", value=42)
                    with gr.Row():
                        btn_refresh = gr.Button("Sync & Refresh Dashboard Performance", variant="primary")
                        btn_load_demo = gr.Button("Load Demo Dataset", variant="secondary")
                    with gr.Row():
                        chart_trend = gr.Plot(label="Demand Development Chart")
                        chart_hours = gr.Plot(label="Hourly Profile Distribution")

                    def refresh_dashboard_kpis():
                        t, avg, pk, tr, ws, oc = get_kpi_indicators()
                        return t, avg, pk, tr, ws, oc, render_ridership_trend(), render_hourly_profile()

                    btn_refresh.click(refresh_dashboard_kpis, outputs=[kpi1, kpi2, kpi3, kpi4, kpi5, kpi6, chart_trend, chart_hours])
                    demo_evt = btn_load_demo.click(load_demo_and_update, outputs=[]).then(refresh_dashboard_kpis, outputs=[kpi1, kpi2, kpi3, kpi4, kpi5, kpi6, chart_trend, chart_hours])

                # 2. DATA INGESTION & PIPELINE
                with gr.Tab("📁 Import Data"):
                    with gr.Row():
                        with gr.Column():
                            file_input = gr.File(label="Upload CSV/XLSX File")
                            url_input = gr.Textbox(label="Dataset URL Input", placeholder="https://example.com/ridership.csv")
                            btn_upload = gr.Button("Run Preprocessing & Automatic Clean", variant="primary")
                        with gr.Column():
                            output_status = gr.Textbox(label="Pipeline Ingestion Status")
                            quality_table = gr.DataFrame(label="Data Quality Assessment Report")
                    preview_table = gr.DataFrame(label="Dataset Preview (First 20 Rows)")
                    upload_evt = btn_upload.click(process_uploaded_dataset, inputs=[file_input, url_input], outputs=[output_status, preview_table, quality_table])

                # 3. ADVANCED DEMAND ANALYSIS
                with gr.Tab("📈 Demand Analysis"):
                    with gr.Row():
                        chart_heatmap = gr.Plot(label="Ridership Density Profile Matrix")
                        chart_importance = gr.Plot(label="Feature Influence Map")
                    btn_exp_refresh = gr.Button("Generate Heatmaps & Explainability Graphs", variant="primary")
                    btn_exp_refresh.click(lambda: (render_pressure_heatmap(), render_feature_importance()), outputs=[chart_heatmap, chart_importance])

                # 4. DEMAND PREDICTIONS
                with gr.Tab("🤖 Demand Prediction"):
                    with gr.Row():
                        with gr.Column():
                            route_drop = gr.Dropdown(choices=ROUTES, label="Target Route Node", value=ROUTES[0] if ROUTES else None)
                            day_drop = gr.Dropdown(choices=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], label="Target Weekday", value="Monday")
                            hour_slider = gr.Slider(minimum=0, maximum=23, step=1, label="Hour of Departure", value=18)
                            prev_demand_input = gr.Number(label="Previous Hour Realized Demand", value=120)
                            weather_drop = gr.Dropdown(choices=WEATHERS, label="Weather Conditions", value=WEATHERS[0])
                            btn_predict = gr.Button("Analyze Predictive Risks", variant="primary")
                        with gr.Column():
                            out_pred = gr.Number(label="Expected Simulated Ridership")
                            out_interval = gr.Textbox(label="Approx. 95% Prediction Interval")
                            out_occ = gr.Textbox(label="Calculated Fleet Occupancy Stress")
                            out_rec = gr.Markdown(label="Generated Tactical Fleet Recommendations")
                    btn_predict.click(run_live_predictions, inputs=[route_drop, day_drop, hour_slider, prev_demand_input, weather_drop], outputs=[out_pred, out_interval, out_occ, out_rec])

                # 5. WHAT-IF SIMULATOR
                with gr.Tab("🔬 What-If Scenario Simulator"):
                    with gr.Row():
                        with gr.Column():
                            sim_capacity = gr.Slider(minimum=10, maximum=1000, step=10, label="Simulated Vehicle Capacity Limit", value=150)
                            sim_multiplier = gr.Slider(minimum=10, maximum=300, step=10, label="Simulated Ridership Load Factor (% of Normal)", value=120)
                            sim_target = gr.Slider(minimum=50, maximum=100, step=5, label="Operational Target Occupancy (% Limit)", value=85)
                            btn_simulate = gr.Button("Run Simulation Scenario", variant="primary")
                        with gr.Column():
                            sim_output = gr.Markdown("**Scenario results will render here after clicking Simulate.**")
                    btn_simulate.click(execute_what_if_analysis, inputs=[sim_capacity, sim_multiplier, sim_target], outputs=[sim_output])

                # 6. DYNAMIC ALLOCATION CHART
                with gr.Tab("📈 Allocation Strategy"):
                    gr.Markdown("### Allocation Profile Simulation (Peak vs Off-Peak Peak-Shaving)")
                    chart_allocation = gr.Plot(label="Allocation Profile Curves")
                    btn_alloc_refresh = gr.Button("Generate Simulation Graph", variant="primary")
                    btn_alloc_refresh.click(render_peak_vs_offpeak_chart, outputs=[chart_allocation])

                # 7. CITIZEN FEEDBACK HUB
                with gr.Tab("🚨 Citizen Reports"):
                    gr.Markdown("### Aggregated Real-time Problem Incident Reports from Citizens")
                    summary_block = gr.Markdown("No reports logged yet.")
                    reports_df_view = gr.DataFrame(label="Active Citizen Incident Log")
                    btn_refresh_reps = gr.Button("Refetch Operational Reports", variant="primary")
                    btn_refresh_reps.click(get_reports_summary, outputs=[summary_block, reports_df_view])

                # 8. RECS EXPORTS
                with gr.Tab("📥 Downloads"):
                    gr.Markdown("### Generate CSV reports based on current predictive capacity breaches")
                    btn_export_recs = gr.Button("Compile Mitigation CSV Report", variant="primary")
                    file_output = gr.File(label="Generated CSV Document for Download")
                    btn_export_recs.click(export_recommendations_csv, outputs=[file_output])

            gr.Markdown("""\n---\n### 🔄 Integrated Citizen-to-Authority Feedback Cycle\n```\n  TRANSPORT DATA ➔ DEMAND FORECASTS ➔ CROWD WARNINGS ➔ CITIZENS (Feedback via Reports)\n  ▲                                                                              │\n  └─────────────── DECISION ACTION (Fleet Allocation Adjustments) ◄──────────────┘\n```""")

with demo:
    # Refresh every dropdown + the dashboard whenever data changes or a visitor opens the page
    choice_targets = [from_st, to_st, j_mode, b_route, route_drop, weather_drop, rep_loc, rep_route]
    upload_evt.then(refresh_choices, outputs=choice_targets)
    upload_evt.then(refresh_dashboard_kpis, outputs=[kpi1, kpi2, kpi3, kpi4, kpi5, kpi6, chart_trend, chart_hours])
    demo_evt.then(refresh_choices, outputs=choice_targets)
    demo.load(refresh_choices, outputs=choice_targets)
    demo.load(refresh_dashboard_kpis, outputs=[kpi1, kpi2, kpi3, kpi4, kpi5, kpi6, chart_trend, chart_hours])

# Seed one sample citizen report so the authority view isn't empty
submit_citizen_report(
    location="Dadar",
    route_station="Central Line",
    problem_type="Overcrowding",
    severity="🔴 Critical",
    description="Dadar interchange is extremely crowded during the 18:00-21:00 rush. Passengers cannot board the Central Line slow locals.",
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.queue().launch(server_name="0.0.0.0", server_port=port)
