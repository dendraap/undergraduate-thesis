import json
import pandas as pd
from pathlib import Path
from datetime import datetime

# =====================
# PATH
# =====================
DATA_PATH = Path(__file__).parent.parent / "data" / "dummy" / "full_data.json"

# =====================
# NORMALIZE COLUMN (ANTI ENCODING)
# =====================
def normalize_col(col: str) -> str:
    return (
        col.replace("Î¼", "μ")
           .replace("Â", "")
           .strip()
    )

# =====================
# COLUMN MAP
# =====================
POLLUTANT_MAP = {
    "pm25": {
        "actual": "pm2_5 (μg/m³)",
        "pred": "pm2_5 (μg/m³)_predicted",
        "ispu": "pm2_5 (μg/m³)_ispu",
        "category": "pm2_5 (μg/m³)_ispu_kategori",
        "unit": "µg/m³",
    },
    "pm10": {
        "actual": "pm10 (μg/m³)",
        "pred": "pm10 (μg/m³)_predicted",
        "ispu": "pm10 (μg/m³)_ispu",
        "category": "pm10 (μg/m³)_ispu_kategori",
        "unit": "µg/m³",
    },
    "co": {
        "actual": "carbon_monoxide (μg/m³)",
        "pred": "carbon_monoxide (μg/m³)_predicted",
        "ispu": "carbon_monoxide (μg/m³)_ispu",
        "category": "carbon_monoxide (μg/m³)_ispu_kategori",
        "unit": "µg/m³",
    },
    "no2": {
        "actual": "nitrogen_dioxide (μg/m³)",
        "pred": "nitrogen_dioxide (μg/m³)_predicted",
        "ispu": "nitrogen_dioxide (μg/m³)_ispu",
        "category": "nitrogen_dioxide (μg/m³)_ispu_kategori",
        "unit": "µg/m³",
    },
    "so2": {
        "actual": "sulphur_dioxide (μg/m³)",
        "pred": "sulphur_dioxide (μg/m³)_predicted",
        "ispu": "sulphur_dioxide (μg/m³)_ispu",
        "category": "sulphur_dioxide (μg/m³)_ispu_kategori",
        "unit": "µg/m³",
    },
    "o3": {
        "actual": "ozone (μg/m³)",
        "pred": "ozone (μg/m³)_predicted",
        "ispu": "ozone (μg/m³)_ispu",
        "category": "ozone (μg/m³)_ispu_kategori",
        "unit": "µg/m³",
    },
}

CATEGORY_COLUMNS = [v["category"] for v in POLLUTANT_MAP.values()]

# =====================
# WEATHER MAP
# =====================
WEATHER_MAP = {
    "temperature": {"col": "temperature_2m (°C)", "unit": "°C"},
    "humidity": {"col": "relative_humidity_2m (%)", "unit": "%"},
    "dew_point": {"col": "dew_point_2m (°C)", "unit": "°C"},
    "rain": {"col": "rain (mm)", "unit": "mm"},
    "pressure": {"col": "surface_pressure (hPa)", "unit": "hPa"},
    "wind_speed": {"col": "wind_speed_10m (km/h)", "unit": "km/h"},
    "wind_direction": {"col": "wind_direction_10m (°)", "unit": "°"},
    "radiation": {"col": "direct_radiation (W/m²)", "unit": "W/m²"},
}

# =====================
# LOAD DATA
# =====================
def load_df() -> pd.DataFrame:
    with open(DATA_PATH, encoding="utf-8", errors="replace") as f:
        raw = json.load(f)

    df = pd.DataFrame(raw)
    df.columns = [normalize_col(c) for c in df.columns]
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    for col in df.columns:
        if col == "timestamp":
            continue
        if col in CATEGORY_COLUMNS:
            df[col] = df[col].astype(str)
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.sort_values("timestamp")

# =====================
# HELPERS
# =====================
def daily_ispu(df, col):
    return (
        df[["timestamp", col]]
        .dropna()
        .set_index("timestamp")
        .resample("1D")
        .mean()
        .dropna()
        .reset_index()
    )
# =====================
# DELTA PERCENTAGE
# =====================
def calc_delta_percent(series):
    if len(series) < 2:
        return None
    prev = series[-2]
    last = series[-1]
    if prev == 0:
        return None
    return round(((last - prev) / prev) * 100, 2)


# =====================
# RECOMMENDATIONS
# =====================
def recommendations(category):
    if not isinstance(category, str):
        return ["No recommendation available"]

    c = category.lower()
    if c in ["baik", "sehat", "sedang"]:
        return ["Outdoor activities are safe", "Maintain healthy lifestyle"]
    if c == "tidak sehat":
        return ["Reduce outdoor activities", "Wear a mask"]
    return ["Avoid outdoor activities", "Wear a mask", "Use air purifier"]


# =====================
# WEATHER API BUILDER
# =====================
def build_weather_block(df):
    weather = {}
    for key, cfg in WEATHER_MAP.items():
        if cfg["col"] not in df.columns:
            continue
        s = df[cfg["col"]].dropna()
        if s.empty:
            continue

        spark = s.tail(24).round(2).tolist()
        weather[key] = {
            "value": round(float(s.iloc[-1]), 2),
            "unit": cfg["unit"],
            "sparkline": spark,
            "delta_percent": calc_delta_percent(spark),
            "window": "Hour"
        }
    return weather


# =====================
# POLLUTANT API BUILDER
# =====================
def build_pollutant_block(df, key):
    cfg = POLLUTANT_MAP[key]

    # ISPU (DAILY)
    ispu_daily = daily_ispu(df, cfg["ispu"])
    latest = ispu_daily.iloc[-1]
    category = df[cfg["category"]].dropna().iloc[-1]
    ispu_spark = ispu_daily[cfg["ispu"]].tail(7).round(1).tolist()
    ispu = {
        "value": round(float(latest[cfg["ispu"]]), 1),
        "category": category,
        "sparkline": ispu_spark,
        "delta_percent": calc_delta_percent(ispu_spark),
        "window": "Day"
    }

    # CONCENTRATION (HOURLY)
    conc = df[cfg["actual"]].dropna()
    consentration_spark = conc.tail(24).round(2).tolist()

    concentration = {
        "value": round(float(conc.iloc[-1]), 2),
        "unit": cfg.get("unit", ""),
        "sparkline": consentration_spark,
        "delta_percent": calc_delta_percent(consentration_spark),
        "window": "Hour"
    }


    # OVERVIEW (HOURLY, 30 DAYS)
    overview_df = df.tail(24 * 30)
    overview = {
        "window_days": 30,
        "actual": [
            {"t": t.isoformat(), "v": float(v)}
            for t, v in zip(overview_df["timestamp"], overview_df[cfg["actual"]])
        ],
        "prediction": [
            {"t": t.isoformat(), "v": float(v)}
            for t, v in zip(overview_df["timestamp"], overview_df[cfg["pred"]])
        ],
        "window": "Hour"

    }

    return {
        "ispu": ispu,
        "concentration": concentration,
        "overview": overview,
        "recommendations": recommendations(category),
    }

# =====================
# MAIN API
# =====================
def get_forecast(model: str):
    df = load_df()
    return {
        "model": model,
        "meta": {"generated_at": datetime.utcnow().isoformat()},
        "weather": build_weather_block(df),
        "pollutants": {
            k: build_pollutant_block(df, k)
            for k in POLLUTANT_MAP.keys()
        },
    }
