import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

def build_features_no_history(current_json: dict, feature_columns: list) -> tuple[pd.DataFrame, datetime, datetime, float]:
    """
    Ersetzt alle Lag-Features bewusst mit dem aktuellen Messwert (No-History / Persistence-Fallback).
    Berechnet Zeitfeatures für t+24.
    """
    # --- Zeitpunkt der Beobachtung (UTC) ---
    obs_dt_utc = datetime.fromtimestamp(current_json["dt"], tz=timezone.utc)
    forecast_dt_utc = obs_dt_utc + timedelta(hours=24)

    # --- Messwerte (metric) ---
    main = current_json.get("main", {})
    wind = current_json.get("wind", {})
    clouds = current_json.get("clouds", {})

    temp = float(main["temp"])          # °C (wegen units=metric)
    pressure = float(main["pressure"])  # hPa
    humidity = float(main["humidity"])  # %
    wind_speed = float(wind.get("speed", 0.0))
    clouds_all = float(clouds.get("all", 0.0))

    # --- Zeitfeatures für den Vorhersagezeitpunkt t+24 ---
    hour = forecast_dt_utc.hour
    doy = forecast_dt_utc.timetuple().tm_yday

    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    doy_sin  = np.sin(2 * np.pi * doy / 365)
    doy_cos  = np.cos(2 * np.pi * doy / 365)

    # --- Feature-Map (Lags = aktueller Wert) ---
    feat = {
        # Temperatur-Lags
        "temp_lag_1": temp,
        "temp_lag_3": temp,
        "temp_lag_6": temp,
        "temp_lag_12": temp,
        "temp_lag_24": temp,

        # Wetter-Lags (alles current)
        "humidity_lag_6": humidity,
        "pressure_lag_6": pressure,
        "pressure_lag_24": pressure,
        "wind_speed_lag_6": wind_speed,
        "clouds_all_lag_6": clouds_all,

        # Zeit
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "doy_sin": doy_sin,
        "doy_cos": doy_cos,
    }

    # --- Safety check: Feature Columns müssen passen ---
    missing = [c for c in feature_columns if c not in feat]
    if missing:
        # Beispiel: dew_point_lag_6 noch in feature_columns gespeichert → dann passt dein Modell nicht zur kostenlosen API
        raise ValueError(
            f"Model expects features not available in no-history setup: {missing}. "
            f"Fix: retrain model without these features OR change feature_columns."
        )

    X = pd.DataFrame([feat], columns=feature_columns)
    return X, obs_dt_utc, forecast_dt_utc, temp