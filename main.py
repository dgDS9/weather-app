from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import app.model as model_store
from app.openweather import fetch_current_weather
from app.features_no_history import build_features_no_history

app = FastAPI(title="Weather ML Forecast API (t+24, no-history)")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
    ],
    allow_credentials=False,   # <— wichtig
    allow_methods=["*"],
    allow_headers=["*"],
)


DEFAULT_LAT = 49.0096
DEFAULT_LON = 8.4053
DEFAULT_CITY = "Karlsruhe"

@app.on_event("startup")
def startup():
    model_store.load_model_artifacts()

@app.get("/forecast-live-current")
def forecast_live_current(lat: float = DEFAULT_LAT, lon: float = DEFAULT_LON, city: str = DEFAULT_CITY):
    current = fetch_current_weather(lat=lat, lon=lon)
    X, obs_dt, fc_dt, temp_now = build_features_no_history(current, model_store.feature_columns)

    pred = float(model_store.model.predict(X)[0])

    return {
        "location": city,
        "observation_time_utc": obs_dt.isoformat(),
        "forecast_time_utc": fc_dt.isoformat(),
        "temp_now_c": round(float(temp_now), 2),
        "temp_pred_t_plus_24_c": round(pred, 2),
        "delta_c": round(pred - float(temp_now), 2),
        "note": "Forecast based on OpenWeather current data + XGBoost (t+24h)"
    }