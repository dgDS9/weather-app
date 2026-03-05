# main.py - FastAPI application for weather forecasting (t+24h, no-history)
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware

import app.model as model_store
from app.openweather import fetch_current_weather
from app.features_no_history import build_features_no_history

# FastAPI application instance
app = FastAPI(title="Weather ML Forecast API (t+24, no-history)")

from fastapi.middleware.cors import CORSMiddleware

# please only englisch comments, because the code is public and may be used by people who do not understand german
# CORS middleware to allow requests from the frontend hosted on GitHub Pages, but disallow credentials for security reasons 
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://dgds9.github.io"
    ],
    allow_credentials=False,   # <— wichtig
    allow_methods=["*"],
    allow_headers=["*"],
)

# Default location (Karlsruhe) for the forecast endpoint, can be overridden by query parameters
DEFAULT_LAT = 49.0096
DEFAULT_LON = 8.4053
DEFAULT_CITY = "Karlsruhe"

# Load model artifacts at startup to avoid loading them on every request, which would be inefficient
@app.on_event("startup")
def startup():
    model_store.load_model_artifacts()


# HEAD request for root endpoint to avoid 405 error, because some monitoring tools might use HEAD requests to check if the service is alive
@app.head("/")
def head_root():
    return Response(status_code=200)


# GET request for root endpoint to provide a simple status check and service information
@app.get("/")
def root():
    return {"status": "ok", "service": "weather-api"}

# Favicon endpoint to prevent 404 errors in browser when accessing the root URL, returns a 204 No Content response
@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)  # No Content, because we don't have a favicon, but this prevents 404 errors in the browser when accessing the root URL

# Forecast endpoint that takes latitude, longitude, and city as query parameters (with defaults) and returns the current temperature and the predicted temperature for t+24h using the XGBoost model. It also calculates the delta between the current temperature and the predicted temperature.
@app.get("/forecast-live-current")
def forecast_live_current(lat: float = DEFAULT_LAT, lon: float = DEFAULT_LON, city: str = DEFAULT_CITY):
    current = fetch_current_weather(lat=lat, lon=lon)
    X, obs_dt, fc_dt, temp_now = build_features_no_history(current, model_store.feature_columns)
    # The model's predict method returns an array, so we take the first element and convert it to a float for the predicted temperature
    pred = float(model_store.model.predict(X)[0])
    # We return a JSON response with the location, observation time, forecast time, current temperature, predicted temperature for t+24h, the delta between the two temperatures, and a note about how the forecast was generated. The temperatures are rounded to 2 decimal places for better readability.
    return {
        "location": city,
        "observation_time_utc": obs_dt.isoformat(),
        "forecast_time_utc": fc_dt.isoformat(),
        "temp_now_c": round(float(temp_now), 2),
        "temp_pred_t_plus_24_c": round(pred, 2),
        "delta_c": round(pred - float(temp_now), 2),
        "note": "Forecast based on OpenWeather current data + XGBoost (t+24h)"
    }