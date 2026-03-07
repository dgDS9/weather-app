```md
# Weather Forecast (t+24h) — XGBoost + FastAPI + OpenWeather

This repository contains a small end-to-end project that turns a machine learning model into a usable web experience:

- A **FastAPI backend** loads an **XGBoost model** and predicts **temperature for t+24 hours**
- Live input signals come from **current weather data** (OpenWeather)
- A lightweight **static frontend** (HTML/CSS/JavaScript) visualizes the current temperature and the prediction

The focus is not only on model training, but on the full path to a working, deployable product (API + UI).

---

## What this project does

1. The backend fetches current weather conditions for a location.
2. Features are built from the current conditions (no history window needed).
3. The XGBoost model predicts the temperature for the next 24 hours.
4. The frontend calls the backend endpoint and displays the results.

---

## Repository structure (high level)

- `main.py`  
  FastAPI application entry point, routing and CORS config

- `app/`  
  Backend modules (OpenWeather client, feature creation, model loading, schemas)

- `models/`  
  Model artifacts (e.g., joblib files)

- `index.html`, `styles.css`, `app.js`  
  Static frontend (UI + fetch logic)

---

## API

### Endpoint: `GET /forecast-live-current`

Provides a “live” prediction based on current weather conditions.

- Optional query parameters:
  - `city` (string)
  - `lat` (float)
  - `lon` (float)

- Response fields (example):
  - `location`
  - `observation_time_utc`
  - `forecast_time_utc`
  - `temp_now_c`
  - `temp_pred_t_plus_24_c`
  - `delta_c`
  - `note`

---

## Local development (overview)

### Backend
- Create and activate a virtual environment
- Install dependencies from `requirements.txt`
- Provide required environment variables (see below)
- Run the FastAPI app with a local ASGI server (e.g., uvicorn)

### Frontend
- Serve the static files with any simple local static server
- Ensure the frontend points to the backend base URL you want to use
  (local for development, hosted URL for production)

---

## Configuration (important)

### Environment variables
This project uses environment variables for configuration and secrets.

Typical variables:
- `OPENWEATHER_API_KEY` (required)

**Do not commit secrets** (API keys, tokens, passwords) into the repository.
Use environment variables locally and in your hosting provider’s settings.

---

## Privacy & compliance notes

- This project is designed to work without cookies, tracking, or user accounts.
- No personal user data is intentionally collected or stored by the application.
- Requests to the backend are standard web requests; infrastructure providers may log technical metadata (e.g., timestamps, request path, and IP address) as part of normal operations.

---

## Notes on free-tier hosting behavior

Some free hosting plans can suspend idle services. In that case, the first request after inactivity may be slower.  
The frontend can include retry logic to handle this gracefully.


