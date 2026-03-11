```md

Not interested in a weather website that only predicts temperature for the next 24 hours and only for beautiful Karlsruhe?

Honestly… me neither 😅 .

And yet, here is the link to exactly that: https://lnkd.in/emHTSVRs

Small note: if the website has been inactive for a while, the initial loading time may take a bit longer. The server runs on a free hosting plan — that is the trade-off.

So why did I build a weather website with such a narrow use case and only limited features?

Because this project was never just about weather.

It was about exploring how much work it really takes to turn the forecast of an AI/ML model into something user-friendly and actually usable. Because building an AI model is only half the job — making it usable is the real challenge.

I built this project in my free time because I keep seeing the same issue in the industry: many people can develop AI or ML models, but far fewer can turn them into usable products or interfaces that help users actually understand and work with the results.

A model output on its own is often not very useful for end users. Without the right context, predictions can be difficult to interpret. That is why frontend applications and clear visualizations matter so much: they make AI outputs more accessible, understandable, and practical.

For this project, I worked with around 50 years of historical weather data, trained and evaluated multiple machine learning models, selected the best-performing one, built an API to process daily data from OpenWeather, and finally turned everything into a live website using JavaScript, CSS, and HTML.
The result is a small end-to-end project that combines machine learning, data processing, API integration, frontend development, and deployment — from model to live application.

Could I have expanded it further? Absolutely.
More forecast days, more weather variables, more features.
But at some point, you have to ship — especially when the next project idea is already waiting 😊 .

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


