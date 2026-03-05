import os
import requests
from dotenv import load_dotenv

# Lädt Variablen aus .env (wenn vorhanden)
load_dotenv()

# API-Key aus Environment holen (nicht hardcoden!)
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
if not OPENWEATHER_API_KEY:
    raise RuntimeError(
        "OPENWEATHER_API_KEY is not set. Put it in a .env file or environment variables."
    )

BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

def fetch_current_weather(lat: float, lon: float, units: str = "metric") -> dict:
    """
    Holt aktuelles Wetter von OpenWeather Current Weather API.
    Gibt das JSON (dict) zurück.
    """
    params = {
        "lat": lat,
        "lon": lon,
        "appid": OPENWEATHER_API_KEY,
        "units": units,  # wichtig: metric = °C statt Kelvin
    }

    r = requests.get(BASE_URL, params=params, timeout=10)
    r.raise_for_status()  # wirft Fehler bei 401/404/500
    return r.json()