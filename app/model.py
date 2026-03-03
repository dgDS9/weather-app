import joblib
from pathlib import Path

MODEL_PATH = Path("models/xgboost_weather_model.joblib")
FEATURES_PATH = Path("models/feature_columns.joblib")

model = None
feature_columns = None

def load_model_artifacts():
    global model, feature_columns
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH.resolve()}")
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Feature list not found: {FEATURES_PATH.resolve()}")

    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURES_PATH)