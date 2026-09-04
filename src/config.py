from pathlib import Path
import os
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "OpenWeatherDataKarlsruhe.csv"
MODEL_DIR = PROJECT_ROOT / "models"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

METRICS_PATH = ARTIFACT_DIR / "model_metrics.csv"
FEATURE_IMPORTANCE_PATH = ARTIFACT_DIR / "feature_importance.csv"
LATEST_RUN_ID_PATH = ARTIFACT_DIR / "latest_training_run_id.txt"
LATEST_MODEL_URI_PATH = ARTIFACT_DIR / "latest_model_uri.txt"

HORIZON_HOURS = 24
VALIDATION_START = pd.Timestamp("2018-01-01", tz="UTC")
TEST_START = pd.Timestamp("2020-01-01", tz="UTC")
RANDOM_STATE = 42

EXTENDED_FEATURES = [
    "temp_lag_1", "temp_lag_3", "temp_lag_6", "temp_lag_12", "temp_lag_24",
    "humidity_lag_6", "pressure_lag_6", "pressure_lag_24",
    "wind_speed_lag_6", "clouds_all_lag_6",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",
]

TARGET = "temp_true_t_plus_24"

MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{MLFLOW_DB_PATH.as_posix()}",
)
MLFLOW_EXPERIMENT_NAME = "weather-forecasting-karlsruhe"
MLFLOW_REGISTERED_MODEL_NAME = "karlsruhe-weather-xgboost"
MLFLOW_MODEL_ALIAS = "champion"
