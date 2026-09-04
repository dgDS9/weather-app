import pandas as pd
import mlflow.xgboost

from config import (
    EXTENDED_FEATURES,
    MLFLOW_MODEL_ALIAS,
    MLFLOW_REGISTERED_MODEL_NAME,
)
from mlflow_utils import configure_mlflow


def load_champion_model():
    configure_mlflow()
    model_uri = (
        f"models:/{MLFLOW_REGISTERED_MODEL_NAME}"
        f"@{MLFLOW_MODEL_ALIAS}"
    )
    return mlflow.xgboost.load_model(model_uri)


def predict_from_feature_dict(feature_values):
    missing = [
        feature for feature in EXTENDED_FEATURES
        if feature not in feature_values
    ]

    if missing:
        raise ValueError(f"Missing features: {missing}")

    model = load_champion_model()
    X = pd.DataFrame([feature_values], columns=EXTENDED_FEATURES)
    return float(model.predict(X)[0])
