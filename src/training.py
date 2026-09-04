import itertools
import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost

from mlflow.models import infer_signature
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

from config import (
    ARTIFACT_DIR,
    EXTENDED_FEATURES,
    LATEST_MODEL_URI_PATH,
    LATEST_RUN_ID_PATH,
    RANDOM_STATE,
    TARGET,
    TEST_START,
    VALIDATION_START,
)
from data_preparation import load_and_prepare_data
from feature_engineering import build_feature_table
from mlflow_utils import configure_mlflow


def temporal_split(df):
    train = df[df["dt_iso"] < VALIDATION_START].copy()
    validation = df[
        (df["dt_iso"] >= VALIDATION_START) &
        (df["dt_iso"] < TEST_START)
    ].copy()
    test = df[df["dt_iso"] >= TEST_START].copy()
    return train, validation, test


def tune_xgboost(df):
    data = df[EXTENDED_FEATURES + [TARGET, "dt_iso"]].dropna()
    train, validation, _ = temporal_split(data)

    X_train = train[EXTENDED_FEATURES]
    y_train = train[TARGET]
    X_val = validation[EXTENDED_FEATURES]
    y_val = validation[TARGET]

    param_grid = {
        "max_depth": [4, 6],
        "learning_rate": [0.05],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
        "min_child_weight": [1],
    }

    best_mae = np.inf
    best_params = None

    for i, values in enumerate(
        itertools.product(*param_grid.values()), start=1
    ):
        params = dict(zip(param_grid.keys(), values))

        with mlflow.start_run(run_name=f"tuning-{i:03d}", nested=True):
            mlflow.log_params(params)

            model = XGBRegressor(
                n_estimators=500,
                objective="reg:squarederror",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                **params,
            )
            model.fit(X_train, y_train)

            pred = model.predict(X_val)
            val_mae = mean_absolute_error(y_val, pred)
            mlflow.log_metric("validation_mae", val_mae)

            if val_mae < best_mae:
                best_mae = val_mae
                best_params = params

    return best_params, best_mae, train, validation


def train_and_log_final_model():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    configure_mlflow()

    df = build_feature_table(load_and_prepare_data())

    with mlflow.start_run(run_name="xgboost-final-training") as run:
        mlflow.set_tags({
            "project": "weather-app",
            "city": "Karlsruhe",
            "forecast_horizon_hours": "24",
            "model_type": "XGBoostRegressor",
        })

        best_params, best_val_mae, train, validation = tune_xgboost(df)

        trainval = pd.concat([train, validation]).sort_values("dt_iso")
        X_trainval = trainval[EXTENDED_FEATURES]
        y_trainval = trainval[TARGET]

        model = XGBRegressor(
            n_estimators=500,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            **best_params,
        )
        model.fit(X_trainval, y_trainval)

        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metric("best_validation_mae", best_val_mae)

        signature = infer_signature(
            X_trainval,
            model.predict(X_trainval),
        )

        model_info = mlflow.xgboost.log_model(
            xgb_model=model,
            name="model",
            signature=signature,
            input_example=X_trainval.head(5),
            model_format="json",
        )

        LATEST_RUN_ID_PATH.write_text(run.info.run_id, encoding="utf-8")
        LATEST_MODEL_URI_PATH.write_text(
            model_info.model_uri,
            encoding="utf-8",
        )

        mlflow.log_dict(
            {"feature_columns": EXTENDED_FEATURES},
            "metadata/feature_columns.json",
        )

        print(f"Run ID: {run.info.run_id}")
        print(f"Model URI: {model_info.model_uri}")
        print(f"Best validation MAE: {best_val_mae:.3f} °C")


if __name__ == "__main__":
    train_and_log_final_model()
