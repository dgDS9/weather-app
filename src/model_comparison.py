"""
Compare candidate forecasting models on the validation period and log the
comparison to MLflow.

Purpose
-------
This script is for experimentation / model selection, NOT for production
retraining. It compares the candidate model classes from the original notebook:

- Naive persistence baseline
- Linear Regression
- XGBoost (basic)
- Vanilla RNN
- LSTM
- 1D CNN

Important methodology
---------------------
The original notebook compared models on the test period. In this production-
oriented version, model selection is done on VALIDATION data. The TEST period
remains untouched for the final evaluation in evaluation.py.

The winning model is NOT registered here. Registration remains the job of the
production training + evaluation + quality-gate flow.
"""

from pathlib import Path

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from config import (
    ARTIFACT_DIR,
    RANDOM_STATE,
    TARGET,
)
from data_preparation import load_and_prepare_data
from feature_engineering import build_feature_table
from mlflow_utils import configure_mlflow
from training import temporal_split


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEQUENCE_LENGTH = 24
EPOCHS = 20
BATCH_SIZE = 64

# Same compact feature set used by LR / RNN / LSTM / CNN in the notebook.
# We deliberately use the same features for the model-class comparison so
# that the comparison is primarily about the algorithm, not extra features.
COMPARISON_FEATURES = [
    "temp_lag_1",
    "temp_lag_3",
    "temp_lag_24",
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
]

MODEL_COMPARISON_PATH = ARTIFACT_DIR / "model_comparison.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def calculate_metrics(y_true, y_pred):
    """Return MAE and RMSE."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return float(mae), float(rmse)


def calculate_skill(mae_model, mae_naive):
    """
    Skill Score versus naive persistence.

    > 0  => better than naive
    = 0  => same as naive
    < 0  => worse than naive
    """
    return float(1 - (mae_model / mae_naive))


def make_sequences(X, y, seq_len=SEQUENCE_LENGTH):
    """Convert tabular hourly rows into rolling sequences for neural networks."""
    X_seq, y_seq = [], []

    for i in range(seq_len, len(X)):
        X_seq.append(X[i - seq_len:i])
        y_seq.append(y[i])

    return np.asarray(X_seq), np.asarray(y_seq)


def log_metrics_to_mlflow(mae, rmse, skill):
    mlflow.log_metrics(
        {
            "validation_mae": mae,
            "validation_rmse": rmse,
            "skill_vs_naive": skill,
        }
    )


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------

def evaluate_naive(validation):
    """
    Naive persistence:
    temperature in 24 h = current temperature.

    To make the comparison fair with sequence models, the first 24 validation
    rows are excluded for ALL models.
    """
    validation_eval = validation.iloc[SEQUENCE_LENGTH:].copy()

    y_true = validation_eval[TARGET].to_numpy()
    y_pred = validation_eval["temp"].to_numpy()

    mae, rmse = calculate_metrics(y_true, y_pred)

    return {
        "model": "Naive Persistence",
        "validation_mae": mae,
        "validation_rmse": rmse,
        "skill_vs_naive": 0.0,
    }


# ---------------------------------------------------------------------------
# Linear Regression
# ---------------------------------------------------------------------------

def evaluate_linear_regression(train, validation, naive_mae):
    X_train = train[COMPARISON_FEATURES]
    y_train = train[TARGET]

    validation_eval = validation.iloc[SEQUENCE_LENGTH:].copy()
    X_val = validation_eval[COMPARISON_FEATURES]
    y_val = validation_eval[TARGET]

    with mlflow.start_run(
        run_name="comparison-linear-regression",
        nested=True,
    ):
        mlflow.set_tag("phase", "model_comparison")
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("feature_count", len(COMPARISON_FEATURES))

        model = LinearRegression()
        model.fit(X_train, y_train)

        pred = model.predict(X_val)

        mae, rmse = calculate_metrics(y_val, pred)
        skill = calculate_skill(mae, naive_mae)

        log_metrics_to_mlflow(mae, rmse, skill)

        mlflow.sklearn.log_model(
            sk_model=model,
            name="candidate_model",
            input_example=X_train.head(5),
        )

    return {
        "model": "Linear Regression",
        "validation_mae": mae,
        "validation_rmse": rmse,
        "skill_vs_naive": skill,
    }


# ---------------------------------------------------------------------------
# XGBoost basic
# ---------------------------------------------------------------------------

def evaluate_xgboost(train, validation, naive_mae):
    X_train = train[COMPARISON_FEATURES]
    y_train = train[TARGET]

    validation_eval = validation.iloc[SEQUENCE_LENGTH:].copy()
    X_val = validation_eval[COMPARISON_FEATURES]
    y_val = validation_eval[TARGET]

    params = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "reg:squarederror",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }

    with mlflow.start_run(
        run_name="comparison-xgboost-basic",
        nested=True,
    ):
        mlflow.set_tag("phase", "model_comparison")
        mlflow.log_param("model_type", "XGBRegressor")
        mlflow.log_param("feature_count", len(COMPARISON_FEATURES))
        mlflow.log_params(params)

        model = XGBRegressor(**params)
        model.fit(X_train, y_train)

        pred = model.predict(X_val)

        mae, rmse = calculate_metrics(y_val, pred)
        skill = calculate_skill(mae, naive_mae)

        log_metrics_to_mlflow(mae, rmse, skill)

        mlflow.xgboost.log_model(
            xgb_model=model,
            name="candidate_model",
            input_example=X_train.head(5),
            model_format="json",
        )

    return {
        "model": "XGBoost Basic",
        "validation_mae": mae,
        "validation_rmse": rmse,
        "skill_vs_naive": skill,
    }


# ---------------------------------------------------------------------------
# Deep-learning preparation
# ---------------------------------------------------------------------------

def prepare_sequence_data(train, validation):
    """
    Fit scaler ONLY on training data, then transform validation data.
    This prevents information leakage from validation into preprocessing.
    """
    X_train_raw = train[COMPARISON_FEATURES].to_numpy()
    y_train = train[TARGET].to_numpy()

    X_val_raw = validation[COMPARISON_FEATURES].to_numpy()
    y_val = validation[TARGET].to_numpy()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)

    X_train_seq, y_train_seq = make_sequences(
        X_train_scaled,
        y_train,
    )

    X_val_seq, y_val_seq = make_sequences(
        X_val_scaled,
        y_val,
    )

    return X_train_seq, y_train_seq, X_val_seq, y_val_seq


def evaluate_keras_model(
    model_name,
    build_model,
    train,
    validation,
    naive_mae,
):
    """
    Shared evaluation routine for RNN, LSTM and CNN.
    """
    import tensorflow as tf
    import mlflow.tensorflow

    tf.keras.utils.set_random_seed(RANDOM_STATE)

    X_train_seq, y_train_seq, X_val_seq, y_val_seq = (
        prepare_sequence_data(train, validation)
    )

    with mlflow.start_run(
        run_name=f"comparison-{model_name.lower().replace(' ', '-')}",
        nested=True,
    ):
        mlflow.set_tag("phase", "model_comparison")
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("sequence_length", SEQUENCE_LENGTH)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("feature_count", len(COMPARISON_FEATURES))

        model = build_model(X_train_seq.shape[2])

        history = model.fit(
            X_train_seq,
            y_train_seq,
            validation_data=(X_val_seq, y_val_seq),
            shuffle=False,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=1,
        )

        pred = model.predict(X_val_seq, verbose=0).flatten()

        mae, rmse = calculate_metrics(y_val_seq, pred)
        skill = calculate_skill(mae, naive_mae)

        log_metrics_to_mlflow(mae, rmse, skill)

        # Last epoch's loss values are useful diagnostics.
        mlflow.log_metric(
            "final_train_loss",
            float(history.history["loss"][-1]),
        )
        mlflow.log_metric(
            "final_validation_loss",
            float(history.history["val_loss"][-1]),
        )

        mlflow.tensorflow.log_model(
            model=model,
            name="candidate_model",
            input_example=X_train_seq[:2],
        )

    return {
        "model": model_name,
        "validation_mae": mae,
        "validation_rmse": rmse,
        "skill_vs_naive": skill,
    }


def build_rnn(n_features):
    import tensorflow as tf

    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(
                shape=(SEQUENCE_LENGTH, n_features)
            ),
            tf.keras.layers.SimpleRNN(
                32,
                activation="tanh",
            ),
            tf.keras.layers.Dense(1),
        ]
    )


def build_lstm(n_features):
    import tensorflow as tf

    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(
                shape=(SEQUENCE_LENGTH, n_features)
            ),
            tf.keras.layers.LSTM(
                units=64,
                activation="tanh",
                recurrent_activation="sigmoid",
            ),
            tf.keras.layers.Dense(1),
        ]
    )


def build_cnn(n_features):
    import tensorflow as tf

    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(
                shape=(SEQUENCE_LENGTH, n_features)
            ),
            tf.keras.layers.Conv1D(
                filters=32,
                kernel_size=3,
                activation="relu",
                padding="causal",
            ),
            tf.keras.layers.Conv1D(
                filters=32,
                kernel_size=3,
                activation="relu",
                padding="causal",
            ),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(1),
        ]
    )


def compile_keras_model(model):
    import tensorflow as tf

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="mse",
    )

    return model


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------

def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    configure_mlflow()

    df = build_feature_table(
        load_and_prepare_data()
    )

    # Use only rows needed by the comparison.
    data = df[
        COMPARISON_FEATURES
        + [TARGET, "temp", "dt_iso"]
    ].dropna().copy()

    train, validation, _ = temporal_split(data)

    if len(validation) <= SEQUENCE_LENGTH:
        raise ValueError(
            "Validation period is too small for "
            f"SEQUENCE_LENGTH={SEQUENCE_LENGTH}."
        )

    results = []

    with mlflow.start_run(
        run_name="model-comparison",
    ) as parent_run:
        mlflow.set_tags(
            {
                "phase": "model_comparison",
                "selection_dataset": "validation",
                "test_set_used": "false",
            }
        )

        # --------------------------------------------------------------
        # 1) Naive baseline
        # --------------------------------------------------------------
        naive_result = evaluate_naive(validation)
        results.append(naive_result)

        naive_mae = naive_result["validation_mae"]

        with mlflow.start_run(
            run_name="comparison-naive-persistence",
            nested=True,
        ):
            mlflow.set_tag("phase", "model_comparison")
            mlflow.log_param(
                "model_type",
                "NaivePersistence",
            )
            log_metrics_to_mlflow(
                naive_result["validation_mae"],
                naive_result["validation_rmse"],
                0.0,
            )

        # --------------------------------------------------------------
        # 2) Linear Regression
        # --------------------------------------------------------------
        results.append(
            evaluate_linear_regression(
                train,
                validation,
                naive_mae,
            )
        )

        # --------------------------------------------------------------
        # 3) XGBoost basic
        # --------------------------------------------------------------
        results.append(
            evaluate_xgboost(
                train,
                validation,
                naive_mae,
            )
        )

        # --------------------------------------------------------------
        # 4) Vanilla RNN
        # --------------------------------------------------------------
        results.append(
            evaluate_keras_model(
                model_name="Vanilla RNN",
                build_model=lambda n: compile_keras_model(
                    build_rnn(n)
                ),
                train=train,
                validation=validation,
                naive_mae=naive_mae,
            )
        )

        # --------------------------------------------------------------
        # 5) LSTM
        # --------------------------------------------------------------
        results.append(
            evaluate_keras_model(
                model_name="LSTM",
                build_model=lambda n: compile_keras_model(
                    build_lstm(n)
                ),
                train=train,
                validation=validation,
                naive_mae=naive_mae,
            )
        )

        # --------------------------------------------------------------
        # 6) 1D CNN
        # --------------------------------------------------------------
        results.append(
            evaluate_keras_model(
                model_name="CNN 1D",
                build_model=lambda n: compile_keras_model(
                    build_cnn(n)
                ),
                train=train,
                validation=validation,
                naive_mae=naive_mae,
            )
        )

        comparison = (
            pd.DataFrame(results)
            .sort_values(
                "validation_mae",
                ascending=True,
            )
            .reset_index(drop=True)
        )

        comparison["rank"] = (
            comparison["validation_mae"]
            .rank(method="min")
            .astype(int)
        )

        comparison = comparison[
            [
                "rank",
                "model",
                "validation_mae",
                "validation_rmse",
                "skill_vs_naive",
            ]
        ]

        comparison.to_csv(
            MODEL_COMPARISON_PATH,
            index=False,
        )

        winner = comparison.iloc[0]

        mlflow.log_artifact(
            str(MODEL_COMPARISON_PATH),
            artifact_path="comparison",
        )
        mlflow.log_param(
            "selected_model_class",
            winner["model"],
        )
        mlflow.log_metric(
            "best_validation_mae",
            float(winner["validation_mae"]),
        )

        print("\n==============================================")
        print("MODEL COMPARISON - VALIDATION SET")
        print("==============================================")
        print(comparison.to_string(index=False))

        print("\nSelected candidate:")
        print(
            f"{winner['model']} "
            f"(MAE={winner['validation_mae']:.3f} °C)"
        )

        print(
            "\nImportant: this script does NOT register the winner. "
            "The production training/evaluation/quality-gate flow "
            "does that later."
        )

        print(
            f"\nMLflow parent run: {parent_run.info.run_id}"
        )
        print(
            f"Saved comparison: {MODEL_COMPARISON_PATH}"
        )


if __name__ == "__main__":
    main()
