import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

from config import (
    ARTIFACT_DIR,
    EXTENDED_FEATURES,
    FEATURE_IMPORTANCE_PATH,
    LATEST_MODEL_URI_PATH,
    LATEST_RUN_ID_PATH,
    METRICS_PATH,
    TARGET,
    TEST_START,
)
from data_preparation import load_and_prepare_data
from feature_engineering import build_feature_table
from mlflow_utils import configure_mlflow


def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse


def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    configure_mlflow()

    run_id = LATEST_RUN_ID_PATH.read_text(encoding="utf-8").strip()
    model_uri = LATEST_MODEL_URI_PATH.read_text(encoding="utf-8").strip()

    model = mlflow.xgboost.load_model(model_uri)

    df = build_feature_table(load_and_prepare_data())
    test = df[df["dt_iso"] >= TEST_START][
        EXTENDED_FEATURES + [TARGET, "temp", "dt_iso"]
    ].dropna()

    y_test = test[TARGET]
    pred = model.predict(test[EXTENDED_FEATURES])

    mae, rmse = calculate_metrics(y_test, pred)

    naive_pred = test["temp"].to_numpy()
    naive_mae, naive_rmse = calculate_metrics(y_test, naive_pred)
    skill = 1 - (mae / naive_mae)

    metrics = pd.DataFrame([
        {
            "model": "Naive Persistence",
            "mae_test": naive_mae,
            "rmse_test": naive_rmse,
            "skill_vs_naive": 0.0,
        },
        {
            "model": "XGBoost Extended Tuned",
            "mae_test": mae,
            "rmse_test": rmse,
            "skill_vs_naive": skill,
        },
    ])
    metrics.to_csv(METRICS_PATH, index=False)

    importance = model.get_booster().get_score(importance_type="gain")
    fi = pd.DataFrame(
        importance.items(),
        columns=["feature", "importance"],
    ).sort_values("importance", ascending=False)
    fi.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    plot_path = ARTIFACT_DIR / "feature_importance.png"
    top = fi.head(15).sort_values("importance")
    plt.figure(figsize=(10, 6))
    plt.barh(top["feature"], top["importance"])
    plt.title("Top 15 Feature Importances")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics({
            "test_mae": float(mae),
            "test_rmse": float(rmse),
            "skill_vs_naive": float(skill),
            "naive_test_mae": float(naive_mae),
            "naive_test_rmse": float(naive_rmse),
        })
        mlflow.log_artifact(str(METRICS_PATH), artifact_path="evaluation")
        mlflow.log_artifact(str(FEATURE_IMPORTANCE_PATH), artifact_path="evaluation")
        mlflow.log_artifact(str(plot_path), artifact_path="evaluation")

    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
