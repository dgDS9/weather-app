import sys
import mlflow
from mlflow import MlflowClient

from config import (
    LATEST_MODEL_URI_PATH,
    LATEST_RUN_ID_PATH,
    MLFLOW_MODEL_ALIAS,
    MLFLOW_REGISTERED_MODEL_NAME,
)
from mlflow_utils import configure_mlflow

MIN_SKILL_VS_NAIVE = 0.02
MAX_MAE = 2.20
MAX_RMSE = 2.90


def run_quality_gate():
    configure_mlflow()
    client = MlflowClient()

    run_id = LATEST_RUN_ID_PATH.read_text(encoding="utf-8").strip()
    model_uri = LATEST_MODEL_URI_PATH.read_text(encoding="utf-8").strip()

    run = client.get_run(run_id)
    metrics = run.data.metrics

    mae = metrics["test_mae"]
    rmse = metrics["test_rmse"]
    skill = metrics["skill_vs_naive"]

    checks = {
        "MAE": mae <= MAX_MAE,
        "RMSE": rmse <= MAX_RMSE,
        "Skill vs Naive": skill >= MIN_SKILL_VS_NAIVE,
    }

    for name, passed in checks.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")

    passed = all(checks.values())

    with mlflow.start_run(run_id=run_id):
        mlflow.set_tag(
            "quality_gate",
            "PASSED" if passed else "FAILED",
        )

    if not passed:
        print("QUALITY GATE FAILED")
        return False

    registered = mlflow.register_model(
        model_uri=model_uri,
        name=MLFLOW_REGISTERED_MODEL_NAME,
    )

    client.set_model_version_tag(
        name=MLFLOW_REGISTERED_MODEL_NAME,
        version=registered.version,
        key="quality_gate",
        value="PASSED",
    )

    client.set_registered_model_alias(
        name=MLFLOW_REGISTERED_MODEL_NAME,
        alias=MLFLOW_MODEL_ALIAS,
        version=registered.version,
    )

    print(
        f"QUALITY GATE PASSED -> "
        f"{MLFLOW_REGISTERED_MODEL_NAME} v{registered.version} "
        f"is now @{MLFLOW_MODEL_ALIAS}"
    )
    return True


if __name__ == "__main__":
    try:
        sys.exit(0 if run_quality_gate() else 1)
    except Exception as exc:
        print(f"Quality Gate error: {exc}")
        sys.exit(1)
