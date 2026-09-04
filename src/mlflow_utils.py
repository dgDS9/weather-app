import mlflow
from config import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI

def configure_mlflow():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
