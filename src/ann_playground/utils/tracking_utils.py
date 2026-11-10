from typing import Any, Dict, List, Optional
import os
import mlflow
import mlflow.keras
from mlflow.entities import Experiment
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from ann_playground.config import MLFLOW_TRACKING_URI


def get_tracking_uri() -> str:
    """
    Resolve active MLflow tracking URI from environment or project configuration.

    Returns
    -------
    str
        Validated URI pointing to the MLflow tracking server or database.
    """
    return os.getenv("MLFLOW_TRACKING_URI", str(MLFLOW_TRACKING_URI))


def init_mlflow_experiment(
    experiment_name: str,
    tracking_uri: Optional[str] = None,
) -> str:
    """
    Safely retrieve, create, or restore an MLflow experiment.

    Parameters
    ----------
    experiment_name : str
        Target experiment name.
    tracking_uri : Optional[str], default=None
        Custom tracking server URI. If None, resolves via get_tracking_uri().

    Returns
    -------
    str
        Identifier of the active MLflow experiment.
    """
    uri: str = tracking_uri or get_tracking_uri()
    mlflow.set_tracking_uri(uri)
    client: MlflowClient = MlflowClient(tracking_uri=uri)

    try:
        experiment: Optional[Experiment] = client.get_experiment_by_name(experiment_name)
    except MlflowException:
        experiment = None

    if experiment is None:
        try:
            experiment_id: str = client.create_experiment(experiment_name)
        except MlflowException:
            # Race condition fallback if created concurrently
            exp = client.get_experiment_by_name(experiment_name)
            experiment_id = exp.experiment_id if exp else "0"
    elif experiment.lifecycle_stage == "deleted":
        client.restore_experiment(experiment.experiment_id)
        experiment_id = experiment.experiment_id
    else:
        experiment_id = experiment.experiment_id

    mlflow.set_experiment(experiment_name=experiment_name)
    return experiment_id


def log_keras_training(
    run_name: str,
    params: Dict[str, Any],
    model: Any,
    history: Dict[str, List[float]],
    metrics_test: Dict[str, float],
    fig_cm: Any,
) -> None:
    """
    Log hyperparameters, learning curves, final test metrics, and confusion matrix for Keras.

    Parameters
    ----------
    run_name : str
        Descriptive name for the MLflow run.
    params : Dict[str, Any]
        Dictionary of hyperparameters.
    model : Any
        Fitted Keras model instance.
    history : Dict[str, List[float]]
        Training metrics dictionary returned by model.fit().
    metrics_test : Dict[str, float]
        Evaluation metrics on the test partition.
    fig_cm : Any
        Plotly confusion matrix figure.
    """
    uri: str = get_tracking_uri()
    mlflow.set_tracking_uri(uri)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)

        first_metric: str = next(iter(history.keys()))
        epochs_count: int = len(history[first_metric])

        for epoch in range(epochs_count):
            step_metrics: Dict[str, float] = {
                metric: float(values[epoch]) for metric, values in history.items()
            }
            mlflow.log_metrics(step_metrics, step=epoch)

        mlflow.log_metrics(metrics_test)
        mlflow.log_figure(fig_cm, "confusion_matrix.html")
        mlflow.keras.log_model(model, artifact_path="model")



def log_numpy_training(
    run_name: str,
    params: Dict[str, Any],
    history: List[Dict[str, float]],
    metrics_test: Dict[str, float],
    fig_cm: Any,
) -> None:
    """
    Log hyperparameters, step-wise metrics, and artifacts for the custom NumPy MLP.

    Parameters
    ----------
    run_name : str
        Descriptive name for the MLflow run.
    params : Dict[str, Any]
        Dictionary of hyperparameters.
    history : List[Dict[str, float]]
        Per-epoch metric records collected during gradient descent.
    metrics_test : Dict[str, float]
        Evaluation metrics on the test partition.
    fig_cm : Any
        Plotly confusion matrix figure.
    """
    uri: str = get_tracking_uri()
    mlflow.set_tracking_uri(uri)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)

        for epoch, metric_dict in enumerate(history):
            mlflow.log_metrics(metric_dict, step=epoch)

        mlflow.log_metrics(metrics_test)
        mlflow.log_figure(fig_cm, "confusion_matrix.html")