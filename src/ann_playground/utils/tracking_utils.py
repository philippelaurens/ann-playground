import os
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional
import mlflow
import mlflow.keras
from mlflow.entities import Experiment
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from ann_playground.config import MLFLOW_TRACKING_URI
from ann_playground.model.numpy_mlp_wrapper import NumpyMLPWrapper


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


def get_metrics_to_log(
        class_report: Dict[str, Any]
) -> Dict[str, float]:
    metrics_to_log = {}
    
    # Parcours du rapport de classification
    for class_name, metrics in class_report.items():

        # On ignore explicitement l'accuracy, macro avg et weighted avg
        if class_name in ['accuracy', 'macro avg', 'weighted avg']:
            continue 
        
        # Cas particulier de l'accuracy globale (valeur flottante simple)
        # if class_name == 'accuracy':
        #   metrics_to_log["accuracy"] = metrics
        #    continue
            
        # Optionnel : Gestion propre des moyennes globales si tu souhaites les logger aussi
        # if class_name in ['macro avg', 'weighted avg']:
        #    prefix = class_name.replace(" ", "_") # donne 'macro_avg' ou 'weighted_avg'
        #    metrics_to_log[f"{prefix}_precision"] = metrics['precision']
        #    metrics_to_log[f"{prefix}_recall"] = metrics['recall']
        #    metrics_to_log[f"{prefix}_f1_score"] = metrics['f1-score']
        #    continue
        
        # Pour chaque classe individuelle
        if isinstance(metrics, dict):
            precision = metrics['precision']
            recall = metrics['recall']
            f1 = metrics['f1-score']
            
            # Formatage des noms de métriques pour MLflow (ex: class_2_f1_score)
            metrics_to_log[f"class_{class_name}_precision"] = precision
            metrics_to_log[f"class_{class_name}_recall"] = recall
            metrics_to_log[f"class_{class_name}_f1_score"] = f1       
            
    return metrics_to_log


def log_keras_training(
    run_name: str,
    params: Dict[str, Any],
    model: Any,
    history: Dict[str, List[float]],
    metrics_test: Dict[str, float],
    fig_cm: Any,
    report_dict: Dict[str, Any],
    output_dir: Path | str = "data/processed_data", # Ajout pour pointer vers les artefacts locaux
) -> None:
    if not mlflow.get_tracking_uri():
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))

    out_path = Path(output_dir)
    config_path = out_path / "pipeline_config.json"

    # Lecture de la configuration pour savoir si la PCA est active
    use_pca = False
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            pipeline_config = json.load(f)
            use_pca = pipeline_config.get("use_pca", False)

    with mlflow.start_run(run_name=run_name):
        # Log des hyperparamètres de base
        mlflow.log_params(params)

        # Si Standard : on log le compte des features en paramètre...
        num_path = out_path / "num_features.json"
        cat_path = out_path / "cat_features.json"
        
        if num_path.exists():
            with open(num_path, "r", encoding="utf-8") as f:
                num_feats = json.load(f)
            mlflow.log_param("num_features_count", len(num_feats))
            # ... ET on conserve la trace exacte de la liste en l'envoyant en artefact
            mlflow.log_artifact(str(num_path), artifact_path="preprocessing")
        
        if cat_path.exists():
            with open(cat_path, "r", encoding="utf-8") as f:
                cat_feats = json.load(f)
            mlflow.log_param("cat_features_count", len(cat_feats))
            # Idem pour les catégorielles
            mlflow.log_artifact(str(cat_path), artifact_path="preprocessing")        

        # Log intelligent selon l'état PCA ou Standard
        if use_pca:
            # Si PCA : on log le nombre de composantes en paramètre
            n_components = pipeline_config.get("n_components")
            if n_components:
                mlflow.log_param("n_components", n_components)
            
            # On log le modèle PCA en tant qu'artefact
            pca_path = out_path / "pca_model.pkl"
            if pca_path.exists():
                mlflow.log_artifact(str(pca_path), artifact_path="preprocessing")

        # Log du ColumnTransformer (commun aux deux modes)
        transformer_path = out_path / "column_transformer.pkl"
        if transformer_path.exists():
            mlflow.log_artifact(str(transformer_path), artifact_path="preprocessing")

        # Log du fichier de configuration global
        if config_path.exists():
            mlflow.log_artifact(str(config_path), artifact_path="config")

        # Log des métriques d'entraînement (époques)
        first_metric: str = next(iter(history.keys()))
        epochs_count: int = len(history[first_metric])

        for epoch in range(epochs_count):
            step_metrics: Dict[str, float] = {
                metric: float(values[epoch]) for metric, values in history.items()
            }
            mlflow.log_metrics(step_metrics, step=epoch)

        # Log des métriques de test, de la matrice de confusion et du modèle Keras
        metrics_to_log = get_metrics_to_log(report_dict)
        
        mlflow.log_metrics(metrics_test)
        mlflow.log_figure(fig_cm, "confusion_matrix.html")
        mlflow.log_metrics(metrics_to_log)
        mlflow.keras.log_model(model, artifact_path="model")


def log_numpy_training(
    run_name: str,
    params: Dict[str, Any],
    model: Any, # Votre instance de NumpyMLP entraînée
    history: List[Dict[str, float]], # Format : [{'loss': 0.5, 'accuracy': 0.8}, ...]
    metrics_test: Dict[str, float],
    fig_cm: Any,
    report_dict: Dict[str, Any],
    output_dir: str = "data/processed_data",
) -> None:
    if not mlflow.get_tracking_uri():
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Lecture de la configuration pour savoir si la PCA est active
    config_path = out_path / "pipeline_config.json"
    use_pca = False
    pipeline_config = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            pipeline_config = json.load(f)
            use_pca = pipeline_config.get("use_pca", False)

    # Sauvegarde locale temporaire du modèle NumpyMLP (avant le log)
    local_model_path = out_path / "numpy_mlp_model.pkl"
    with open(local_model_path, "wb") as f:
        pickle.dump(model, f)

    # Démarrage du run unique MLflow englobant TOUT
    with mlflow.start_run(run_name=run_name):
        
        # Log des hyperparamètres de base
        mlflow.log_params(params)

        num_path = out_path / "num_features.json"
        cat_path = out_path / "cat_features.json"
        
        if num_path.exists():
            with open(num_path, "r", encoding="utf-8") as f:
                num_feats = json.load(f)
            mlflow.log_param("num_features_count", len(num_feats))
            mlflow.log_artifact(str(num_path), artifact_path="preprocessing")
        
        if cat_path.exists():
            with open(cat_path, "r", encoding="utf-8") as f:
                cat_feats = json.load(f)
            mlflow.log_param("cat_features_count", len(cat_feats))
            mlflow.log_artifact(str(cat_path), artifact_path="preprocessing")

        # Log intelligent selon l'état PCA ou Standard
        if use_pca:
            n_components = pipeline_config.get("n_components")
            if n_components:
                mlflow.log_param("n_components", n_components)
            
            pca_path = out_path / "pca_model.pkl"
            if pca_path.exists():
                mlflow.log_artifact(str(pca_path), artifact_path="preprocessing")
                
        # Log du ColumnTransformer et de la Config (communs)
        transformer_path = out_path / "column_transformer.pkl"
        if transformer_path.exists():
            mlflow.log_artifact(str(transformer_path), artifact_path="preprocessing")

        if config_path.exists():
            mlflow.log_artifact(str(config_path), artifact_path="config")

        # Log des métriques d'entraînement époque par époque (NumPy)
        for epoch, metric_dict in enumerate(history):
            mlflow.log_metrics(metric_dict, step=epoch)

        # Log des métriques de test et de la matrice de confusion
        metrics_to_log = get_metrics_to_log(report_dict)
        mlflow.log_metrics(metrics_test)
        mlflow.log_figure(fig_cm, "confusion_matrix.html")
        mlflow.log_metrics(metrics_to_log)

        # Log officiel du modèle NumPy via le Wrapper pyfunc
        mlflow.pyfunc.log_model(
            artifact_path="numpy_model",
            python_model=NumpyMLPWrapper(),
            artifacts={"numpy_model_file": str(local_model_path)},
            registered_model_name="NumpyMLP_Classifier"
        )