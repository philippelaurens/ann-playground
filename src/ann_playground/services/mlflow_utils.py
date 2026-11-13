import os
import json
import ast
import mlflow
import pandas as pd

from collections import defaultdict
from mlflow.tracking import MlflowClient

from ann_playground.config import EXPERIMENT_NAME


def fetch_available_runs(max_results=None):
    # Construction dynamique des arguments pour mlflow.search_runs
    search_kwargs = {
        "experiment_names": [EXPERIMENT_NAME],
        "order_by": ["start_time DESC"]
    }
    
    # Si un max_results est spécifié, on l'ajoute au dictionnaire
    if max_results is not None:
        search_kwargs["max_results"] = max_results
        
    runs_df = mlflow.search_runs(**search_kwargs)
    
    if runs_df.empty:
        return {}
    
    runs_dict = {}
    for _, row in runs_df.iterrows():
        run_id = row["run_id"]
        run_name = row.get("tags.mlflow.runName", run_id[:8])
        
        # Extraction sécurisée de quelques métriques ou paramètres clés (adapte selon tes logs)
        acc = row.get("metrics.accuracy", None)
        val_loss = row.get("metrics.val_loss", None)
        lr = row.get("params.learning_rate", None)
        
        # Construction d'une chaîne de détails textuelle
        details = []
        if acc is not None and pd.notna(acc):
            details.append(f"Acc: {float(acc):.3f}")
        if val_loss is not None and pd.notna(val_loss):
            details.append(f"Val Loss: {float(val_loss):.3f}")
        if lr is not None and pd.notna(lr):
            details.append(f"LR: {lr}")
        
        details_str = f" [{', '.join(details)}]" if details else ""
        
        # Libellé enrichi visible dans le selectbox Streamlit
        display_label = f"{run_name}{details_str} ({run_id[:8]}...)"
        runs_dict[display_label] = run_id
        
    return runs_dict


def load_parameters_lists_for_run(run_id):

    # Initialisation du client (pointe par exemple sur votre serveur MLflow ou le dossier local mlruns)
    client = MlflowClient()

    # Récupérer les métadonnées globales du run (statut, date, etc.)
    run = client.get_run(run_id)
    parameters = run.data.params
    metrics = run.data.metrics

    return parameters, metrics


def load_feature_lists_for_run(run_id):
    # Valeurs par défaut au cas où les fichiers manquent
    num_features = []
    cat_features = []
    matrix_confusion_content = None

    try:
        # Télécharger les artéfacts enregistrés
        local_dir = mlflow.artifacts.download_artifacts(run_id=run_id)

        # Chemin du fichier num_features.json
        num_path = os.path.join(local_dir, "preprocessing", "num_features.json")
        if os.path.exists(num_path):
            with open(num_path, "r", encoding="utf-8") as f:
                num_features = json.load(f)

        # Chemin du fichier cat_features.json
        cat_path = os.path.join(local_dir, "preprocessing", "cat_features.json")
        if os.path.exists(cat_path):
            with open(cat_path, "r", encoding="utf-8") as f:
                cat_features = json.load(f)

        # Chemin du fichier confusion_matrix.html
        html_path = os.path.join(local_dir, "confusion_matrix.html")
        if os.path.exists(html_path):
            with open(html_path, "r", encoding="utf-8") as f:
                matrix_confusion_content = f.read()

    except Exception as e:
        # Optionnel : loguer l'erreur ou l'afficher dans Streamlit si besoin
        print(f"Erreur lors du chargement des artéfacts pour le run {run_id}: {e}")

    return num_features, cat_features, matrix_confusion_content


def load_sample_artifacts(run_id):
    try:
        # Télécharge le fichier df_samples.parquet depuis les artéfacts du run
        local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="preprocessing/df_samples.parquet")
        df_samples = pd.read_parquet(local_path)
        return df_samples
    except Exception as e:
        # Si le fichier n'existe pas encore dans les vieux runs
        return None


def build_structured_runs_table(max_results=None):
    """
    Interroge MLflow et retourne un dictionnaire imbriqué sous la forme :
    tab_runs[engine][activation][hidden_layers] = [ {run_id, run_name, metrics, display_label}, ... ]
    """
    search_kwargs = {
        "experiment_names": [EXPERIMENT_NAME],
        "order_by": ["start_time DESC"]
    }
    
    if max_results is not None:
        search_kwargs["max_results"] = max_results
        
    runs_df = mlflow.search_runs(**search_kwargs)
    
    if runs_df.empty:
        return {}
    
    # Dictionnaire imbriqué par défaut (évite les KeyError lors de l'insertion)
    tab_runs = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    for _, row in runs_df.iterrows():
        run_id = row["run_id"]
        run_name = row.get("tags.mlflow.runName", run_id[:8])
        
        # 1. Extraction des paramètres d'architecture (adapte les clés si tes params ont d'autres noms)
        engine = str(row.get("params.engine", "Standard"))
        activation = str(row.get("params.activation", "ReLU"))
        hidden_layers = str(row.get("params.hidden_layers", "Indéterminé"))

        layers_list = ast.literal_eval(hidden_layers)
        num_layers = len(layers_list)
        activation_str = f"{activation} ({num_layers})"
        
        # 2. Extraction des métriques pour affichage / tri ultérieur
        acc = row.get("metrics.accuracy", None)
        val_loss = row.get("metrics.val_loss", None)
        
        # Extraction sécurisée des paramètres et métriques
        lr = row.get("params.learning_rate", None)
        dropout = row.get("params.dropout_rate", None)
        batch_norm = row.get("params.batch_norm", None)
        batch_size = row.get("params.batch_size", None)
        num_features_count = row.get("params.num_features_count", None)
        cat_features_count  = row.get("params.cat_features_count", None)
        
        # Gestion de l'epoch (prend la meilleure si elle existe, sinon l'epoch standard)
        best_epoch = row.get("metrics.best_epoch", row.get("params.epochs", None))

        # Gestion de la PCA (oui avec n_components, ou non)
        n_components = row.get("params.n_components", row.get("params.pca_n_components", None))
        pca_status = f"oui ({n_components})" if n_components is not None and pd.notna(n_components) else "non"
        pca_status += f" / {num_features_count}-{cat_features_count}"

        val_accuracy = row.get("metrics.val_accuracy", None)
        val_loss = row.get("metrics.val_loss", None)
        epochs = row.get("params.epochs", None)
        best_epoch = row.get("metrics.best_epoch", None)
        
        epochs_str = str(int(epochs)) if epochs is not None and pd.notna(epochs) else "N/A"

        if best_epoch is not None and pd.notna(best_epoch):
            epochs_display = f"{epochs_str} (best: {int(best_epoch)})"
        else:
            epochs_display = epochs_str

        # Construction de l'objet descriptif enrichi
        run_info = {
            "run_id": run_id,
            "run_name": run_name,
            "activation": activation_str,
            "learning_rate": float(lr) if lr is not None and pd.notna(lr) else None,
            "dropout_rate": float(dropout) if dropout is not None and pd.notna(dropout) else None,
            "batch_norm": batch_norm,
            "epochs": epochs_display,
            "batch_size": int(batch_size) if batch_size is not None and pd.notna(batch_size) else None,
            "pca": pca_status,
            "val_accuracy": float(val_accuracy) if val_accuracy is not None and pd.notna(val_accuracy) else None,
            "val_loss": float(val_loss) if val_loss is not None and pd.notna(val_loss) else None,
            "display_label": f"{run_name} (Acc: {acc:.3f})" if acc is not None and pd.notna(acc) else f"{run_name} ({run_id[:8]}...)"
        }

        # Insertion dans la structure imbriquée
        tab_runs[engine][activation][hidden_layers].append(run_info)

    for engine, activations_dict in tab_runs.items():
            for activation, layers_dict in activations_dict.items():
                for layer_key, runs_list in layers_dict.items():
                    runs_list.sort(
                        key=lambda x: x["val_accuracy"] if x["val_accuracy"] is not None else -float('inf'),
                        reverse=True
                    )
        
                    
    return tab_runs