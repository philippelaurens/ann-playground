import os
import json
import joblib
import logging
import mlflow.keras
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Configuration globale des logs
# Récupération du logger (ici celui d'uvicorn pour unifier les flux)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("uvicorn")

app = FastAPI(
    title="Student Success Prediction API",
    description="API d'inférence dynamique avec pré-traitement par ColumnTransformer"
)

# Caches en mémoire pour éviter les re-téléchargements
model_cache = {}
preprocessing_cache = {}

# Liste officielle de vos 16 features attendues
NUM_FEATURES = [
    "Previous qualification (grade)",
    "Admission grade",
    "Age at enrollment",
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Unemployment rate",
    "Inflation rate",
    "GDP"
]

CAT_FEATURES = [
    "Marital Status",
    "Application mode",
    "Application order",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (without evaluations)"
]

EXPECTED_FEATURES = NUM_FEATURES + CAT_FEATURES


# Avec class PredictionRequest(BaseModel), Pydantic intercepte le JSON envoyé par Streamlit, 
#  vérifie qu'il y a bien une clé racine nommée features, et s'assure que 
#  toutes les valeurs à l'intérieur sont bien des float, int ou str. 
# Si un utilisateur envoie par erreur une liste ou un objet complexe au lieu d'une valeur numérique, 
#  FastAPI rejette immédiatement la requête avec une erreur propre (422 Unprocessable Entity).
class PredictionRequest(BaseModel):
    features: dict[str, float | int | str] = Field(
        ..., 
        description="Dictionnaire dynamique des features transmis par Streamlit"
    )
        
def load_run_artifacts(run_id: str):
    """Charge et met en cache les artéfacts MLflow pour un run donné."""
    if run_id in model_cache:
        return model_cache[run_id], preprocessing_cache.get(run_id)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(tracking_uri)

    logger.info("----- Inside load_run_artifacts -----")

    try:
        # Téléchargement des artéfacts du run
        local_dir = mlflow.artifacts.download_artifacts(run_id=run_id)

        
        # Config du pipeline
        with open(os.path.join(local_dir, "config/pipeline_config.json"), "r") as f:
            pipeline_config = json.load(f)
            
        # Column Transformer
        column_transformer = joblib.load(os.path.join(local_dir, "preprocessing/column_transformer.pkl"))
        
        # Column Transformer
        label_encoder = joblib.load(os.path.join(local_dir, "preprocessing/label_encoder.pkl"))
        
        # PCA optionnelle
        pca_model = None
        if pipeline_config.get("use_pca", True):
            pca_model = joblib.load(os.path.join(local_dir, "preprocessing/pca_model.pkl"))
            
        # Modèle Keras
        model = mlflow.keras.load_model(f"runs:/{run_id}/model")


        # Mise en cache
        model_cache[run_id] = model
        preprocessing_cache[run_id] = {
            "column_transformer": column_transformer,
            "pca_model": pca_model,
            "label_encoder": label_encoder,
            "config": pipeline_config
        }
        
        return model, preprocessing_cache[run_id]

    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Erreur de chargement pour le run {run_id} : {str(e)}")


@app.post("/predict/{run_id}")
def predict_with_run(run_id: str, payload: PredictionRequest):
    model, artifacts = load_run_artifacts(run_id)

    col_transformer = artifacts["column_transformer"]
    pca_model = artifacts["pca_model"]
    label_encoder = artifacts["label_encoder"]

    expected_features = EXPECTED_FEATURES
    
    try:
        received_data = payload.features
        logger.info(f"----- received_data ----- {received_data}")
        
        # Vérification dynamique des colonnes manquantes
        missing_cols = [col for col in expected_features if col not in received_data]
        if missing_cols:
            raise HTTPException(
                status_code=400, 
                detail=f"Colonnes manquantes pour ce run : {missing_cols}"
            )
            
        # Construction du DataFrame en filtrant et ordonnant selon les besoins du run
        X_raw = pd.DataFrame([received_data])[expected_features]
        
        # Application du ColumnTransformer du run
        X_transformed = col_transformer.transform(X_raw)
        
        # Application conditionnelle de la PCA du run
        if pca_model is not None:
            X_transformed = pca_model.transform(X_transformed)
            
        # Inférence Keras
        # Le modèle se termine par une couche softmax à 3 neurones, Keras renvoie un tableau de probabilités 
        #  pour chaque classe (par exemple : [0.05, 0.15, 0.80]).
        #  np.argmax(preds[0]) scanne le tableau des probabilités et récupère l'indice de la valeur la plus haute.
        preds = model.predict(X_transformed)
        
        # Récupération de l'indice ayant la probabilité maximale (0, 1 ou 2)
        pred_idx = int(np.argmax(preds[0]))
        pred_proba = float(np.max(preds[0]))

        # Décodage et association des probabilités via le LabelEncoder
        if label_encoder is not None:
            # Traduction de l'indice (ex: 2) en libellé textuel (ex: "Graduate")
            pred_label = str(label_encoder.inverse_transform([pred_idx])[0])
            
            # Dictionnaire associant chaque vrai nom de classe à sa probabilité prédite
            class_names = label_encoder.classes_ # ["Dropout", "Enrolled", "Graduate"]
            probs_per_class = {
                str(class_names[i]): float(prob) 
                for i, prob in enumerate(preds[0])
            }
        else:
            pred_label = str(pred_idx)
            probs_per_class = {f"Class_{i}": float(prob) for i, prob in enumerate(preds[0])}

        # Retour de la réponse JSON structurée pour Streamlit
        return {
            "prediction_index": pred_idx,
            "prediction_label": pred_label,
            "confidence": pred_proba,
            "probabilities_per_class": probs_per_class,
            "raw_output": preds[0].tolist(),
            "model_used": run_id,
            "features_used_count": len(expected_features),
            "pca_used": pca_model is not None
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur d'inférence dynamique : {str(e)}")
    
        