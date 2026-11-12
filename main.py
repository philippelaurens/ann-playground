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
    "Marital status",
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

EXPECTED_FEATURES = [*NUM_FEATURES, *CAT_FEATURES]


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

    try:
        # Téléchargement des artéfacts du run
        local_dir = mlflow.artifacts.download_artifacts(run_id=run_id)
        
        # Config du pipeline
        with open(os.path.join(local_dir, "config/pipeline_config.json"), "r") as f:
            pipeline_config = json.load(f)
            
        # Column Transformer
        column_transformer = joblib.load(os.path.join(local_dir, "preprocessing/column_transformer.pkl"))
        
        # PCA optionnelle
        pca_model = None
        if pipeline_config.get("use_pca", False):
            pca_model = joblib.load(os.path.join(local_dir, "preprocessing/pca_model.pkl"))
            
        # Modèle Keras
        model = mlflow.keras.load_model(f"runs:/{run_id}/model")
        
        # Mise en cache
        model_cache[run_id] = model
        preprocessing_cache[run_id] = {
            "column_transformer": column_transformer,
            "pca_model": pca_model,
            "config": pipeline_config
        }
        
        return model, preprocessing_cache[run_id]

    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Erreur de chargement pour le run {run_id} : {str(e)}")


@app.post("/predict/{run_id}")
def predict_with_run(run_id: str, payload: PredictionRequest):
    logger.info("----- Requête reçue sur l'endpoint racine '/predict' -----")
    model, artifacts = load_run_artifacts(run_id)

    col_transformer = artifacts["column_transformer"]
    pca_model = artifacts["pca_model"]
    expected_features = artifacts["expected_features"]
    
    try:
        received_data = payload.features
        
        # Vérification dynamique des colonnes manquantes
        missing_cols = [col for col in expected_features if col not in received_data]
        if missing_cols:
            raise HTTPException(
                status_code=400, 
                detail=f"Colonnes manquantes pour ce run : {missing_cols}"
            )
            
        # Construction du DataFrame en filtrant et ordonnant exactement selon les besoins de ce run spécifique
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
        pred_value = float(preds[0][0]) if preds.shape[1] == 1 else float(np.argmax(preds[0]))
        
        return {
            "prediction": pred_value,
            "raw_output": preds[0].tolist(),
            "model_used": run_id,
            "features_used_count": len(expected_features),
            "pca_used": pca_model is not None
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur d'inférence dynamique : {str(e)}")
    
        