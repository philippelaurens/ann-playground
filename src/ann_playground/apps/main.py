import os
import json
import joblib
import mlflow.keras
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any

app = FastAPI(title="ML Inferences API with Full Preprocessing")

# Caches en mémoire
model_cache = {}
preprocessing_cache = {}

class PredictionRequest(BaseModel):
    # Dictionnaire ou liste de valeurs brutes correspondant aux colonnes d'origine
    features: List[float] = Field(..., description="Vecteur des features brutes d'entrée")

def load_run_artifacts(run_id: str):
    """Charge le modèle, le column_transformer, la PCA éventuelle et la config du run."""
    if run_id in model_cache:
        return model_cache[run_id], preprocessing_cache.get(run_id)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(tracking_uri)

    try:
        # Téléchargement des artéfacts depuis MLflow
        client = mlflow.tracking.MlflowClient()
        
        # Téléchargement du dossier d'artéfacts du run dans un dossier local temporaire ou direct
        local_dir = mlflow.artifacts.download_artifacts(run_id=run_id)
        
        # Chargement de la configuration du pipeline
        config_path = os.path.join(local_dir, "pipeline_config.json")
        with open(config_path, "r") as f:
            pipeline_config = json.load(f)
            
        # Chargement du Column Transformer
        col_transformer_path = os.path.join(local_dir, "column_transformer.pkl")
        column_transformer = joblib.load(col_transformer_path)
        
        # Chargement conditionnel de la PCA
        pca_model = None
        if pipeline_config.get("use_pca", False):
            pca_path = os.path.join(local_dir, "pca_model.pkl")
            pca_model = joblib.load(pca_path)
            
        # Chargement du modèle Keras
        model_uri = f"runs:/{run_id}/model"
        model = mlflow.keras.load_model(model_uri)
        
        # Mise en cache
        model_cache[run_id] = model
        preprocessing_cache[run_id] = {
            "column_transformer": column_transformer,
            "pca_model": pca_model,
            "config": pipeline_config
        }
        
        return model, preprocessing_cache[run_id]

    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Erreur de chargement des artéfacts pour le run {run_id} : {str(e)}")

@app.post("/predict/{run_id}")
def predict_with_run(run_id: str, payload: PredictionRequest):
    model, artifacts = load_run_artifacts(run_id)
    
    col_transformer = artifacts["column_transformer"]
    pca_model = artifacts["pca_model"]
    
    try:
        # 1. Les données brutes entrent sous forme de tableau 2D
        X_raw = np.array([payload.features])
        
        # 2. Application du Column Transformer (Standardisation, Encodage, etc.)
        # Attention : assurez-vous que X_raw est sous forme de DataFrame ou array 
        # compatible avec la façon dont le ColumnTransformer a été fitté.
        X_transformed = col_transformer.transform(X_raw)
        
        # 3. Application conditionnelle de la PCA si activée dans la config
        if pca_model is not None:
            X_transformed = pca_model.transform(X_transformed)
            
        # 4. Inférence avec le modèle Keras
        preds = model.predict(X_transformed)
        pred_value = float(preds[0][0]) if preds.shape[1] == 1 else float(np.argmax(preds[0]))
        
        return {
            "prediction": pred_value,
            "raw_output": preds[0].tolist(),
            "model_used": run_id,
            "pca_used": pca_model is not None
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur durant le pipeline de preprocessing ou d'inférence : {str(e)}")