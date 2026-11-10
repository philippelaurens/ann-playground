from pathlib import Path
from typing import Any, Dict, List, Tuple
import time
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

from ann_playground.config import DATA, CLEAN_DATA_DIR, PROCESSED_DATA_DIR

from ann_playground.data.data_utils import (
    load_source_data,
    export_train_test_parquet,
    load_prepared_datasets,
    export_pca_parquet,
)
from ann_playground.utils.preprocessing_utils import (
    count_duplicates,
    missing_summary,
    special_columns,
    classify_feature_types,
    correlated_features,
    drop_redundant_features,
    target_correlations,
    preprocess_data,
)
from ann_playground.data.data_visualisation import (
    plot_pca_cumulative_variance,
    scree_plot_with_ideal_components,
)

# ---------------------------------------------------------------------------
# Authentification
# ---------------------------------------------------------------------------

# Vérification de sécurité obligatoire sur chaque sous-pages
if not st.session_state.get("authenticated", False):
    st.warning("⚠️ Veuillez vous connecter depuis la page d'accueil pour accéder à cette section.")
    st.stop()

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="ANN Playground : Evaluation",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# Barre latérale : Source de données & Target
# ==============================================================================

st.title("ANN Playground : Inference")

with st.sidebar:
    with st.empty().container():
        st.caption("Module générique d'évaluation")


# ==============================================================================
# Structure principale en onglets
# ==============================================================================

import streamlit as st
import requests

st.subheader("🎯 Inférence via FastAPI")

# Supposons que vous avez récupéré le run_id sélectionné par l'utilisateur
# (par exemple via un selectbox alimenté par vos runs MLflow)
selected_run_id = st.session_state.get("selected_run_id", "votre_run_id_par_defaut")

# --- 1. Formulaire de saisie des features ---
with st.form("inference_form"):
    st.markdown("### Saisir les caractéristiques d'entrée")
    
    # Exemple avec quelques champs numériques (à adapter selon vos features réelles)
    f1 = st.number_input("Feature 1", value=0.0)
    f2 = st.number_input("Feature 2", value=0.0)
    f3 = st.number_input("Feature 3", value=0.0)
    
    # Bouton de soumission du formulaire
    submitted = st.form_submit_button("Lancer la prédiction")

# --- 2. Appel de l'API et traitement de la réponse ---
if submitted:
    # Construction du payload attendu par la classe Pydantic (PredictionRequest)
    payload = {
        "features": [f1, f2, f3]
    }
    
    # URL de votre service FastAPI (en interne sur le réseau Docker, ou http://localhost:8000 si test en local hors Docker)
    api_url = f"http://api:8000/predict/{selected_run_id}"
    
    try:
        # Requête POST vers FastAPI
        with st.spinner("Calcul de la prédiction en cours..."):
            response = requests.post(api_url, json=payload)
        
        # Vérification du code de statut HTTP
        if response.status_code == 200:
            # Récupération du dictionnaire JSON renvoyé par l'API
            result = response.json()
            
            # Affichage propre des résultats reçus
            st.success("Prédiction effectuée avec succès !")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Prédiction finale", value=result["prediction"])
            with col2:
                st.info(f"Modèle utilisé : `{result['model_used'][:8]}...`\n\nPCA active : `{result['pca_used']}`")
            
            with st.expander("Voir les sorties brutes du modèle (raw_output)"):
                st.json(result["raw_output"])
                
        else:
            # Affichage de l'erreur renvoyée par FastAPI (ex: 404 si run_id inconnu, 400 si erreur de shape)
            st.error(f"Erreur API ({response.status_code}) : {response.json().get('detail', response.text)}")
            
    except requests.exceptions.ConnectionError:
        st.error("Impossible de joindre le service FastAPI (`http://api:8000`). Vérifiez que le conteneur `api` est bien démarré.")


# ------------------------------------------------------------------------------
# 1 · Diagnostic structurel
# ------------------------------------------------------------------------------



# ------------------------------------------------------------------------------
# 2 · Distributions & Relations
# ------------------------------------------------------------------------------



# ------------------------------------------------------------------------------
# 3 · Colinéarité & Sélection
# ------------------------------------------------------------------------------
    
