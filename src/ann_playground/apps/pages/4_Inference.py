
import streamlit as st
import requests
import mlflow
import json
import os

from mlflow.tracking import MlflowClient

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



def load_feature_lists_for_run(run_id):

    # Initialisation du client (pointe par exemple sur votre serveur MLflow ou le dossier local mlruns)
    client = MlflowClient()

    # Récupérer les métadonnées globales du run (statut, date, etc.)
    run = client.get_run(run_id)
    st.markdown(f"Statut du run :{run.info.status}")
    st.markdown(f"Paramètres : {run.data.params}")
    st.markdown(f"Métriques : {run.data.metrics}")

    # Télécharger les artéfacts enregistrés (comme votre column_transformer.pkl ou vos listes de features)
    # MLflow télécharge l'artéfact dans un dossier temporaire local et vous retourne le chemin
    # 1. Télécharger le dossier d'artéfacts du run sélectionné
    local_dir = mlflow.artifacts.download_artifacts(run_id=run_id)

    st.markdown(f"local_dir : {local_dir}")

    # 2. Charger les listes depuis les fichiers JSON sauvegardés dans les artéfacts
    with open(os.path.join(local_dir, "preprocessing/num_features.json"), "r") as f:
        num_features = json.load(f)

    with open(os.path.join(local_dir, "preprocessing/cat_features.json"), "r") as f:
        cat_features = json.load(f)

    return num_features, cat_features



# ==============================================================================
# Structure principale en onglets
# ==============================================================================


st.subheader("🎯 Inférence via FastAPI")

run_id = "f72d427571414cdf9951d485e237b388"

# Supposons que vous avez récupéré le run_id sélectionné par l'utilisateur
# (par exemple via un selectbox alimenté par vos runs MLflow)
selected_run_id = st.session_state.get("selected_run_id", run_id)

# Supposons que vos artéfacts vous renvoient ces deux listes pour le run sélectionné :
num_features, cat_features = load_feature_lists_for_run(selected_run_id)

user_inputs = {}

with st.form("prediction_form"):
    st.subheader("Variables Numériques")
    for feature in num_features:
        user_inputs[feature] = st.number_input(label=feature, value=0.0)
        
    st.subheader("Variables Catégorielles")
    for feature in cat_features:
        # Ici, vous pouvez même interroger les catégories valides si vous les avez stockées, 
        # ou laisser un champ texte / selectbox basique
        user_inputs[feature] = st.text_input(label=feature, value="")
        
    submitted = st.form_submit_button("Lancer la prédiction")

# Appel de l'API et traitement de la réponse ---
if submitted:

    # URL de votre service FastAPI (en interne sur le réseau Docker, ou http://localhost:8000 si test en local hors Docker)
    api_url = f"http://api:8000/predict/{selected_run_id}"

    # Construction du payload attendu par la classe Pydantic (PredictionRequest)
    payload = {
        "features": user_inputs
    }

    try:
        # On envoie tout dans un seul dictionnaire global sous la clé "features"
        response = requests.post(api_url,json=payload)
        
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
    
