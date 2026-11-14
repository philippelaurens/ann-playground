import os
import streamlit as st
import pandas as pd
import requests
import logging
import mlflow
import json
import re
import streamlit.components.v1 as components

from ann_playground.config import DATA, CLEAN_DATA_DIR, PROCESSED_DATA_DIR, EXPERIMENT_NAME

from ann_playground.services.mlflow_utils import (
    fetch_available_runs,
    load_parameters_lists_for_run,
    load_feature_lists_for_run,
    load_sample_artifacts,
    load_label_encoder
)

module_name = os.path.basename(__file__)
logger = logging.getLogger(module_name)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    force=True
)
logger.info("Appel")

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
        st.caption("Page d'inférence")

    # Chargement des runs disponibles
    try:
        runs_dict = fetch_available_runs()
    except Exception as e:
        st.sidebar.error(f"Erreur de connexion MLflow : {e}")
        st.stop()
        
    if runs_dict:
        # Affichage du select dans la sidebar
        selected_label = st.sidebar.selectbox(
            "Sélectionner un Run",
            options=list(runs_dict.keys())
        )
        
        # Récupération du run_id correspondant au choix de l'utilisateur
        selected_run_id = runs_dict[selected_label]
        
        st.sidebar.success(f"Run actif : `{selected_run_id[:8]}`")
    else:
        st.sidebar.warning("Aucun run trouvé dans MLflow.")
        selected_run_id = None



# ==============================================================================
# Structure principale en onglets
# ==============================================================================

run_id = selected_run_id

if selected_run_id is not None:

    # Supposons que vous avez récupéré le run_id sélectionné par l'utilisateur
    # (par exemple via un selectbox alimenté par vos runs MLflow)
    selected_run_id = st.session_state.get("selected_run_id", run_id)

    tab_run, tab_inference = st.tabs(
        [
            "1 · Configuration du Run MLflow",
            "2 · Inference",
        ]
    )

    # ------------------------------------------------------------------------------
    # 1 · Diagnostic structurel
    # ------------------------------------------------------------------------------

    parameters, metrics = load_parameters_lists_for_run(selected_run_id)
    num_features, cat_features, matrix_confusion_content = load_feature_lists_for_run(selected_run_id)
    label_encoder = load_label_encoder(selected_run_id)

    with tab_run:

        st.markdown(f"#### Run ID de l'entraînement : {run_id}")

        if parameters and metrics:

            # ### Metrics et Matrice de confusion
            d1, d2, d3, d4, d5, d6 = st.columns(6)
            d1.metric("accuracy", f"{metrics['accuracy'] * 100:.2f} %")
            d2.metric("loss", f"{metrics['loss'] * 100:.2f} %")
            d3.metric("val_accuracy", f"{metrics['val_accuracy'] * 100:.2f} %")
            d4.metric("val_loss", f"{metrics['val_loss'] * 100:.2f} %")
            d5.metric("Learning Rate", parameters['learning_rate'])
            d6.metric("Dropout Rate", parameters['dropout_rate'])

            st.iframe(matrix_confusion_content, height=550)

            # ### Performance par Classe
            metrics_data = {}
            for k, v in metrics.items():
                if k.startswith("class_"):
                    # On enlève "class_" et on sépare au dernier underscore pour isoler la métrique (precision, recall, f1_score)
                    body = k[6:]
                    class_raw, metric = body.rsplit("_", 1)
                    
                    # Identification de la métrique par son suffixe exact
                    if body.endswith("_precision"):
                        class_raw = body[:-10]
                        metric = "Precision"
                    elif body.endswith("_recall"):
                        class_raw = body[:-7]
                        metric = "Recall"
                    elif body.endswith("_f1_score"):
                        class_raw = body[:-9]
                        metric = "F1-Score"
                    else:
                        continue

                    if label_encoder is not None:
                        match = re.search(r'\d+', str(class_raw))
                        if match:
                            class_idx = int(match.group())
                            if 0 <= class_idx < len(label_encoder.classes_):
                                class_name = str(label_encoder.classes_[class_idx])
                            else:
                                logger.warning(f"Index {class_idx} hors limites pour '{class_raw}'")
                                class_name = str(class_raw)
                        else:
                            # Pas de chiffre : c'est déjà un libellé
                            class_name = str(class_raw)
                
                    if class_name not in metrics_data:
                        metrics_data[class_name] = {"Classe": class_name}
                        
                    metrics_data[class_name][metric] = v

            df_metrics = pd.DataFrame(list(metrics_data.values()))

            st.subheader("Performance par Classe")
            if not df_metrics.empty:        
                st.dataframe(
                    df_metrics, 
                    width='stretch', 
                    hide_index=True,
                    column_config={
                        "Precision": st.column_config.NumberColumn(format="%.3f"),
                        "Recall": st.column_config.NumberColumn(format="%.3f"),
                        "F1-Score": st.column_config.NumberColumn(format="%.3f"),
                    }
                )
            else:
                st.info("Aucune métrique par classe disponible pour ce run.")

            # ### Configuration du Run MLflow
            st.subheader("Configuration du Run MLflow")
            items = list(parameters.items())
            rows = []
            for i in range(0, len(items), 2):
                row = list(items[i])
                if i + 1 < len(items):
                    row.extend(list(items[i+1]))
                else:
                    row.extend(["", ""])  # Gestion propre de l'élément impair (le 11ème)
                rows.append(row)

            df_5x4 = pd.DataFrame(rows, columns=["Paramètre", "Valeur", "Paramètre ", "Valeur "])
            st.dataframe(df_5x4, width='stretch', hide_index=True)

        else:
            st.error("Erreur : Impossible de récupérer les listes de paramètres et des métrics")
            st.stop()


    # ------------------------------------------------------------------------------
    # 2 · Distributions & Relations
    # ------------------------------------------------------------------------------

    with tab_inference:

        st.markdown(f"#### Run ID de l'entraînement : {run_id}")

        # Récupération des samples pour le run actif
        df_samples = load_sample_artifacts(run_id)

        if num_features or cat_features:

            user_inputs = {}
            expected_features = num_features + cat_features

            # Gestion de l'état dans session_state
            if "prediction_result" not in st.session_state:
                st.session_state.prediction_result = None
            if "form_submitted" not in st.session_state:
                st.session_state.form_submitted = False  

            # Initialisation des valeurs par défaut dans le session_state si besoin
            if "form_values" not in st.session_state:
                st.session_state.form_values = {}

            # Si des échantillons existent pour ce run, on propose un sélecteur rapide
            if df_samples is not None and not df_samples.empty:
                st.markdown("### ⚡ Remplissage rapide par échantillon")
                sample_cols = st.columns(len(df_samples))
                
                for idx, row in df_samples.iterrows():
                    # On essaie d'identifier le nom de la classe ou un libellé parlant
                    # Cherche dynamiquement le nom de la colonne cible (adapte la liste selon ton dataset)
                    target_col = next((c for c in ["Target", "target", "class", "label", "status"] if c in row), None)
                    if target_col:
                        label_name = str(row[target_col])
                    else:
                        label_name = f"Exemple {idx + 1}"
                    with sample_cols[idx]:
                        if st.button(f"Charger : {label_name}", key=f"btn_sample_{idx}"):
                            # On injecte chaque valeur de la ligne dans le session_state
                            for col in expected_features:
                                if col in row:
                                    st.session_state.form_values[col] = row[col]
                            st.rerun() 

            with st.expander(
                "📝 Paramètres d'entrée (Modifier les variables)", 
                expanded=not st.session_state.form_submitted
            ):
                with st.form("prediction_form"):
                    st.subheader("Variables Numériques")
                    if num_features:
                        # On découpe les features numériques par blocs de 3
                        for i in range(0, len(num_features), 3):
                            cols = st.columns(3)
                            batch = num_features[i:i+3]
                            for j, feature in enumerate(batch):
                                with cols[j]:
                                    default_val = float(st.session_state.form_values.get(feature, 0.0))
                                    # user_inputs[feature] = st.number_input(label=feature, value=0.0)
                                    user_inputs[feature] = st.number_input(label=feature, value=default_val)
                                    
                    st.subheader("Variables Catégorielles")
                    if cat_features:
                        # On découpe les features catégorielles par blocs de 3
                        for i in range(0, len(cat_features), 3):
                            cols = st.columns(3)
                            batch = cat_features[i:i+3]
                            for j, feature in enumerate(batch):
                                with cols[j]:
                                    default_val = str(st.session_state.form_values.get(feature, ""))
                                    # user_inputs[feature] = st.text_input(label=feature, value="")
                                    user_inputs[feature] = st.text_input(label=feature, value=default_val)
                                    
                    submitted = st.form_submit_button("Lancer la prédiction")

            # Appel de l'API et traitement de la réponse ---
            if submitted:
                st.session_state.form_submitted = True

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
                        st.session_state.prediction_result = response.json()
                    else:
                        try:
                            error_msg = response.json().get("detail", response.text)
                        except Exception:
                            error_msg = response.text if response.text else "Réponse vide"
                        st.error(f"Erreur API ({response.status_code}) : {error_msg}")
                        st.session_state.prediction_result = None

                except Exception as e:
                    st.error(f"Impossible de joindre l'API : {e}")
                    st.session_state.prediction_result = None
                
                # Force le rechargement pour replier l'expander instantanément
                st.rerun()

            # Affichage des résultats (si disponibles)
            if st.session_state.prediction_result is not None:
                result = st.session_state.prediction_result
                
                st.success("Prédiction effectuée avec succès !")
                
                # Bouton pour ré-ouvrir le formulaire si l'utilisateur veut modifier un paramètre
                if st.button("🔄 Modifier les valeurs d'entrée"):
                    st.session_state.form_submitted = False
                    st.session_state.prediction_result = None
                    st.rerun()

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(label="Classe Prédite", value=result["prediction_label"], delta=f"Indice : {result['prediction_index']}")
                with col2:
                    st.metric(label="Confiance", value=f"{result['confidence'] * 100:.1f}%")
                with col3:
                    st.info(f"**Modèle :** `{result['model_used'][:8]}...`\n\n**PCA :** `{result['pca_used']}`")

                # Graphique des probabilités
                st.subheader("Répartition des probabilités")
                probs_dict = result["probabilities_per_class"]
                if probs_dict:
                    df_probs = pd.DataFrame(list(probs_dict.items()), columns=["Classe", "Probabilité"]).set_index("Classe")
                    st.bar_chart(df_probs)

                with st.expander("Voir la charge utile brute (JSON)"):
                    st.json(result)

        else:
            st.error("Erreur : Impossible de récupérer les listes de features pour ce run (listes vides ou introuvables).")
            st.stop()

else:
    st.error("Aucun run trouvé dans MLflow.")