from pathlib import Path
from typing import Any, Dict, List, Tuple
import time
import numpy as np
import pandas as pd
import logging
from collections import defaultdict
import plotly.express as px
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA


from ann_playground.services.mlflow_utils import (
    fetch_available_runs,
    build_structured_runs_table
)

logger = logging.getLogger(__name__)
# Force l'activation des logs et leur affichage dans la sortie standard (terminal / docker)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    force=True
)
logger = logging.getLogger(__name__)

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

st.title("Analyse et Evaluation Keras / NumPy")

# Chargement des runs disponibles
try:
    tab_runs = build_structured_runs_table(max_results=50)
except Exception as e:
    st.sidebar.error(f"Erreur de connexion MLflow : {e}")
    st.stop()


# ==============================================================================
# Sidebar
# ==============================================================================

with st.sidebar:
    st.caption("Module générique de diagnostic")

    if tab_runs:
        selected_engine = st.selectbox("Moteur (Engine)", list(tab_runs.keys()))

        available_activations = list(tab_runs[selected_engine].keys())
        selected_activations = st.multiselect(
            "Fonctions d'activation",
            options=available_activations,
            default=available_activations[:1],
        )

        available_layers = set()
        for act in selected_activations:
            available_layers.update(tab_runs[selected_engine][act].keys())
        available_layers = sorted(available_layers)

        selected_layers = st.multiselect(
            "Couches cachées",
            options=available_layers,
            default=available_layers[:1] if available_layers else [],
        )

        matching_runs = []
        for act in selected_activations:
            for layer in selected_layers:
                matching_runs.extend(tab_runs[selected_engine][act].get(layer, []))

        # Le clic est mémorisé dans session_state
        if st.button("📋 Afficher les runs correspondants"):
            st.session_state["show_matching"] = True

        # Réinitialise l'affichage si les filtres changent
        current_filters = (selected_engine, tuple(selected_activations), tuple(selected_layers))
        if st.session_state.get("last_filters") != current_filters:
            st.session_state["show_matching"] = False
            st.session_state["last_filters"] = current_filters

        show_matching = st.session_state.get("show_matching", False)
    else:
        matching_runs = []
        show_matching = False


# ==============================================================================
# Structure principale en onglets
# ==============================================================================

tab_diag, tab_autre = st.tabs(
    [
        "1 · Configuration du Run MLflow",
        "2 · Inference",
    ]
)


 

# ------------------------------------------------------------------------------
# 1 · Diagnostic structurel
# ------------------------------------------------------------------------------

with tab_diag:
    if show_matching and matching_runs:

        st.subheader(f"📂 Runs correspondants ({len(matching_runs)} trouvés)")
        st.markdown(f"**Configuration :** Engine: `{selected_engine}` | Activation: `{selected_activations}` | Couches: `{selected_layers}`")
        
        matching_runs = []
        engines_list = selected_engine if isinstance(selected_engine, list) else [selected_engine]
        for engine in engines_list:
            for act in selected_activations:
                for layer in selected_layers:
                    if layer in tab_runs.get(engine, {}).get(act, {}):
                        matching_runs.extend(tab_runs[engine][act][layer])

        matching_runs.sort(
            key=lambda x: (
                x["val_accuracy"] if x["val_accuracy"] is not None and pd.notna(x["val_accuracy"]) else -float('inf'),
                str(x["run_id"])
            ),
            reverse=True
        )

        # Transformation de la liste de dictionnaires en DataFrame pandas pour un affichage propre
        df_display = pd.DataFrame(matching_runs)[["activation", "val_accuracy", "val_loss", "learning_rate", "dropout_rate", "batch_norm", "epochs", "pca"]]
        df_display.columns = ["Activation (Layers)", "Val Accuracy", "Val Loss", "Learning Rate", "Dropout Rate", "Batch Norm", "Epochs", "PCA"]

        # Affichage sous forme de tableau interactif
        # st.dataframe(df_display, use_container_width=True, hide_index=True)
        
        # Affichage du tableau interactif avec activation de la sélection de ligne unique
        event = st.dataframe(
            df_display, 
            use_container_width=True, 
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="runs_table_interactive"
        )

        # Extraction robuste des lignes sélectionnées (compatible objet Streamlit)
        selected_rows = []
        logger.info(f"event = {event!r}") 

        if event is not None:
            sel = event.selection
            selected_rows = sel["rows"] if isinstance(sel, dict) else getattr(sel, "rows", [])

        logger.info(f"Lignes sélectionnées : {selected_rows}")

        # Affichage des détails si une ligne est cliquée
        if selected_rows:
            idx = selected_rows[0]
            if idx < len(matching_runs):
                chosen_run = matching_runs[idx]

                with st.expander(f"🔍 Détails du run : {chosen_run.get('run_name', 'Inconnu')} (ID: {chosen_run['run_id']})", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Run ID", chosen_run["run_id"])
                        st.metric("Learning Rate", chosen_run["learning_rate"])
                    with col2:
                        st.metric("Val Accuracy", f"{chosen_run['val_accuracy']:.4f}" if chosen_run['val_accuracy'] is not None else "N/A")
                        st.metric("Val Loss", f"{chosen_run['val_loss']:.4f}" if chosen_run['val_loss'] is not None else "N/A")
                    with col3:
                        st.metric("Dropout", chosen_run["dropout_rate"])
                        st.metric("Batch Norm", str(chosen_run["batch_norm"]))
                    
                    # st.json(chosen_run)
        
    elif show_matching and not matching_runs:
        st.warning("Aucun run ne correspond à cette combinaison exacte.")
    else:
        st.info("👈 Utilisez les filtres dans la barre latérale et cliquez sur **'Afficher les runs correspondants'** pour voir les détails au centre.")



# ------------------------------------------------------------------------------
# 2 · Distributions & Relations
# ------------------------------------------------------------------------------

with tab_autre:
    st.warning("Aucun runs MLflow.")


# ------------------------------------------------------------------------------
# 3 · Colinéarité & Sélection
# ------------------------------------------------------------------------------
    
