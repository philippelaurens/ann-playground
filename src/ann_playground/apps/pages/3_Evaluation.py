import os
from pathlib import Path
from typing import Any, Dict, List, Tuple
import re
import ast
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
    build_structured_runs_table,
    load_parameters_lists_for_run,
    load_feature_lists_for_run,
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


def _to_list(x):
    return ast.literal_eval(x) if isinstance(x, str) else list(x)

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

tab_diag, tab_overfitting, tab_acc_vs_loss, tab_hyperparams= st.tabs(
    [
        "1 · Configuration du Run MLflow",
        "2 · Analyse du sur-apprentissage",
        "3 - Compromis accuracy / loss",
        "4 - Effet moyen d'un hyperparamètre"                                                             
    ]
)

# ------------------------------------------------------------------------------
# 1 · Diagnostic structurel
# ------------------------------------------------------------------------------

df_display = None

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
        df_display = pd.DataFrame(matching_runs)[["activation", "hidden_layers", "val_accuracy", "val_loss", "learning_rate", "dropout_rate", "batch_norm", "epochs", "pca"]]
        df_display.columns = ["Activation", "Layers", "Val Accuracy", "Val Loss", "Learning Rate", "Dropout Rate", "Batch Norm", "Epochs", "PCA"]

        # Affichage sous forme de tableau interactif
        # st.dataframe(df_display, width='stretch', hide_index=True)
        
        # Affichage du tableau interactif avec activation de la sélection de ligne unique
        event = st.dataframe(
            df_display, 
            width='stretch', 
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="runs_table_interactive"
        )

        # Extraction robuste des lignes sélectionnées (compatible objet Streamlit)
        selected_rows = []

        if event is not None:
            sel = event.selection
            selected_rows = sel["rows"] if isinstance(sel, dict) else getattr(sel, "rows", [])

        # Affichage des détails si une ligne est cliquée
        if selected_rows:
            idx = selected_rows[0]
            if idx < len(matching_runs):
                chosen_run = matching_runs[idx]

                parameters, metrics = load_parameters_lists_for_run(chosen_run['run_id'])
                num_features, cat_features, matrix_confusion_content = load_feature_lists_for_run(chosen_run['run_id'])
                label_encoder = load_label_encoder(chosen_run['run_id'])

                with st.expander(f"🔍 Détails du run : {chosen_run.get('run_name', 'Inconnu')} (ID: {chosen_run['run_id']})", expanded=True):

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Val Accuracy", f"{metrics['val_accuracy']:.4f}" if metrics['val_accuracy'] is not None else "N/A")
                    with col2:
                        st.metric("Val Loss", f"{metrics['val_loss']:.4f}" if metrics['val_loss'] is not None else "N/A")
                    with col3:
                        st.metric("Test Accuracy", f"{metrics['test_accuracy']:.4f}" if metrics['test_accuracy'] is not None else "N/A")
                    with col4:
                        st.metric("Test Loss", f"{metrics['test_loss']:.4f}" if metrics['test_loss'] is not None else "N/A")
                    
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
                                        class_name += f" (Classe {class_idx})"
                                    else:
                                        logger.warning(f"Index {class_idx} hors limites pour '{class_raw}'")
                                        class_name = str(class_raw)
                                else:
                                    # Pas de chiffre : c'est déjà un libellé
                                    class_name = str(class_raw)
                            else:
                                class_name = str(class_raw)
                        
                            if class_name not in metrics_data:
                                metrics_data[class_name] = {"Classe": class_name}
                                
                            metrics_data[class_name][metric] = v

                    df_metrics = pd.DataFrame(list(metrics_data.values()))

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

                    st.iframe(matrix_confusion_content, height=550)

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
                
    elif show_matching and not matching_runs:
        st.warning("Aucun run ne correspond à cette combinaison exacte.")
    else:
        st.info("👈 Utilisez les filtres dans la barre latérale et cliquez sur **'Afficher les runs correspondants'** pour voir les détails au centre.")



# ------------------------------------------------------------------------------
# 2 · Distributions & Relations
# ------------------------------------------------------------------------------

if show_matching and matching_runs:

    if df_display is not None and not df_display.empty:
        df_viz = df_display.copy()

        layers_list = df_viz["Layers"].apply(_to_list)
        df_viz["Layers_str"] = layers_list.astype(str)
        df_viz["Profondeur"] = layers_list.apply(len)
        df_viz["Largeur 1re couche"] = layers_list.apply(lambda l: l[0])

        # "20 (best: 15)" -> total=20, best=15
        ep = df_viz["Epochs"].astype(str).str.extract(r"(\d+)\s*\(best:\s*(\d+)\)").astype(float)
        df_viz["Epochs total"] = ep[0]
        df_viz["Best epoch"] = ep[1]
        df_viz["Ratio best/total"] = df_viz["Best epoch"] / df_viz["Epochs total"]

        df_viz["PCA actif"] = df_viz["PCA"].astype(str).str.startswith("oui")
        df_viz["Run"] = (
            df_viz["Activation"] + " " + df_viz["Layers_str"]
            + " | do=" + df_viz["Dropout Rate"].astype(str)
            + " | #" + df_viz.index.astype(str)
        )
    elif show_matching and not matching_runs:
        st.warning("Aucun run ne correspond à cette combinaison exacte.")
    else:
        st.info("👈 Utilisez les filtres dans la barre latérale et cliquez sur **'Afficher les runs correspondants'** pour voir les détails au centre.")


with tab_overfitting:

    if df_display is not None and not df_display.empty:

        st.subheader(f"Analyse du sur-apprentissage")

        st.markdown(
            "Ratio proche de 0 : la meilleure epoch arrive très tôt, le reste de "
                        "l'entraînement dégrade la validation (sur-apprentissage ou early stopping trop patient)."
        )

        d = df_viz.dropna(subset=["Ratio best/total"]).sort_values("Ratio best/total")
        fig = px.bar(
            d, x="Ratio best/total", y="Run", orientation="h",
            color="Val Loss", color_continuous_scale="RdYlGn_r",
            hover_data=["Best epoch", "Epochs total", "Val Accuracy"],
        )
        fig.update_layout(height=max(350, 28 * len(d)), yaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("👈 Utilisez les filtres dans la barre latérale et cliquez sur **'Afficher les runs correspondants'** pour voir les détails au centre.")


with tab_acc_vs_loss:

    if df_display is not None and not df_display.empty:

        st.subheader(f"Compromis accuracy / loss")

        st.markdown(f"**Les meilleurs runs sont en haut à gauche (accuracy haute, loss basse).**")
        
        fig = px.scatter(
            df_viz, x="Val Loss", y="Val Accuracy",
            color="Dropout Rate", symbol="Activation", size="Epochs total",
            hover_name="Run",
            hover_data=["Layers_str", "Best epoch", "PCA"],
            color_continuous_scale="Viridis",
        )
        fig.update_layout(
            coloraxis_colorbar=dict(
                title="Dropout",
                y=0.35,          # position verticale (0 = bas, 1 = haut)
                len=0.5,         # hauteur de la barre
                yanchor="middle",
            ),
            legend=dict(
                y=1, yanchor="top",
            ),
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("👈 Utilisez les filtres dans la barre latérale et cliquez sur **'Afficher les runs correspondants'** pour voir les détails au centre.")


with tab_hyperparams:

    if df_display is not None and not df_display.empty:

        st.subheader(f"Effet moyen d'un hyperparamètre")

        st.markdown(f"**Avec n petit, une différence inférieure à l'écart-type n'est pas concluante.**")
        
        factor = st.selectbox(
            "Paramètre à analyser",
            ["Activation", "Layers_str", "Dropout Rate", "PCA actif", "Epochs total"],
        )
        metric = st.radio("Métrique", ["Val Accuracy", "Val Loss"], horizontal=True)
        agg = (
            df_viz.groupby(factor)[metric]
            .agg(["mean", "std", "count"]).reset_index()
        )
        agg["std"] = agg["std"].fillna(0)
        fig = px.bar(
            agg, x=factor, y="mean", error_y="std", text="count",
            labels={"mean": f"{metric} (moyenne ± écart-type)"},
        )
        fig.update_traces(texttemplate="n=%{text}", textposition="outside")
        fig.update_xaxes(type="category")
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("👈 Utilisez les filtres dans la barre latérale et cliquez sur **'Afficher les runs correspondants'** pour voir les détails au centre.")



