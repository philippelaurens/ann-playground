from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import logging
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from ann_playground.config import PROCESSED_DATA_DIR, EXPERIMENT_NAME
from ann_playground.model.keras_mlp import StreamlitEpochCallback, build_keras_mlp
from ann_playground.model.numpy_mlp import NumpyMLP

from ann_playground.data.data_utils import (
    load_prepared_datasets,
)
from ann_playground.data.data_visualisation import (
    plot_learning_curves,
)
from ann_playground.utils.tracking_utils import (
    init_mlflow_experiment,
    log_keras_training,
    log_numpy_training,
)
from ann_playground.services.mlflow_utils import (
    fetch_available_runs
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    force=True
)


def cached_load_prepared_datasets() -> Tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Load train and test Parquet matrices from processed data directory."""
    return load_prepared_datasets()


# ---------------------------------------------------------------------------
# Authentification
# ---------------------------------------------------------------------------

# Vérification de sécurité obligatoire sur chaque sous-pages
if not st.session_state.get("authenticated", False):
    st.warning("⚠️ Veuillez vous connecter depuis la page d'accueil pour accéder à cette section.")
    st.stop()


# ---------------------------------------------------------------------------
# Start MLFlow
# ---------------------------------------------------------------------------

try:
    experiment_id = init_mlflow_experiment(EXPERIMENT_NAME)
except Exception as err:
    st.warning(f"MLflow service connection unavailable: {err}. Runs will attempt reconnect on launch.")
    experiment_id = "0"


# ---------------------------------------------------------------------------
# UI Setup and Sidebar
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="ANN Playground · ANN Lab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("ANN Playground : ANN Lab")

df_train, df_test = cached_load_prepared_datasets()

with st.sidebar:
    with st.empty().container():    
        st.caption("Conception et évaluation de topologies MLP")

        engine = st.selectbox("Moteur de calcul", ["Keras (TensorFlow)", "NumPy MLP (POO)"])

        st.divider()
        st.markdown("#### Hyperparamètres d'architecture")

        n_layers = st.slider("Nombre de couches cachées", 1, 4, 2)
        hidden_units: List[int] = []
        cols = st.columns(n_layers)
        for i in range(n_layers):
            default_val = 64 // (2**i) if 64 // (2**i) >= 8 else 8
            units = cols[i].number_input(
                f"L{i + 1}", min_value=4, max_value=256, value=default_val, step=4
            )
            hidden_units.append(int(units))

        activation_fn = st.selectbox("Fonction d'activation", ["relu", "tanh", "sigmoid"])
        learning_rate = st.select_slider(
            "Taux d'apprentissage (lr)",
            options=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
            value=0.001,
        )

        if engine == "Keras (TensorFlow)":
            dropout = st.slider("Taux de Dropout", 0.0, 0.5, 0.2, 0.05)
            batch_norm = st.checkbox("Batch Normalization", value=True)
        else:
            dropout = 0.0
            batch_norm = False
            st.info(
                "Le moteur NumPy exécute une descente de gradient matricielle avec Softmax."
            )

        epochs = st.slider(
            "Nombre d'epochs", min_value=10, max_value=300, value=60, step=10
        )
        batch_size = st.select_slider(
            "Taille de batch", options=[16, 32, 64, 128, 256], value=32
        )


# ==============================================================================
# Structure principale en onglets
# ==============================================================================

if df_train is None or df_test is None:
    st.warning(
        "Aucune matrice de données n'a été détectée dans `data/processed_data/`."
    )
    st.info("Exécutez `1_data_studio.py` pour partitionner et prétraiter les données.")
    st.stop()

X_train = df_train.drop(columns=["Target"]).values.astype(np.float32)
y_train = df_train["Target"].values.astype(np.int64)

X_test = df_test.drop(columns=["Target"]).values.astype(np.float32)
y_test = df_test["Target"].values.astype(np.int64)


n_features = X_train.shape[1]
n_classes = len(np.unique(y_train))

st.subheader("Environnement expérimental")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Features en entrée", n_features)
c2.metric("Classes cibles", n_classes)
c3.metric("Échantillons Train", len(X_train))
c4.metric("Échantillons Test", len(X_test))

tab_train, tab_eval, tab_diagnose = st.tabs(
    [
        "1 · Entraînement en direct",
        "2 · Évaluation & Matrice de confusion",
        "3 · Diagnostic de généralisation",
    ]
)


# ------------------------------------------------------------------------------
# 1 · Entraînement
# ------------------------------------------------------------------------------
with tab_train:
    st.markdown("#### Lancement de l'apprentissage")
    launch_btn = st.button("Démarrer l'entraînement", type="primary")

    progress_bar = st.empty()
    chart_spot = st.empty()

    history = None

    if launch_btn:
        if engine == "Keras (TensorFlow)":
            model = build_keras_mlp(
                input_dim=n_features,
                num_classes=n_classes,
                hidden_layers=hidden_units,
                activation=activation_fn,
                dropout_rate=dropout,
                use_batch_norm=batch_norm,
                learning_rate=learning_rate,
            )

            cb = StreamlitEpochCallback(
                total_epochs=epochs,
                chart_placeholder=chart_spot,
                progress_bar=progress_bar,
            )

            history = model.fit(
                X_train,
                y_train,
                validation_data=(X_test, y_test),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[cb],
                verbose=0,
            )

            y_pred = np.argmax(model.predict(X_test), axis=1)
            test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

            class_labels = [f"Classe {c}" for c in sorted(np.unique(y_test))]
            cm = confusion_matrix(y_test, y_pred)
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                x=class_labels,
                y=class_labels,
                color_continuous_scale="Blues",
                title=f"Confusion Matrix - Keras ({activation_fn})",
            )
            report_dict = classification_report(
                y_test, y_pred, target_names=class_labels, output_dict=True
            )

            run_params = {
                "engine": "keras",
                "hidden_layers": str(hidden_units),
                "activation": activation_fn,
                "learning_rate": learning_rate,
                "dropout_rate": dropout,
                "batch_norm": batch_norm,
                "epochs": epochs,
                "batch_size": batch_size,
            }
            metrics_summary = {
                "test_loss": float(test_loss),
                "test_accuracy": float(test_acc),
                "best_epoch": int(pd.DataFrame(history.history)["val_loss"].idxmin()) + 1
            }

            log_keras_training(
                run_name=f"keras_mlp_{activation_fn}_lr{learning_rate}",
                params=run_params,
                model=model,
                history=history.history,
                metrics_test=metrics_summary,
                fig_cm=fig_cm,
                report_dict=report_dict,
            )

            st.session_state["trained_model"] = model
            st.session_state["history_df"] = pd.DataFrame(history.history)
            st.session_state["history_obj"] = pd.DataFrame(history.history)
            st.session_state["y_pred"] = y_pred
            st.session_state["engine"] = "keras"
            st.success("Entraînement Keras terminé et enregistré dans MLflow.")

        else:
            y_train_oh = np.eye(n_classes)[y_train]
            y_test_oh = np.eye(n_classes)[y_test]

            np_model = NumpyMLP(
                [n_features] + hidden_units + [n_classes], activation=activation_fn
            )
            records: List[Dict[str, float]] = []

            for ep in range(epochs):
                indices = np.random.permutation(len(X_train))
                X_shuffled = X_train[indices]
                y_shuffled = y_train_oh[indices]

                for b in range(0, len(X_train), batch_size):
                    xb = X_shuffled[b : b + batch_size]
                    yb = y_shuffled[b : b + batch_size]
                    np_model.train_step(xb, yb, lr=learning_rate)

                train_loss, train_acc = np_model.evaluate(X_train, y_train_oh)
                val_loss, val_acc = np_model.evaluate(X_test, y_test_oh)

                records.append(
                    {
                        "loss": train_loss,
                        "val_loss": val_loss,
                        "accuracy": train_acc,
                        "val_accuracy": val_acc,
                    }
                )

                if (ep + 1) % 5 == 0 or (ep + 1) == epochs:
                    progress_bar.progress(
                        (ep + 1) / epochs, text=f"Epoch {ep + 1}/{epochs}"
                    )
                    df_h = pd.DataFrame(records)
                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            y=df_h["loss"],
                            name="Train Loss",
                            line=dict(color="#439cc8"),
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            y=df_h["val_loss"],
                            name="Val Loss",
                            line=dict(color="#f08a5d", dash="dash"),
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            y=df_h["accuracy"],
                            name="Train Accuracy",
                            yaxis="y2",
                            line=dict(color="#2ca02c"),
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            y=df_h["val_accuracy"],
                            name="Val Accuracy",
                            yaxis="y2",
                            line=dict(color="#d62728", dash="dash"),
                        )
                    )
                    fig.update_layout(
                        title="Courbes d'apprentissage NumPy",
                        xaxis=dict(title="Epoch"),
                        yaxis=dict(title="Loss", rangemode="tozero"),
                        yaxis2=dict(
                            title="Accuracy",
                            overlaying="y",
                            side="right",
                            range=[0.0, 1.0],
                        ),
                        height=400,
                    )
                    chart_spot.plotly_chart(fig, width='stretch')

            test_loss, test_acc = np_model.evaluate(X_test, y_test_oh)
            y_pred = np_model.predict(X_test)

            class_labels = [f"Classe {c}" for c in sorted(np.unique(y_test))]
            cm = confusion_matrix(y_test, y_pred)
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                x=class_labels,
                y=class_labels,
                color_continuous_scale="Blues",
                title=f"Confusion Matrix - NumPy ({activation_fn})",
            )
            report_dict = classification_report(
                y_test, y_pred, target_names=class_labels, output_dict=True
            )

            run_params = {
                "engine": "numpy_oop",
                "hidden_layers": str(hidden_units),
                "activation": activation_fn,
                "learning_rate": learning_rate,
                "epochs": epochs,
                "batch_size": batch_size,
            }
            metrics_summary = {
                "test_loss": float(test_loss),
                "test_accuracy": float(test_acc),
            }

            log_numpy_training(
                run_name=f"numpy_mlp_{activation_fn}_lr{learning_rate}",
                model=np_model,
                params=run_params,
                history=records,
                metrics_test=metrics_summary,
                fig_cm=fig_cm,
                report_dict=report_dict,
            )

            st.session_state["trained_model"] = np_model
            st.session_state["history_df"] = pd.DataFrame(records)
            st.session_state["y_pred"] = y_pred
            st.session_state["engine"] = "numpy"
            st.success("Entraînement NumPy terminé et enregistré dans MLflow.")

    if "history_df" in st.session_state and not launch_btn:
        df_h = st.session_state["history_df"]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(y=df_h["loss"], name="Train Loss", line=dict(color="#439cc8"))
        )
        fig.add_trace(
            go.Scatter(
                y=df_h["val_loss"],
                name="Val Loss",
                line=dict(color="#f08a5d", dash="dash"),
            )
        )
        fig.add_trace(
            go.Scatter(
                y=df_h["accuracy"],
                name="Train Accuracy",
                yaxis="y2",
                line=dict(color="#2ca02c"),
            )
        )
        fig.add_trace(
            go.Scatter(
                y=df_h["val_accuracy"],
                name="Val Accuracy",
                yaxis="y2",
                line=dict(color="#d62728", dash="dash"),
            )
        )
        fig.update_layout(
            xaxis=dict(title="Epoch"),
            yaxis=dict(title="Loss", rangemode="tozero"),
            yaxis2=dict(
                title="Accuracy", overlaying="y", side="right", range=[0.0, 1.0]
            ),
            height=400,
        )
        chart_spot.plotly_chart(fig, width='stretch')

# ------------------------------------------------------------------------------
# 2 · Évaluation
# ------------------------------------------------------------------------------
with tab_eval:
    st.subheader("Performance détaillée sur l'ensemble de test")

    if "y_pred" not in st.session_state:
        st.info("Lancez un entraînement pour visualiser le rapport de classification.")
    else:
        y_pred = st.session_state["y_pred"]
        class_labels = [f"Classe {c}" for c in sorted(np.unique(y_test))]

        col_rep, col_mat = st.columns([1, 1])

        with col_rep:
            st.markdown("##### Rapport de classification")
            report_dict = classification_report(
                y_test, y_pred, target_names=class_labels, output_dict=True
            )
            df_report = pd.DataFrame(report_dict).transpose().round(3)
            st.dataframe(df_report, width='stretch')

        with col_mat:
            st.markdown("##### Matrice de confusion")
            cm = confusion_matrix(y_test, y_pred)
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                x=class_labels,
                y=class_labels,
                color_continuous_scale="Blues",
                labels=dict(x="Prédit", y="Réel", color="Nombre"),
            )
            fig_cm.update_layout(height=380, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_cm, width='stretch')

    if "history_obj" in st.session_state and not st.session_state["history_obj"].empty:
        st.subheader("Courbes d'apprentissage")

        df = st.session_state["history_obj"]
        col_center, _ = st.columns([0.9, 0.1])
        with col_center:
            fig = plot_learning_curves(df, "Learning Curve")
            st.pyplot(fig)


# ------------------------------------------------------------------------------
# 3 · Diagnostic
# ------------------------------------------------------------------------------
with tab_diagnose:
    st.subheader("Audit du surapprentissage / sous-apprentissage")

    if "history_df" not in st.session_state:
        st.info("Entraînez un modèle pour auditer le comportement de généralisation.")
    else:
        hist = st.session_state["history_df"]
        final_train_loss = float(hist["loss"].iloc[-1])
        final_val_loss = float(hist["val_loss"].iloc[-1])
        final_train_acc = float(hist["accuracy"].iloc[-1])
        final_val_acc = float(hist["val_accuracy"].iloc[-1])

        gap_acc = final_train_acc - final_val_acc
        min_val_loss_epoch = int(hist["val_loss"].idxmin()) + 1

        d1, d2, d3 = st.columns(3)
        d1.metric("Écart Train / Val (Accuracy)", f"{gap_acc * 100:.2f} %")
        d2.metric("Meilleure époque (Val Loss)", f"Epoch {min_val_loss_epoch}")
        d3.metric("Val Loss finale", f"{final_val_loss:.4f}")

        st.markdown("#### Diagnostic automatique")
        if gap_acc > 0.12 or (final_val_loss > hist["val_loss"].min() * 1.25):
            # st.error(
            #     "Surapprentissage (Overfitting) détecté : La perte de validation remonte ou l'écart "
            #     "d'exactitude entre le train et le test dépasse 12 %. Préconisations : augmenter le Dropout, "
            #    "ajouter de la régularisation L2, ou réduire le nombre de neurones/couches."
            # )

            # Mettre en valeur le constat du sur-apprentissage
            st.warning("""
            ⚠️ **Constat d'Overfitting :**  
            Le réseau de neurones a commencé à apprendre par cœur les particularités et le bruit du jeu d'entraînement, au détriment de sa capacité de généralisation sur des données qu'il n'a jamais vues (la validation).
            """)

            st.markdown("### 🛠️ Leviers pour corriger le sur-apprentissage dans Keras (MLP)")
            st.markdown("""
            Pour remédier à ce sur-apprentissage, plusieurs leviers permettent de corriger rapidement ce type d'écart dans un modèle Keras (MLP) :
            """)

            # Affichage des 4 leviers sous forme de colonnes ou de blocs Markdown
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("""
                #### 1. Ajouter du Dropout dans l'architecture
                Insérer des couches de `Dropout` (par exemple `layers.Dropout(0.3)` ou `0.5`) entre les couches denses permet de désactiver aléatoirement un pourcentage de neurones à chaque étape d'entraînement, forçant le réseau à trouver des motifs plus robustes.

                #### 2. Mettre en place un Early Stopping
                Si on lance l'entraînement avec un nombre d'époques très élevé, il est très probable que le modèle continue d'apprendre sur le train alors que les performances sur la validation stagnent ou se dégradent. L'arrêt précoce stoppe l'entraînement dès que la `val_loss` cesse de s'améliorer.
                """)

            with col2:
                st.markdown("""
                #### 3. Utiliser une régularisation des poids (L2)
                Ajouter une pénalité L2 (`kernel_regularizer=tf.keras.regularizers.l2(0.001)`) sur les couches denses permet d'éviter que les poids synaptiques ne s'envolent vers des valeurs trop élevées.

                #### 4. Alléger l'architecture du réseau
                Si le réseau comporte trop de couches ou trop de neurones par rapport à la taille du dataset, le réduire un peu l'empêchera de mémoriser les données par cœur.
                """)
        elif final_train_acc < 0.65:
            st.warning(
                "Sous-apprentissage (Underfitting) probable : La capacité du modèle est insuffisante "
                "pour séparer les classes. Préconisations : augmenter le nombre d'époques, ajouter des "
                "neurones par couche ou ajuster le taux d'apprentissage."
            )
        else:
            st.success(
                "Généralisation équilibrée : Les courbes de train et de validation convergent de "
                "manière cohérente sans divergence marquée."
            )