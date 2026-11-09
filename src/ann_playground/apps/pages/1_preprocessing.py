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
    page_title="ANN Playground · Data Studio",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

def cached_load_source_data(file_path: Path | Any) -> pd.DataFrame:
    """Fonction wrapper mise en cache pour Streamlit qui appelle la logique de chargement."""
    return load_source_data(file_path)

def cached_load_prepared_datasets() -> Tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Load train and test Parquet matrices from processed data directory."""
    return load_prepared_datasets()


# ==============================================================================
# Barre latérale : Source de données & Target
# ==============================================================================

st.title("ANN Playground : Data Studio")

with st.sidebar:
    with st.empty().container():
        st.caption("Module générique de préparation et diagnostic")

        data_source_mode = st.radio(
            "Source de données",
        # ["Fichier du projet (config)", "Téléverser un fichier local"],
        ["Fichier du projet (config)"],
        )

        if data_source_mode == "Fichier du projet (config)":
            if DATA.exists():
                df_raw = cached_load_source_data(DATA)
                st.success(f"Fichier chargé : `{DATA.name}`")
        else:
            uploaded = st.file_uploader("Fichier tabulaire", type=["csv", "parquet"])
            if uploaded is not None:
                df_raw = cached_load_source_data(uploaded)
                st.success(f"Fichier chargé : `{uploaded.name}`")

        column_candidates = ["Aucune"] + list(df_raw.columns)
        selected_target = st.selectbox(
            "Variable cible (Target)", column_candidates, index=len(column_candidates) - 1
        )
        target_name = None if selected_target == "Aucune" else selected_target


# ==============================================================================
# Structure principale en onglets
# ==============================================================================

# Injection CSS ciblant les data-testid stables de Streamlit
st.markdown("""
<style>
    /* 1. Élargit le conteneur principal de la page */
    [data-testid="block-container"] {
        max-width: 100% !important;
        padding-left: 3rem;
        padding-right: 3rem;
        padding-top: 2rem;
    }

    /* 2. Force la barre d'onglets à s'étirer sur toute la largeur */
    .stTabs [data-baseweb="tab-list"] {
        width: 100%;
        display: flex;
    }

    /* 3. Répartit uniformément l'espace entre les onglets */
    .stTabs [data-baseweb="tab"] {
        flex-grow: 1;
        justify-content: center;
    }
</style>
""", unsafe_allow_html=True)

tab_audit, tab_eda, tab_filter, tab_prep, tab_pca= st.tabs(
    [
        "1 · Diagnostic structurel",
        "2 · Distributions & Relations",
        "3 · Colinéarité & Sélection",
        "4 · Partition & Transformation",
        "5 · Analyse en Composantes Principales (PCA)",
    ]
)

if "export_ready" not in st.session_state:
    st.session_state["export_ready"] = False

# ------------------------------------------------------------------------------
# 1 · Diagnostic structurel
# ------------------------------------------------------------------------------
with tab_audit:
    st.subheader("Diagnostic de conformité du jeu de données")

    specs = special_columns(df_raw)
    typed_features = classify_feature_types(df_raw, target_name)
    duplicate_count = count_duplicates(df_raw)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Observations", df_raw.shape[0])
    m2.metric("Variables totales", df_raw.shape[1])
    m3.metric("Doublons stricts", duplicate_count)
    m4.metric("Colonnes constantes", len(specs["constant"]))
    m5.metric("Colonnes vides", len(specs["empty"]))

    st.markdown("#### Typage statistique réel inféré")
    t1, t2, t3 = st.columns(3)
    t1.info(
        f"**Quantitatives réelles ({len(typed_features['continuous'])})**\n\n"
        + ", ".join(typed_features["continuous"])
    )
    t2.warning(
        f"**Catégorielles / Codes entiers ({len(typed_features['categorical'])})**\n\n"
        + ", ".join(typed_features["categorical"])
    )
    t3.success(
        f"**Binaires / Drapeaux ({len(typed_features['binary'])})**\n\n"
        + ", ".join(typed_features["binary"])
    )

    st.markdown("#### Bilan des valeurs manquantes")
    df_missing = missing_summary(df_raw)
    cols_with_na = df_missing[df_missing["missing_count"] > 0]
    if cols_with_na.empty:
        st.success("Aucune valeur manquante détectée.")
    else:
        st.dataframe(cols_with_na, use_container_width=True)

    with st.expander("Aperçu des 10 premières lignes"):
        st.dataframe(df_raw.head(10), use_container_width=True)


# ------------------------------------------------------------------------------
# 2 · Distributions & Relations
# ------------------------------------------------------------------------------
with tab_eda:
    st.subheader("Exploration des distributions et séparabilité")

    c_left, c_right = st.columns(2)

    with c_left:
        st.markdown("##### Distribution univariée")
        var_to_plot = st.selectbox("Sélectionner une variable", list(df_raw.columns))
        is_num = pd.api.types.is_numeric_dtype(df_raw[var_to_plot])

        if is_num and df_raw[var_to_plot].nunique() > 15:
            fig_dist = px.histogram(
                df_raw,
                x=var_to_plot,
                nbins=35,
                marginal="box",
                color=(
                    target_name
                    if target_name and df_raw[target_name].nunique() <= 10
                    else None
                ),
                color_discrete_sequence=px.colors.qualitative.Prism,
            )
            st.plotly_chart(fig_dist, use_container_width=True)
        else:
            val_counts = df_raw[var_to_plot].astype(str).value_counts().reset_index()
            val_counts.columns = [var_to_plot, "Effectif"]
            fig_bar = px.bar(
                val_counts.head(20),
                x=var_to_plot,
                y="Effectif",
                color_discrete_sequence=["#439cc8"],
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    with c_right:
        st.markdown("##### Analyse de la variable cible")
        if target_name:
            target_series = df_raw[target_name]
            target_dist = target_series.value_counts(dropna=False).reset_index()
            target_dist.columns = [target_name, "Effectif"]
            target_dist["Proportion (%)"] = (
                target_dist["Effectif"] / len(df_raw) * 100
            ).round(2)

            fig_target = px.pie(
                target_dist,
                names=target_name,
                values="Effectif",
                hole=0.45,
                color_discrete_sequence=px.colors.qualitative.Safe,
            )
            st.plotly_chart(fig_target, use_container_width=True)
            st.dataframe(target_dist, use_container_width=True)
        else:
            st.info(
                "Définissez une cible dans la barre latérale pour activer la répartition."
            )


# ------------------------------------------------------------------------------
# 3 · Colinéarité & Sélection
# ------------------------------------------------------------------------------
with tab_filter:
    st.subheader("Audit de multicolinéarité linéaire")

    corr_threshold = st.slider(
        "Seuil de corrélation absolue (|r|)",
        min_value=0.60,
        max_value=0.98,
        value=0.85,
        step=0.01,
    )

    # numeric_df = df_raw.select_dtypes(include=[np.number])
    numeric_df = df_raw[typed_features['continuous']]
    pairs = correlated_features(numeric_df, threshold=corr_threshold)

    col_corr1, col_corr2 = st.columns([1, 1])

    with col_corr1:
        st.markdown(
            f"**Paires redondantes identifiées (|r| > {corr_threshold}) :** {len(pairs)}"
        )
        if pairs:
            df_pairs = pd.DataFrame(
                pairs, columns=["Feature 1", "Feature 2", "Corrélation"]
            )
            st.dataframe(
                df_pairs.style.format({"Corrélation": "{:.3f}"}),
                use_container_width=True,
            )
        else:
            st.success("Aucune paire de variables ne dépasse ce seuil.")

    with col_corr2:
        if target_name:
            st.markdown("##### Priorisation par corrélation à la cible")
            # Encodage numérique temporaire de la cible si catégorielle
            if not pd.api.types.is_numeric_dtype(df_raw[target_name]):
                y_temp = pd.Series(
                    LabelEncoder().fit_transform(df_raw[target_name].astype(str)),
                    index=df_raw.index,
                    name=target_name,
                )
            else:
                y_temp = df_raw[target_name]

            corrs = target_correlations(
                df_raw.drop(columns=[target_name]), y_temp, n_top=12
            )
            st.dataframe(
                corrs.to_frame("Corrélation absolue"), use_container_width=True
            )

            if st.button("Éliminer automatiquement les variables redondantes"):
                filtered_df, dropped_cols = drop_redundant_features(
                    df_raw, y_or_target=y_temp, threshold=corr_threshold
                )
                st.session_state["df_filtered"] = filtered_df
                st.session_state["dropped_features"] = dropped_cols
                st.warning(
                    f"Variables écartées ({len(dropped_cols)}) : {', '.join(dropped_cols)}"
                )

    # Matrice de corrélation
    if len(numeric_df.columns) > 1:
        corr_matrix = numeric_df.corr().round(2)
        fig_heat = px.imshow(
            corr_matrix,
            text_auto='.2f',
            aspect="equal",
            color_continuous_scale="RdBu_r",
            zmin=-1.0,
            zmax=1.0,
        )
        fig_heat.update_layout(
            width=1000,
            height=1000,
            xaxis_title="",
            yaxis_title=""
        )

        st.plotly_chart(fig_heat, use_container_width=True)

        st.markdown("---")
        st.subheader("📊 Interprétation de la matrice de corrélation")

        st.markdown("""
        Cette matrice de corrélation apporte des enseignements très clairs :
        """)

        # Utilisation d'onglets ou de blocs visuels pour structurer l'analyse
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            ### 1. Le bloc académique (Crédits & Unités)
            * **Forte redondance :** Corrélations intenses au centre (souvent $> 0.75$, voire $0.94$ entre S1 et S2 pour les crédits). Un étudiant qui s'inscrit et valide ses unités au premier semestre fait de même au second.
            * **Impact sur l'ACP :** La première composante principale ($PC_1$) va massivement capturer cette dimension de **volume et de progression académique**.
            
            ### 2. Le bloc des notes (`Grade`)
            * **Lien modéré :** Corrélation autour de $0.58$ pour l'admission/qualification et $0.68$ pour les notes semestrielles.
            * **Impact sur l'ACP :** Ces variables de performance pure formeront un axe de **"niveau académique"** ou de compétence intrinsèque.
            """)

        with col2:
            st.markdown("""
            ### 3. Indicateurs macroéconomiques
            * **Indépendance totale :** Valeurs proches de $0$ face aux variables académiques (max $-0.34$ entre chômage et PIB).
            * **Impact sur l'ACP :** Information totalement orthogonale. Ces variables exogènes structureront des composantes spécifiques (probablement $PC_3$ ou $PC_4$).
            """)

        # Encadré synthétique pour la conclusion sur le Réseau de Neurones
        st.info("""
        💡 **Faut-il supprimer des variables à cause de la multicolinéarité ?**  
        **Non.** Les réseaux de neurones (MLP) tolèrent très bien la redondance. Si deux variables sont fortement corrélées (ex: crédits S1 et S2), le réseau va simplement répartir les poids synaptiques de manière similaire sans "casser" l'optimisation par descente de gradient. De plus, l'ACP se chargera d'orthogonaliser l'espace.
        """)        


# ------------------------------------------------------------------------------
# 4 · Partition & Transformation
# ------------------------------------------------------------------------------
with tab_prep:
    st.subheader("Préparation, normalisation et export étanche")

    df_current = st.session_state.get("df_filtered", df_raw.copy())

    if target_name is None:
        st.error(
            "Une variable cible valide doit être sélectionnée dans la barre latérale pour segmenter le jeu de données."
        )
        st.stop()

    st.write(
        f"Dimensions actuelles : `{df_current.shape[0]}` lignes, `{df_current.shape[1]}` colonnes."
    )

    col_p1, col_p2, col_p3 = st.columns(3)
    test_ratio = col_p1.slider("Proportion du jeu de test", 0.10, 0.40, 0.20, 0.05)
    random_seed = col_p2.number_input("Graine aléatoire (seed)", value=42, step=1)
    use_stratify = col_p3.checkbox("Stratification sur la cible", value=True)

    # Réévaluation des types après d'éventuelles suppressions
    types_ready = classify_feature_types(df_current, target_name)
    num_features = types_ready["continuous"]
    cat_features = types_ready["categorical"]

    # Option manuelle pour surcharger les sélections de type
    with st.expander(
        "Ajustement manuel des affectations de colonnes pour le preprocessing"
    ):
        num_features = st.multiselect(
            "Variables à standardiser (numériques)",
            df_current.columns,
            default=num_features,
        )
        cat_features = st.multiselect(
            "Variables à encoder en One-Hot (catégorielles)",
            df_current.columns,
            default=cat_features,
        )

    if st.button("Générer les matrices Train / Test", type="primary"):
        X_df = df_current.drop(columns=[target_name])
        y_raw = df_current[target_name]

        strat_target = y_raw if use_stratify and y_raw.nunique() <= 20 else None

        # 1. Découpage train/test AVANT toute transformation
        X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
            X_df,
            y_raw,
            test_size=test_ratio,
            random_state=int(random_seed),
            stratify=strat_target,
        )

        # 2. Encodage de la target
        label_enc = LabelEncoder()
        y_train = label_enc.fit_transform(y_train_raw.astype(str))
        y_test = label_enc.transform(y_test_raw.astype(str))

        # # 3. Fit du ColumnTransformer uniquement sur X_train
        X_train_proc, X_test_proc, transformer = preprocess_data(
            X_train=X_train_raw,
            X_test=X_test_raw,
            numeric_columns=num_features,
            categorical_columns=cat_features,
        )

        # 4. Récupération des noms de colonnes transformées
        if hasattr(transformer, "get_feature_names_out"):
            encoded_cols = [str(c) for c in transformer.get_feature_names_out()]
        else:
            encoded_cols = [f"f_{i}" for i in range(X_train_proc.shape[1])]

        df_train_features = pd.DataFrame(X_train_proc, columns=encoded_cols)
        df_test_features = pd.DataFrame(X_test_proc, columns=encoded_cols)

        # # 5. Export unique au format Parquet
        train_path, test_path = export_train_test_parquet(
            X_train=df_train_features,
            X_test=df_test_features,
            y_train=y_train,
            y_test=y_test,
            output_dir=PROCESSED_DATA_DIR,
            target_name=target_name,
        )

        df_train_ready = df_train_features.copy()
        df_train_ready[target_name] = y_train

        df_test_ready = df_test_features.copy()
        df_test_ready[target_name] = y_test

        st.success(
            f"Matrices exportées avec succès dans `{PROCESSED_DATA_DIR}`\n\n"
            f"- Train set : `{df_train_features.shape[0]}` lignes, `{len(encoded_cols)}` features\n"
            f"- Test set  : `{df_test_features.shape[0]}` lignes\n"
            f"- Classes cibles : {dict(enumerate(label_enc.classes_))}"
        )
        st.dataframe(df_train_features.head(5), use_container_width=True)

        st.session_state["export_ready"] = True
        st.session_state["button_ready"] = True



# ------------------------------------------------------------------------------
# 5 · Analyse en Composantes Principales (PCA)
# ------------------------------------------------------------------------------

with tab_pca:

    if st.session_state.get("export_ready", False) == False:
        
        st.subheader("L’Analyse en Composantes Principales (PCA)")

        st.markdown(
            """
            L’Analyse en Composantes Principales est une méthode statistique de réduction de dimension, utilisée pour explorer et simplifier des
            jeux de données comportant de nombreuses variables corrélées entre elles.
            
            Son principe : projeter les données dans un nouvel espace dont les axes, appelés composantes principales, sont des combinaisons linéaires des 
            variables d’origine. Ces axes sont orthogonaux entre eux et ordonnés selon la part de variance qu’ils expliquent. 
            La première composante capte le maximum d’information, la deuxième le maximum de ce qui reste, 
            et ainsi de suite. En ne conservant que les premières composantes, on réduit considérablement la dimension des données en perdant un minimum d’information.
            """
        )

        st.markdown("#### Les 3 piliers méthodologiques :")

        st.markdown(
            """
            1. **Découpage train/test en amont** : Il faut séparer les données *avant* toute transformation, ce qui évite que des informations du jeu de test ne fuitent dans l'entraînement.
            2. **Fit du `ColumnTransformer` uniquement sur `X_train`** : *(Via `fit_transform` sur le train, puis `transform` seul sur le test).* Les moyennes, les encodages et les imputations sont calculés uniquement sur le train et appliqués sagement au test.
            3. **PCA appliquée après le preprocessing** : La PCA prend en entrée `X_train_prep` (qui est déjà proprement encodé et normalisé). C'est ce qu'il faut faire, car une PCA a besoin de variables numériques continues et centrées-réduites pour fonctionner correctement.
            """
        )

        st.info(
            "💡 **Bénéfice direct :** Les matrices réduites sont parfaitement orthogonales, "
            "stabilisent l'apprentissage du MLP et réduisent la complexité dimensionnelle sans compromettre l'intégrité des données."
        )

        st.markdown("#### A faire :")     

        st.markdown(
            """
            1. Sélectionner l'onglet **Partition & Transformation**.
            2. Cliquer sur le bonton **Générer les matrices Train/Test**
            3. Sélectionner l'obket **Analyse en Composantes Principales (PCA)**
            """
        )        
    
    else:

        X_train, X_test = cached_load_prepared_datasets()

        X_train_prep = X_train.drop(columns=[target_name]).values.astype(np.float32)
        y_train = X_train[target_name].values.astype(np.int64)

        X_test_prep = X_test.drop(columns=[target_name]).values.astype(np.float32)
        y_test = X_test[target_name].values.astype(np.int64)


        st.subheader("Analyse en Composantes Principales (PCA)")
        st.write(
            f"Dimensions actuelles : `{X_train_prep.shape[0]}` lignes, `{X_train_prep.shape[1]}` colonnes."
        )

        fig_pca, n_comp = plot_pca_cumulative_variance(X_train_prep, threshold=0.95)
        fig_scree, n_var, n_kaiser = scree_plot_with_ideal_components(X_train_prep, variance_threshold=0.95)

        c_left, c_right = st.columns(2, vertical_alignment="center")

        with c_left:
            st.markdown("##### La méthode des éboulis")
            st.pyplot(fig_scree)
            plt.close(fig_scree)
        with c_right:
            st.metric(label="Recommandation (95% de variance)", value=f"{n_var} composantes")
            st.caption(
                "**Recommandation (95 % de variance)** : nombre minimal de composantes principales "
                "à conserver pour que la variance cumulée expliquée atteigne 95 %. "
                "C'est le critère le plus utilisé quand on veut préserver l'essentiel de l'information "
                "du jeu de données."
            )
            st.metric(label="Critère de Kaiser (Valeur propre > 1)", value=f"{n_kaiser} composantes")
            st.caption(
                "**Critère de Kaiser** : on ne retient que les composantes dont la valeur propre est "
                "supérieure à 1, c'est-à-dire celles qui expliquent plus de variance qu'une variable "
                "d'origine standardisée prise isolément. Il est valable sur des données centrées-réduites "
                "et tend à être plus sévère (moins de composantes) que le seuil de 95 %."
            )
            
        n_components_choisi = st.slider(
            label="Nombre de composantes principales à conserver",
            min_value=1,
            max_value=int(X_train_prep.shape[1]),
            value=int(n_var),
            step=1,
            key="slider_pca_components",
            help="Glisse pour ajuster la dimensionnalité de l'espace d'entrée du modèle."
        )

        button_container = st.container()
        with button_container:
            # La condition normale : désactivé si export_ready est déjà True
            is_ready = st.session_state.get("button_ready", True) == False

            if st.button(
                "Générer les matrices Train / Test (PCA)", 
                type="primary",
                disabled=is_ready
            ):
                
                pca_personnalise = PCA(n_components=n_components_choisi, random_state=42)
                X_train_pca = pca_personnalise.fit_transform(X_train_prep)
                X_test_pca = pca_personnalise.transform(X_test_prep)

                train_file, test_file = export_pca_parquet(
                    X_train_pca=X_train_pca,
                    X_test_pca=X_test_pca,
                    y_train=y_train,
                    y_test=y_test,
                    output_dir=PROCESSED_DATA_DIR,
                    target_name=target_name
                )

                st.success(
                    f"✅ Matrices PCA exportées avec succès dans `{PROCESSED_DATA_DIR}`\n\n"
                    f"- **Train set (PCA)** : `{X_train_pca.shape[0]}` lignes, `{X_train_pca.shape[1]}` composantes principales\n"
                    f"- **Test set (PCA)** : `{X_test_pca.shape[0]}` lignes, `{X_test_pca.shape[1]}` composantes principales\n"
                    f"- **Cible (`{target_name}`)** : {len(np.unique(y_train))} classes uniques"
                )  

                st.session_state["button_ready"] = False

                time.sleep(5)
                st.rerun()