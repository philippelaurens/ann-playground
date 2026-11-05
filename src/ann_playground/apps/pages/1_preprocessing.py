from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler

# Supprimez st.set_page_config() d'ici (il doit rester uniquement dans app.py)

from ann_playground.config import DATA, CLEAN_DATA_DIR, PROCESSED_DATA_DIR



# ==============================================================================

@st.cache_data
def load_source_data(file_path: Path | Any) -> pd.DataFrame:
    """Charge un fichier CSV ou Parquet et nettoie les noms de colonnes."""
    if isinstance(file_path, (str, Path)):
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Fichier introuvable : {path}")
        df = (
            pd.read_csv(path, sep=None, engine="python")
            if path.suffix == ".csv"
            else pd.read_parquet(path)
        )
    else:
        name = file_path.name
        if name.endswith(".csv"):
            df = pd.read_csv(file_path, sep=None, engine="python")
        elif name.endswith(".parquet"):
            df = pd.read_parquet(file_path)
        else:
            raise ValueError("Format non supporté (CSV ou Parquet requis).")

    # Suppression systématique des espaces et tabulations masquées dans les entêtes
    df.columns = df.columns.str.strip()
    return df

def special_columns(df: pd.DataFrame, max_modalities: int = 20) -> dict[str, list[str]]:
    """Identify and categorize DataFrame columns according to structural features.

    Args:
        df: Input DataFrame to analyze.
        max_modalities: Maximum unique count to define low vs high cardinality.

    Returns:
        Dictionary mapping structural categories to lists of column names.
    """
    string_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    return {
        "empty": df.columns[df.isna().all()].tolist(),
        "constant": df.columns[df.nunique(dropna=False) <= 1].tolist(),
        "numeric": df.select_dtypes(include=["number"]).columns.tolist(),
        "categorical": string_cols,
        "high_cardinality": [
            c for c in string_cols if df[c].nunique(dropna=True) > max_modalities
        ],
        "boolean": [
            col
            for col in df.columns
            if set(df[col].dropna().unique()).issubset({True, False})
        ],
    }


def classify_feature_types(
    df: pd.DataFrame, target_col: str | None = None, cardinality_threshold: int = 15
) -> dict[str, list[str]]:
    """Isole les types statistiques réels en distinguant numériques continus et codes catégoriels."""
    cols = [c for c in df.columns if c != target_col]
    binary_cols: list[str] = []
    categorical_cols: list[str] = []
    continuous_cols: list[str] = []

    for col in cols:
        series = df[col].dropna()
        n_unique = series.nunique()
        unique_vals = set(series.unique())

        if unique_vals.issubset({0, 1}) and n_unique <= 2:
            binary_cols.append(col)
        elif pd.api.types.is_numeric_dtype(series):
            if n_unique <= cardinality_threshold:
                categorical_cols.append(col)
            else:
                continuous_cols.append(col)
        else:
            categorical_cols.append(col)

    return {
        "binary": binary_cols,
        "categorical": categorical_cols,
        "continuous": continuous_cols,
    }


def count_duplicates(df: pd.DataFrame, subset: list[str] | None = None) -> int:
    """Count total duplicated observations in a DataFrame.

    Args:
        df: Input DataFrame.
        subset: Column keys to consider.

    Returns:
        Total count of duplicated records.
    """
    return int(df.duplicated(subset=subset, keep="first").sum())


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute data types, missing value counts, and fill rates per column.

    Args:
        df: Input DataFrame to inspect.

    Returns:
        Summary DataFrame sorted by missing values in descending order.
    """
    return pd.DataFrame(
        {
            "type": df.dtypes,
            "missing_count": df.isna().sum(),
            "fill_rate_%": (df.count() / len(df) * 100).round(2),
        }
    ).sort_values(by="missing_count", ascending=False)


# ==============================================================================

def correlated_features(
    X: pd.DataFrame, threshold: float = 0.8
) -> list[tuple[str, str, float]]:
    """
    Identifie les paires de variables fortement corrélées entre elles.

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame contenant les variables numériques.
    threshold : float, default=0.8
        Seuil de corrélation absolue minimale pour retenir une paire.

    Returns
    -------
    list[tuple[str, str, float]]
        Liste de tuples (feature_1, feature_2, valeur_correlation) triés par corrélation.
    """
    X_numeric = X.select_dtypes(include=[np.number])
    corr_matrix = X_numeric.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    collinear = [
        (col, row, float(upper.loc[row, col]))
        for col in upper.columns
        for row in upper.index
        if upper.loc[row, col] > threshold
    ]
    return sorted(collinear, key=lambda item: item[2], reverse=True)


def target_correlations(
    X: pd.DataFrame, y: pd.Series | np.ndarray, n_top: int = 10
) -> pd.Series:
    """
    Calcule la corrélation linéaire absolue entre les variables numériques et la cible.

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame de variables explicatives.
    y : pd.Series | np.ndarray
        Variable cible numérique.
    n_top : int, default=10
        Nombre de variables les plus corrélées à retourner.

    Returns
    -------
    pd.Series
        Corrélations absolues triées par ordre décroissant.
    """
    X_numeric = X.select_dtypes(include=[np.number]).copy()
    target_values = y.values if isinstance(y, pd.Series) else y
    X_numeric["__target__"] = target_values

    correlations = (
        X_numeric.corr()["__target__"]
        .drop(labels=["__target__"])
        .abs()
        .sort_values(ascending=False)
    )
    return correlations.head(n_top)

def drop_redundant_features(
    X: pd.DataFrame,
    y_or_target: pd.Series | np.ndarray | str,
    threshold: float = 0.90,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Supprime les variables numériques redondantes en conservant, pour chaque paire
    fortement corrélée (|r| > threshold), la variable la plus corrélée à la cible.

    Les variables non numériques présentes dans X sont conservées sans modification.

    Parameters
    ----------
    X : pd.DataFrame
        Ensemble des variables explicatives (ou DataFrame complet contenant la cible).
    y_or_target : pd.Series | np.ndarray | str
        Nom de la colonne cible (si présente dans X) ou vecteur cible (Series / ndarray).
    threshold : float, default=0.90
        Seuil de corrélation linéaire absolue au-delà duquel deux variables
        sont jugées redondantes.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        - DataFrame nettoyé des colonnes redondantes.
        - Liste des noms des variables supprimées (triée par ordre alphabétique).
    """
    df_full = X.copy()

    if isinstance(y_or_target, str):
        target_col = y_or_target
        if target_col not in df_full.columns:
            raise ValueError(f"Target '{target_col}' absente du DataFrame fourni.")
    else:
        target_series = (
            y_or_target.copy()
            if isinstance(y_or_target, pd.Series)
            else pd.Series(y_or_target, index=df_full.index, name="target")
        )
        target_col = target_series.name or "target"
        df_full[target_col] = target_series.values

    # 1. Matrice de corrélation absolue entre features numériques
    features_df = df_full.drop(columns=[target_col])
    corr_matrix = features_df.corr(numeric_only=True).abs()

    # 2. Corrélation absolue de chaque variable numérique avec la cible
    target_corr = (
        df_full.corr(numeric_only=True).abs()[target_col].drop(labels=[target_col])
    )

    valid_cols = corr_matrix.columns.intersection(target_corr.index)
    corr_matrix = corr_matrix.loc[valid_cols, valid_cols]
    target_corr = target_corr.loc[valid_cols]

    to_drop: set[str] = set()
    cols = corr_matrix.columns

    # 3. Parcours des paires au-dessus du seuil et arbitrage
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            col_a = cols[i]
            col_b = cols[j]
            if corr_matrix.loc[col_a, col_b] > threshold:
                if target_corr[col_a] >= target_corr[col_b]:
                    to_drop.add(col_b)
                else:
                    to_drop.add(col_a)

    dropped_list = sorted(to_drop)
    reduced_df = features_df.drop(columns=dropped_list, errors="ignore")

    if isinstance(y_or_target, str):
        reduced_df[target_col] = df_full[target_col]

    return reduced_df, dropped_list



# ==============================================================================

def preprocess_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame | None = None,
    numeric_columns: list[str] | None = None,
    categorical_columns: list[str] | None = None,
    scaler: BaseEstimator | None = None,
) -> tuple[np.ndarray, np.ndarray | None, ColumnTransformer]:
    """Fit a ColumnTransformer on train data and transform both sets without leakage.

    Args:
        X_train: Training feature DataFrame used to fit scalers and encoders.
        X_test: Optional test feature DataFrame to transform.
        numeric_columns: Columns to scale. Auto-detected if None.
        categorical_columns: Categorical columns to one-hot encode. Auto-detected if None.
        scaler: Scikit-learn scaler instance (default: RobustScaler).

    Returns:
        Tuple of (processed_X_train, processed_X_test, fitted_column_transformer).

    Raises:
        ValueError: If no valid numerical or categorical features are provided.
    """
    if scaler is None:
        scaler = RobustScaler()

    if numeric_columns is None:
        numeric_columns = X_train.select_dtypes(include=[np.number]).columns.tolist()

    if categorical_columns is None:
        categorical_columns = X_train.select_dtypes(
            include=["object", "category", "string"]
        ).columns.tolist()

    transformers = []
    if numeric_columns:
        transformers.append(("num", clone(scaler), numeric_columns))
    if categorical_columns:
        transformers.append(
            (
                "cat",
                OneHotEncoder(
                    drop="first", handle_unknown="ignore", sparse_output=False
                ),
                categorical_columns,
            )
        )

    if not transformers:
        raise ValueError(
            "No valid numeric or categorical columns found for transformation."
        )

    column_transformer = ColumnTransformer(transformers=transformers, remainder="drop")

    X_train_processed = column_transformer.fit_transform(X_train)
    X_test_processed = (
        column_transformer.transform(X_test) if X_test is not None else None
    )

    return X_train_processed, X_test_processed, column_transformer




def export_train_test_parquet(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    y_test: pd.Series | np.ndarray,
    output_dir: Path | str = "data/processed_data",
    target_name: str = "target",
    transform_y: str | None = None,
    drop_cols: list[str] | None = None,
) -> tuple[Path, Path]:
    """Export prepared training and test sets as compressed Parquet files.

    Concatenates features and target into single tables matching the structure
    expected by ANN Lab ('train_prepared.parquet' and 'test_prepared.parquet').

    Args:
        X_train: Training features DataFrame.
        X_test: Test features DataFrame.
        y_train: Training target labels or continuous values.
        y_test: Test target labels or continuous values.
        output_dir: Directory where artifacts will be saved.
        target_name: Column name assigned to the target variable.
        transform_y: Optional transform ('log1p' supported).
        drop_cols: Columns to exclude from the export.

    Returns:
        Tuple of (train_filepath, test_filepath).
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    X_tr = X_train.drop(columns=drop_cols or [], errors="ignore").reset_index(drop=True)
    X_te = X_test.drop(columns=drop_cols or [], errors="ignore").reset_index(drop=True)

    y_tr_arr = y_train.values if isinstance(y_train, pd.Series) else np.asarray(y_train)
    y_te_arr = y_test.values if isinstance(y_test, pd.Series) else np.asarray(y_test)

    if transform_y == "log1p":
        y_tr_vals = np.log1p(y_tr_arr)
        y_te_vals = np.log1p(y_te_arr)
    else:
        y_tr_vals = y_tr_arr
        y_te_vals = y_te_arr

    X_tr[target_name] = y_tr_vals
    X_te[target_name] = y_te_vals

    train_path = out_path / "train_prepared.parquet"
    test_path = out_path / "test_prepared.parquet"

    X_tr.to_parquet(train_path, index=False)
    X_te.to_parquet(test_path, index=False)

    return train_path, test_path


# ==============================================================================
# Barre latérale : Source de données & Target
# ==============================================================================

with st.sidebar:
    st.title("Data Studio")
    st.caption("Module générique de préparation et diagnostic")

    data_source_mode = st.radio(
        "Source de données",
        ["Fichier du projet (config)", "Téléverser un fichier local"],
    )

    if data_source_mode == "Fichier du projet (config)":
        if DATA.exists():
            df_raw = load_source_data(DATA)
            st.success(f"Fichier chargé : `{DATA.name}`")
        else:
            st.error(f"Fichier introuvable à l'emplacement : {DATA}")
            st.stop()
    else:
        uploaded = st.file_uploader("Fichier tabulaire", type=["csv", "parquet"])
        if uploaded is not None:
            df_raw = load_source_data(uploaded)
            st.success(f"Fichier chargé : `{uploaded.name}`")
        else:
            st.info("Importez un fichier pour initialiser l'espace de travail.")
            st.stop()

    column_candidates = ["Aucune"] + list(df_raw.columns)
    selected_target = st.selectbox(
        "Variable cible (Target)", column_candidates, index=len(column_candidates) - 1
    )
    target_name = None if selected_target == "Aucune" else selected_target



# ==============================================================================
# Structure principale en onglets
# ==============================================================================

st.set_page_config(layout="wide")

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

tab_audit, tab_eda, tab_filter, tab_prep = st.tabs(
    [
        "1 · Diagnostic structurel",
        "2 · Distributions & Relations",
        "3 · Colinéarité & Sélection",
        "4 · Partition & Transformation",
    ]
)


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

    numeric_df = df_raw.select_dtypes(include=[np.number])
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
            text_auto=False,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1.0,
            zmax=1.0,
        )
        st.plotly_chart(fig_heat, use_container_width=True)


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
            target_name="target",
        )

        df_train_ready = df_train_features.copy()
        df_train_ready["target"] = y_train

        df_test_ready = df_test_features.copy()
        df_test_ready["target"] = y_test

        st.session_state["export_ready"] = True
        st.success(
            f"Matrices exportées avec succès dans `{PROCESSED_DATA_DIR}`\n\n"
            f"- Train set : `{df_train_features.shape[0]}` lignes, `{len(encoded_cols)}` features\n"
            f"- Test set  : `{df_test_features.shape[0]}` lignes\n"
            f"- Classes cibles : {dict(enumerate(label_enc.classes_))}"
        )
        st.dataframe(df_train_features.head(5), use_container_width=True)
