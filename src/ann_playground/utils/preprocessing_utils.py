# ==============================================================================
# Module de logique métier pour le prétraitement des données
# ==============================================================================

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler

# ------------------------------------------------------------------------------
# Utility functions for tabular data diagnostic, feature typing,
#   data cleaning, and dataset exporting using Pandas and Scikit-Learn.
# ------------------------------------------------------------------------------

"""
    - special_columns(df: pd.DataFrame, max_modalities: int) -> dict[str, list[str]]
    - classify_feature_types(df: pd.DataFrame, target_col: str | None = None, cardinality_threshold: int = 15) -> dict[str, list[str]]:
    - count_duplicates(df: pd.DataFrame, subset: list[str] | None) -> int
    - missing_summary(df: pd.DataFrame) -> pd.DataFrame
"""

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
    df: pd.DataFrame, 
    target_col: str | None = None, 
    cardinality_threshold: int = 15
) -> dict[str, list[str]]:
    """Isole les types statistiques réels en s'appuyant sur une liste explicite pour les codes catégoriels."""
    
    cols = [c for c in df.columns if c != target_col]
    
    # Liste explicite des colonnes catégorielles encodées en entiers ou en texte dans ce dataset
    explicit_categorical = {
        "Marital Status", "Application mode", "Application order", "Course", 
        "Previous qualification", "Nacionality", 
        "Mother's qualification", "Father's qualification", 
        "Mother's occupation", "Father's occupation",
        "Curricular units 1st sem (without evaluations)",
        "Curricular units 2nd sem (without evaluations)",
        "Daytime/evening attendance", "Displaced", "Educational special needs", 
        "Debtor", "Tuition fees up to date", "Gender", "Scholarship holder", 
        "International"
    }

    binary_cols: list[str] = []
    categorical_cols: list[str] = []
    continuous_cols: list[str] = []

    for col in cols:
        series = df[col].dropna()
        n_unique = series.nunique()
        unique_vals = set(series.unique())

        # 1. Priorité à la liste explicite
        if col in explicit_categorical:
            categorical_cols.append(col)
        
        # 2. Détection stricte des binaires (0 / 1)
        elif unique_vals.issubset({0, 1}) and n_unique <= 2:
            binary_cols.append(col)
            
        # 3. Traitement des numériques restants
        elif pd.api.types.is_numeric_dtype(series):
            if pd.api.types.is_float_dtype(series):
                continuous_cols.append(col)  # Vrais continus (taux, PIB, notes)
            elif n_unique <= cardinality_threshold:
                categorical_cols.append(col)  # Petits entiers discrets restants
            else:
                continuous_cols.append(col)  # Grands entiers numériques (ex: Age, Notes)
        else:
            categorical_cols.append(col)  # Colonnes textuelles (objets)

    return {
        "binary": binary_cols,
        "categorical": categorical_cols,
        "continuous": continuous_cols,
    }


def classify_feature_types_or(
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



# ------------------------------------------------------------------------------
# Outils d'analyse exploratoire, sélection de features, visualisations
#   et exportation de jeux de données.
# ------------------------------------------------------------------------------

"""
    - correlated_features()(X, threshold) -> list[tuple[str, str, float]]
    - target_correlations(X, y, n_top) -> pd.Series
    - drop_redundant_features(X, y_or_target, threshold) -> pd.DataFrame
"""

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


# ------------------------------------------------------------------------------
# Utility functions for fitting a ColumnTransformer on train data 
#   and transform both sets without leakage.
# ------------------------------------------------------------------------------

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
