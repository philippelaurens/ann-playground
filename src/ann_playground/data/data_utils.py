from pathlib import Path
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple
from ucimlrepo import fetch_ucirepo

from pathlib import Path
import numpy as np
import pandas as pd

import json
import joblib
from sklearn.compose import ColumnTransformer


from ann_playground.config import PROCESSED_DATA_DIR


def fetch_and_save_dataset(
    id: int,
    output_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series | pd.DataFrame]:
    """Fetch a dataset from the UCI Machine Learning Repository by its ID,

    save it locally as a CSV file, and return its components.

    Args:
        id: The UCI repository dataset ID.
        output_path: The file path where the combined CSV dataset will be saved.

    Returns:
        A tuple containing:
            - df: The combined DataFrame of features and targets.
            - X: The features DataFrame.
            - y: The target(s) Series or DataFrame.
    """
    # Fetch the dataset from the UCI repository using its unique identifier
    dataset = fetch_ucirepo(id=id)

    # Extract features and targets separately
    X = dataset.data.features
    y = dataset.data.targets

    # Combine features and targets into a single DataFrame for storage
    df = pd.concat([X, y], axis=1)
    # print(df.metadata)
    # print(df.variables)

    # Export the consolidated dataset to a CSV file without index
    df.to_csv(output_path, index=False)

    return df, X, y


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


def load_prepared_datasets() -> Tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Load train and test Parquet matrices from processed data directory."""
    train_file = PROCESSED_DATA_DIR / "train_prepared.parquet"
    test_file = PROCESSED_DATA_DIR / "test_prepared.parquet"

    if not train_file.exists() or not test_file.exists():
        return None, None

    return pd.read_parquet(train_file), pd.read_parquet(test_file)


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

    # Nettoyage des colonnes à exclure et réinitialisation des index
    X_tr = X_train.drop(columns=drop_cols or [], errors="ignore").reset_index(drop=True)
    X_te = X_test.drop(columns=drop_cols or [], errors="ignore").reset_index(drop=True)

    # Conversion en tableau 1D sécurisée (gère les Series pandas et les arrays 1D/2D numpy)
    y_tr_arr = (y_train.to_numpy() if isinstance(y_train, pd.Series) else np.asarray(y_train)).ravel()
    y_te_arr = (y_test.to_numpy() if isinstance(y_test, pd.Series) else np.asarray(y_test)).ravel()

    # Application de la transformation cible avec validation
    if transform_y == "log1p":
        y_tr_vals = np.log1p(y_tr_arr)
        y_te_vals = np.log1p(y_te_arr)
    elif transform_y is not None:
        raise ValueError(f"Transformée non supportée : '{transform_y}'. Utilisez 'log1p' ou None.")
    else:
        y_tr_vals = y_tr_arr
        y_te_vals = y_te_arr

    # Ajout de la cible dans les DataFrames
    X_tr[target_name] = y_tr_vals
    X_te[target_name] = y_te_vals

    # Chemins de sortie
    train_path = out_path / "train_prepared.parquet"
    test_path = out_path / "test_prepared.parquet"

    # Export au format Parquet
    X_tr.to_parquet(train_path, index=False)
    X_te.to_parquet(test_path, index=False)

    return train_path, test_path


def export_pca_parquet(
    X_train_pca: np.ndarray,
    X_test_pca: np.ndarray,
    y_train: pd.Series | np.ndarray,
    y_test: pd.Series | np.ndarray,
    output_dir: Path | str = "data/processed_data",
    target_name: str = "Target",
) -> tuple[Path, Path]:
    """
        Exporte les matrices transformées par ACP (PCA) ainsi que leurs variables 
        cibles associées vers des fichiers au format Parquet.

        Cette fonction convertit les tableaux NumPy issus de la PCA en DataFrames 
        pandas nommés dynamiquement (PC_1, PC_2, etc.), y joint les étiquettes cibles, 
        crée le répertoire de destination si nécessaire et sauvegarde le tout.

        Parameters
        ----------
        X_train_pca : np.ndarray
            Tableau NumPy contenant les données d'entraînement réduites par PCA.
        X_test_pca : np.ndarray
            Tableau NumPy contenant les données de test réduites par PCA.
        y_train : pd.Series or np.ndarray
            Vecteur ou série de la variable cible pour l'entraînement.
        y_test : pd.Series or np.ndarray
            Vecteur ou série de la variable cible pour le test.
        output_dir : Path or str, default="data/processed_data"
            Chemin du répertoire de sortie où enregistrer les fichiers Parquet.
        target_name : str, default="target"
            Nom de la colonne attribuée à la variable cible dans les fichiers finaux.

        Returns
        -------
        tuple[Path, Path]
            Un tuple contenant le chemin absolu ou relatif vers le fichier Parquet 
            d'entraînement (`train_path`) et vers le fichier Parquet de test (`test_path`).
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    train_path = out_path / "train_prepared.parquet"
    test_path = out_path / "test_prepared.parquet"   

    # Conversion des tableaux NumPy PCA en DataFrames pandas avec des noms de colonnes explicites
    cols = [f"PC_{i+1}" for i in range(X_train_pca.shape[1])]
    
    df_train_pca = pd.DataFrame(X_train_pca, columns=cols)
    df_test_pca = pd.DataFrame(X_test_pca, columns=cols)

    # 2. Ajout de la variable cible
    df_train_pca[target_name] = y_train.values if hasattr(y_train, "values") else y_train
    df_test_pca[target_name] = y_test.values if hasattr(y_test, "values") else y_test

    # 3. Export au format Parquet
    df_train_pca.to_parquet(train_path, index=False)
    df_test_pca.to_parquet(test_path, index=False)

    return train_path, test_path