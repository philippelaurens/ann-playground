import json
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import LabelEncoder

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA


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


def load_prepared_datasets(
    output_dir: Path | str = PROCESSED_DATA_DIR
) -> tuple[
    pd.DataFrame | None,
    pd.DataFrame | None,
    ColumnTransformer | None,
    PCA | None,
    list[str] | None,
    list[str] | None
]:
    """Charge automatiquement les jeux de données (standard ou PCA), 
    les artefacts de preprocessing et les listes de features associées.

    Returns:
        Tuple contenant :
            - df_train (pd.DataFrame | None)
            - df_test (pd.DataFrame | None)
            - transformer (ColumnTransformer | None)
            - pca (PCA | None)
            - num_features (list[str] | None)
            - cat_features (list[str] | None)
    """
    out_path = Path(output_dir)
    config_path = out_path / "pipeline_config.json"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Aucun fichier de configuration trouvé dans '{out_path}'. "
            "Veuillez exécuter le preprocessing ou la PCA au préalable."
        )

    # Lecture de l'état du pipeline
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    # Chargement conditionnel des fichiers Parquet
    use_pca = config.get("use_pca", False)
    if use_pca:
        train_path = out_path / "train_pca.parquet"
        test_path = out_path / "test_pca.parquet"
    else:
        train_path = out_path / "train_prepared.parquet"
        test_path = out_path / "test_prepared.parquet"

    df_train = pd.read_parquet(train_path) if train_path.exists() else None
    df_test = pd.read_parquet(test_path) if test_path.exists() else None

    return df_train, df_test


def export_train_test_parquet(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    y_test: pd.Series | np.ndarray,
    num_features: list[str] | None = None,
    cat_features: list[str] | None = None,
    transformer: ColumnTransformer | None = None,
    label_enc: LabelEncoder | None = None,
    df_samples: pd.DataFrame | None = None,
    output_dir: Path | str = "data/processed_data",
    target_name: str = "target",
    transform_y: str | None = None,
    drop_cols: list[str] | None = None,
) -> tuple[Path, Path]:
    """Export prepared training and test sets as compressed Parquet files, 
    along with preprocessing artifacts and feature lists.

    Args:
        X_train: Training features DataFrame.
        X_test: Test features DataFrame.
        y_train: Training target labels or continuous values.
        y_test: Test target labels or continuous values.
        num_features: List of numerical column names.
        cat_features: List of categorical column names.
        transformer: Fitted ColumnTransformer instance.
        output_dir: Directory where artifacts and parquet files will be saved.
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

    # Export au format Parquet
    train_path = out_path / "train_prepared.parquet"
    test_path = out_path / "test_prepared.parquet"
    X_tr.to_parquet(train_path, index=False)
    X_te.to_parquet(test_path, index=False)

    # Sauvegarde des échantillons de test par classe (si fournis)
    samples_path = out_path / "df_samples.parquet"
    if df_samples is not None and not df_samples.empty:
        df_samples.to_parquet(samples_path, index=False)

    # Sauvegarde automatique des métadonnées et du transformateur dans le même répertoire
    if num_features is not None:
        with open(out_path / "num_features.json", "w", encoding="utf-8") as f:
            json.dump(num_features, f, ensure_ascii=False, indent=4)

    if cat_features is not None:
        with open(out_path / "cat_features.json", "w", encoding="utf-8") as f:
            json.dump(cat_features, f, ensure_ascii=False, indent=4)


    if transformer is not None:
        joblib.dump(transformer, out_path / "column_transformer.pkl")

    # Sauvegarde du LabelEncoder fitté (avec joblib car c'est un objet scikit-learn)
    if label_enc is not None:
        joblib.dump(label_enc, out_path / "label_encoder.pkl")

    config_state = {
        "use_pca": False
    }
    with open(out_path / "pipeline_config.json", "w", encoding="utf-8") as f:
        json.dump(config_state, f, ensure_ascii=False, indent=4)

    return train_path, test_path


def export_pca_parquet(
    X_train_pca: np.ndarray,
    X_test_pca: np.ndarray,
    y_train: pd.Series | np.ndarray,
    y_test: pd.Series | np.ndarray,
    pca: PCA | None = None,
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
    train_path = out_path / "train_pca.parquet"
    test_path = out_path / "test_prepared.parquet"   
    test_path = out_path / "test_pca.parquet"

    # Conversion des tableaux NumPy PCA en DataFrames pandas avec des noms de colonnes explicites
    cols = [f"PC_{i+1}" for i in range(X_train_pca.shape[1])]
    
    df_train_pca = pd.DataFrame(X_train_pca, columns=cols)
    df_test_pca = pd.DataFrame(X_test_pca, columns=cols)

    # Ajout de la variable cible
    df_train_pca[target_name] = y_train.values if hasattr(y_train, "values") else y_train
    df_test_pca[target_name] = y_test.values if hasattr(y_test, "values") else y_test

    # Export au format Parquet
    df_train_pca.to_parquet(train_path, index=False)
    df_test_pca.to_parquet(test_path, index=False)

    # Sauvegarde de l'objet PCA dans le même répertoire
    if pca is not None:
        joblib.dump(pca, out_path / "pca_model.pkl")

    # Mise à jour intelligente du fichier de configuration (bascule use_pca à True)
    config_path = out_path / "pipeline_config.json"

    config_state = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config_state = json.load(f)

    # On met à jour uniquement les champs relatifs à la PCA
    config_state.update({
        "use_pca": True,
        "n_components": X_train_pca.shape[1]
    })

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_state, f, ensure_ascii=False, indent=4)

    return train_path, test_path