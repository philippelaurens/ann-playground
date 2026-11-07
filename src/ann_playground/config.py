import os
from pathlib import Path
from typing import Final
from pyprojroot import here

# La racine est détectée automatiquement grâce au pyproject.toml ou .git à la racine
BASE_DIR: Final[Path] = here()

DATA_DIR: Final[Path] = BASE_DIR / "data"
MODELS_DIR: Final[Path] = BASE_DIR / "models"
LOGS_DIR: Final[Path] = MODELS_DIR / "logs"
SRC_DIR: Final[Path] = BASE_DIR / "src"
PACKAGE_DIR: Final[Path] = BASE_DIR / "src" / "ann_playground"
RAW_DATA_DIR: Final[Path] = DATA_DIR / "raw_data"
CLEAN_DATA_DIR: Final[Path] = DATA_DIR / "clean_data"
PROCESSED_DATA_DIR: Final[Path] = DATA_DIR / "processed_data"

DATA: Final[Path] = RAW_DATA_DIR / "data.csv"

MLFLOW_DATA_DIR: Final[Path] = BASE_DIR / "mlflow-data"
MLFLOW_DB_PATH: Final[Path] = MLFLOW_DATA_DIR / "mlflow.db"
MLFLOW_TRACKING_URI: Final[str] = os.getenv(
    "MLFLOW_TRACKING_URI",
    # f"sqlite:///{MLFLOW_DB_PATH.resolve()}",
    "http://mlflow:5000",
)
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME", "")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD", "")
EXPERIMENT_NAME: str = "ANN_Playground"