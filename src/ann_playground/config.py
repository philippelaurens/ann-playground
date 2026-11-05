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