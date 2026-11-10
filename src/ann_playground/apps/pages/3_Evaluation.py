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
    page_title="ANN Playground : Evaluation",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# Barre latérale : Source de données & Target
# ==============================================================================

st.title("ANN Playground : Evaluation")

with st.sidebar:
    with st.empty().container():
        st.caption("Module générique d'évaluation")


# ==============================================================================
# Structure principale en onglets
# ==============================================================================




# ------------------------------------------------------------------------------
# 1 · Diagnostic structurel
# ------------------------------------------------------------------------------



# ------------------------------------------------------------------------------
# 2 · Distributions & Relations
# ------------------------------------------------------------------------------



# ------------------------------------------------------------------------------
# 3 · Colinéarité & Sélection
# ------------------------------------------------------------------------------
    
