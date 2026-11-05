import streamlit as st

from ann_playground.config import DATA, CLEAN_DATA_DIR, PROCESSED_DATA_DIR

st.set_page_config(
    page_title="ANN Playground · Home",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("ANN Playground : Deep Learning & Multi-Layer Perceptron")

def main():
    st.divider()

    col_status_1, col_status_2 = st.columns(2)

    with col_status_1:
        st.markdown("### Source brute")
        if DATA.exists():
            st.success(f"Données brutes détectées : `{DATA.resolve()}`")
        else:
            st.error(f"Fichier brut manquant : `{DATA.resolve()}`")

    with col_status_2:

        st.markdown("### Données préparées (Artefacts)")
        
        # Vérification des fichiers issus de la préparation et de l'ACP (PCA)
        x_train_pca = CLEAN_DATA_DIR / "X_train_pca.pkl"
        x_test_pca = CLEAN_DATA_DIR / "X_test_pca.pkl"
        y_train = CLEAN_DATA_DIR / "y_train.pkl"
        y_test = CLEAN_DATA_DIR / "y_test.pkl"
        df_parquet = CLEAN_DATA_DIR / "df.parquet"

        # On s'assure que les matrices principales pour l'entraînement et l'évaluation sont présentes
        required_artifacts = [x_train_pca, x_test_pca, y_train, y_test]
        
        if all(art.exists() for art in required_artifacts):
            st.success(f"Artefacts d'entraînement (PCA) disponibles dans `{CLEAN_DATA_DIR}`")
            # Optionnel : afficher un petit détail des fichiers trouvés
            st.caption("✅ `X_train_pca.pkl`, `X_test_pca.pkl`, `y_train.pkl`, `y_test.pkl` détectés.")
        else:
            st.warning(
                "Certains artefacts préparés sont manquants. Veuillez exécuter le module "
                "`1_data_studio` pour générer les matrices d'apprentissage."
            )

if __name__ == "__main__":
    main()