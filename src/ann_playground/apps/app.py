import hmac
import logging
from pathlib import Path

import streamlit as st
from ann_playground.config import DATA,  PROCESSED_DATA_DIR

logger = logging.getLogger(__name__)


REQUIRED_ARTIFACTS = [
    "train_prepared.parquet",
    "test_prepared.parquet",
]

st.set_page_config(
    page_title="Home",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Authentification
# ---------------------------------------------------------------------------

def is_authenticated() -> bool:
    return st.session_state["authenticated"]

def get_expected_credentials() -> tuple[str, str] | None:
    """Lit les identifiants attendus dans st.secrets (jamais en dur)."""
    try:
        user, password = st.secrets["AUTH_USER"], st.secrets["AUTH_PASSWORD"]
        logger.warning("Secrets trouvés : user=%r, password_len=%d", user, len(password))
        return user, password
    except (KeyError, FileNotFoundError) as exc:
        logger.warning("Secrets introuvables : %r", exc)
        return None

def credentials_are_valid(username: str, password: str) -> bool:
    expected = get_expected_credentials()
    if expected is None:
        return False
    expected_user, expected_pass = expected
    return hmac.compare_digest(username, expected_user) and hmac.compare_digest(
        password, expected_pass
    )


def login() -> None:
    """Callback du bouton de connexion."""
    ok = credentials_are_valid(
        st.session_state.get("username", ""),
        st.session_state.get("password", ""),
    )
    st.session_state["authenticated"] = ok
    st.session_state["login_error"] = not ok


def logout() -> None:
    """Callback du bouton de déconnexion."""
    st.session_state["authenticated"] = False
    st.session_state["login_error"] = False


def require_authentication() -> None:
    """Affiche le formulaire de connexion et bloque la page si nécessaire."""
    
    st.session_state.setdefault("authenticated", False)
    st.session_state.setdefault("login_error", False)
    
    if not is_authenticated():
        _, center, _ = st.columns([1, 2, 1])
        with center:
            st.title("🔒 Accès restreint - ANN Playground")
            st.text_input("Nom d'utilisateur", key="username")
            st.text_input("Mot de passe", type="password", key="password")
            st.button("Se connecter", on_click=login)

            if st.session_state["login_error"]:
                st.error("Identifiants incorrects.")
        st.stop()


# ---------------------------------------------------------------------------
# Home
# ---------------------------------------------------------------------------

def render_home() -> None:

    st.title("ANN Playground : Deep Learning & Multi-Layer Perceptron")
    st.divider()

    col_raw, col_artifacts = st.columns(2)
    
    with col_raw:
        st.markdown("### Source brute")
        if DATA.exists():
            st.success(f"Données brutes détectées : `{DATA.resolve()}`")
        else:
            st.error(f"Fichier brut manquant : `{DATA.resolve()}`")
            
    with col_artifacts:
        st.markdown("### Données préparées (Artefacts)")
        missing = [ name for name in REQUIRED_ARTIFACTS if not (PROCESSED_DATA_DIR / name).exists() ]

        if not missing:
            st.success(f"Artefacts d'entraînement disponibles dans `{PROCESSED_DATA_DIR}`")
        else:
            missing_list = ", ".join(f"`{name}`" for name in missing)
            st.warning(
                f"Artefacts manquants : {missing_list}. Veuillez exécuter le module "
                "de preprocessing pour générer les données d'apprentissage."
            )
        
    st.button("Se déconnecter", on_click=logout)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

# Authentication
require_authentication()

# Home
render_home()