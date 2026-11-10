import mlflow
import mlflow.pyfunc
import pickle
from pathlib import Path
from typing import Any
from mlflow.pyfunc import PythonModelContext

class NumpyMLPWrapper(mlflow.pyfunc.PythonModel):
    """Wrapper MLflow pour le modèle NumpyMLP personnalisé."""

    def load_context(self, context: PythonModelContext) -> None:
        """Charge le modèle sérialisé depuis les artefacts du run."""
        model_path = context.artifacts["numpy_model_file"]
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

    def predict(
        self, context: PythonModelContext, model_input: Any
      ) -> Any:
        """Effectue une prédiction standardisée via l'API pyfunc de MLflow.
        
        Args:
            context: Contexte d'exécution MLflow.
            model_input: Tableau numpy ou DataFrame en entrée.
        """
        # Conversion en numpy array si l'entrée est un DataFrame Pandas
        if hasattr(model_input, "to_numpy"):
            x = model_input.to_numpy()
        else:
            x = model_input
            
        return self.model.predict(x)