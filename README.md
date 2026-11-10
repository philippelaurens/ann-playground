# Machine Learning Application: Part One (Preprocessing & Training)

This repository contains the preprocessing, training, and tracking infrastructure for the custom NumPy Multi-Layer Perceptron (`NumpyMLP`) model, integrated with MLflow and Streamlit.

---

## 1. Authentication & Configuration Setup

To ensure secure tracking, storage, and deployment, configure the following files in your environment.

### A. Environment File: `.env`
Used by `docker-compose.yml` to define secure secrets and server routing.

```env
MLFLOW_FLASK_SERVER_SECRET_KEY=ee0dd1023b0061b2789e666bb25c883a2cea4670a1d865e6e9951c1456bb3b49
MLFLOW_AUTH_ADMIN_USERNAME=xxxxx
MLFLOW_AUTH_ADMIN_PASSWORD=xxxxx

EXTERNAL_HOST=xxxxx.xxxxxxx.xxx:5000
```

### B. Basic Auth Configuration: `mlflow-data/basic_auth.yml`
Configures authentication permissions and SQLite tracking database location for the MLflow server.

```yaml
[mlflow]
default_permission = READ
database_uri = sqlite:////app/mlflow-data/mlflow.db
admin_username = xxxxx
admin_password = xxxxx
```

### C. Streamlit Secrets: `.streamlit/secrets.toml`
Stores credentials for Streamlit application access to MLflow or protected UI views.

```toml
AUTH_USER = "xxxxx"
AUTH_PASSWORD = "xxxxx"
```

---

## 2. MLflow Architecture & Storage Scheme

The client-server MLflow setup relies on two persistent volume mounts:

```
.
├── mlflow-data/          # Mounted to /app/mlflow-data (SQLite database + basic_auth.yml)
├── mlartifacts/          # Mounted to /app/mlartifacts (Serialized models, feature maps, plots)
```

* **`mlflow-data/`**: Stores metadata, experiments, runs, parameters, metrics, and user access records.
* **`mlartifacts/`**: Stores physical binaries (`.pkl`, `.json`, `.html`, PyFunc wrapped models).

---

## 3. MLflow Pyfunc Wrapper for Inference: Benefits & Usage

### Overview
While tracking metrics and saving raw `.pkl` files covers basic experiment logging, wrapping custom NumPy models (`NumpyMLP`) into MLflow's standard `pyfunc` interface (`PythonModel`) unlocks production-grade MLOps capabilities during inference.

### Key Benefits at Inference Time

#### 1. Unified & Agnostic Prediction API
* **Without Wrapper:** Your inference script must import original source classes (`NumpyMLP`), instantiate objects manually, and call model-specific methods (e.g., `model.forward(X)`).
* **With Wrapper:** Provides a standardized interface regardless of the underlying framework (NumPy, PyTorch, Keras, or Scikit-Learn):

```python
import mlflow

# Standardized loading across all model types
model = mlflow.pyfunc.load_model(f"runs:/{run_id}/numpy_model")

# Uniform prediction interface
predictions = model.predict(X_new)
```

#### 2. Absolute Traceability & Dynamic Run Selection
* **Without Wrapper:** Relying on hardcoded local file paths (e.g., `data/processed_data/numpy_mlp_model.pkl`) creates brittle pipelines.
* **With Wrapper:** Models are loaded directly using unique MLflow identifiers (`runs:/<run_id>/numpy_model`). Streamlit can dynamically pull and execute any historical model run on demand.

#### 3. Environment & Dependency Consistency
When logging via `mlflow.pyfunc.log_model`, MLflow packages execution details and dependencies, guaranteeing that reloading the model across environments executes under identical runtime conditions.

#### 4. Zero-Code Production Deployment (Model Serving)
MLflow can spin up a production-ready REST API endpoint natively via CLI without writing FastAPI/Flask wrapper code:

```bash
mlflow models serve -m "runs:/<run_id>/numpy_model" --port 5001
```

---

## 4. Summary Comparison

| Feature | Direct `.pkl` Load | MLflow `pyfunc` Wrapper |
| :--- | :--- | :--- |
| **Local Script Inference** | ✅ Simple & Fast | ✅ Supported |
| **Framework Agnostic API** | ❌ No | ✅ Yes (`model.predict()`) |
| **Dynamic Run Retrieval** | ❌ Hardcoded paths | ✅ Built-in via Run ID |
| **Automated Model Serving** | ❌ Manual API code | ✅ Out-of-the-box CLI |
| **Production Ready** | ❌ Low | ✅ High |

---

## 5. Wrapper Implementation Template

```python
import pickle
import mlflow
import mlflow.pyfunc
from mlflow.pyfunc import PythonModelContext

class CustomModelWrapper(mlflow.pyfunc.PythonModel):
    """Generic MLflow wrapper for custom models."""

    def load_context(self, context: PythonModelContext) -> None:
        """Load the underlying model object from logged artifacts."""
        with open(context.artifacts["model_file"], "rb") as f:
            self.model = pickle.load(f)

    def predict(self, context: PythonModelContext, model_input: object) -> object:
        """Standardized prediction call."""
        if hasattr(model_input, "to_numpy"):
            model_input = model_input.to_numpy()
        return self.model.predict(model_input)
```