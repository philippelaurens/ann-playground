import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class LogisticRegression(BaseEstimator, ClassifierMixin):

    def __init__(self, lr: float = 0.01, epochs: int = 500) -> None:
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.b = None
        self.loss_history = []
        self.classes_ = []

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Fonction d'activation Sigmoïde."""
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def compute_bce_loss(self, y_true: np.ndarray, y_pred_prob: np.ndarray) -> float:
        """Calcul de la Binary Cross-Entropy Loss avec clipping anti-log(0)."""
        eps = 1e-15
        y_pred_prob = np.clip(y_pred_prob, eps, 1.0 - eps)
        loss = -np.mean(y_true * np.log(y_pred_prob) + (1 - y_true) * np.log(1 - y_pred_prob))
        return float(loss)

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        m, n = X.shape

        self.w = np.zeros(n, dtype=float)
        self.b = 0.0
        self.loss_history = []

        for epoch in range(self.epochs):
            # 1. Propagation avant (Forward pass)
            z = np.dot(X, self.w) + self.b
            y_hat = self.sigmoid(z)

            # 2. Calcul de la perte BCE
            loss = self.compute_bce_loss(y, y_hat)
            self.loss_history.append(loss)

            # 3. Calcul des gradients
            dw = (1.0 / m) * np.dot(X.T, (y_hat - y))
            db = (1.0 / m) * np.sum(y_hat - y)

            # 4. Mise à jour des poids (Descente de gradient)
            self.w -= self.lr * dw
            self.b -= self.lr * db

            self.classes_ = np.unique(y)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Renvoie la probabilité d'appartenir à la classe 1."""
        if self.w is None or self.b is None:
            raise ValueError("Il faut entraîner le modèle avec .fit() avant de prédire.")
        X = np.asarray(X, dtype=float)
        z = np.dot(X, self.w) + self.b
        return self.sigmoid(z)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Renvoie la classe binaire (0 ou 1) selon un seuil."""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)