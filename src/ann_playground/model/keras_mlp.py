"""
Module : keras_mlp.py
Description : Multi-Layer Perceptron factory for Keras and interactive Streamlit
              callbacks for live epoch metric tracking.

Classes :
    - StreamlitEpochCallback : Keras callback updating Streamlit Plotly visualizer.

Functions :
    - build_keras_mlp : Instantiates and compiles a sequential Keras classification model.
"""

from typing import Any, List

import keras
import pandas as pd
import plotly.graph_objects as go
from keras import layers, models, optimizers


class StreamlitEpochCallback(keras.callbacks.Callback):
    """Callback to stream training progress directly to Streamlit placeholders."""

    def __init__(
        self, total_epochs: int, chart_placeholder: Any, progress_bar: Any
    ) -> None:
        """Initialize the callback tracker.

        Args:
            total_epochs: Total number of training epochs.
            chart_placeholder: Streamlit placeholder for the Plotly figure.
            progress_bar: Streamlit placeholder for the progress indicator.
        """
        super().__init__()
        self.total_epochs: int = total_epochs
        self.chart_placeholder: Any = chart_placeholder
        self.progress_bar: Any = progress_bar
        self.history_records: List[dict[str, float]] = []

    def on_epoch_end(self, epoch: int, logs: dict[str, float] | None = None) -> None:
        """Update metrics and refresh the plot on each epoch completion."""
        if logs is None:
            return

        self.history_records.append(
            {
                "epoch": epoch + 1,
                "loss": logs.get("loss", 0.0),
                "val_loss": logs.get("val_loss", 0.0),
                "accuracy": logs.get("accuracy", 0.0),
                "val_accuracy": logs.get("val_accuracy", 0.0),
            }
        )

        progress = (epoch + 1) / self.total_epochs
        self.progress_bar.progress(
            progress, text=f"Training: Epoch {epoch + 1}/{self.total_epochs}"
        )

        if (epoch + 1) % 2 == 0 or (epoch + 1) == self.total_epochs:
            df_hist = pd.DataFrame(self.history_records)
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=df_hist["epoch"],
                    y=df_hist["loss"],
                    name="Train Loss",
                    line=dict(color="#439cc8"),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=df_hist["epoch"],
                    y=df_hist["val_loss"],
                    name="Val Loss",
                    line=dict(color="#f08a5d", dash="dash"),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=df_hist["epoch"],
                    y=df_hist["accuracy"],
                    name="Train Accuracy",
                    yaxis="y2",
                    line=dict(color="#2ca02c"),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=df_hist["epoch"],
                    y=df_hist["val_accuracy"],
                    name="Val Accuracy",
                    yaxis="y2",
                    line=dict(color="#d62728", dash="dash"),
                )
            )

            fig.update_layout(
                title="Learning Dynamics (Keras)",
                xaxis=dict(title="Epoch"),
                yaxis=dict(title="Loss", rangemode="tozero"),
                yaxis2=dict(
                    title="Accuracy", overlaying="y", side="right", range=[0.0, 1.0]
                ),
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                ),
                height=400,
                margin=dict(l=20, r=20, t=50, b=20),
            )
            self.chart_placeholder.plotly_chart(fig, use_container_width=True)


def build_keras_mlp(
    input_dim: int,
    num_classes: int,
    hidden_layers: List[int],
    activation: str,
    dropout_rate: float,
    use_batch_norm: bool,
    learning_rate: float,
) -> keras.Model:
    """Build and compile a multi-class MLP with Keras.

    Args:
        input_dim: Number of input features.
        num_classes: Number of target classes.
        hidden_layers: List containing units per hidden layer.
        activation: Non-linear activation function name.
        dropout_rate: Dropout regularization fraction.
        use_batch_norm: Whether to insert batch normalization layers.
        learning_rate: Adam optimizer learning rate.

    Returns:
        Compiled Keras Model instance.
    """
    model = models.Sequential()
    model.add(layers.Input(shape=(input_dim,)))

    for neurons in hidden_layers:
        model.add(layers.Dense(neurons))
        if use_batch_norm:
            model.add(layers.BatchNormalization())
        model.add(layers.Activation(activation))
        if dropout_rate > 0.0:
            model.add(layers.Dropout(dropout_rate))

    model.add(layers.Dense(num_classes, activation="softmax"))

    opt = optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
