from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_precision_recall_curves(pr_curves: dict[str, dict[str, list[float]]], title: str):
    """Construit une figure matplotlib pour les courbes précision/rappel."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for class_name, curve_data in pr_curves.items():
        ax.plot(curve_data["recall"], curve_data["precision"], label=class_name)
    ax.set_title(title)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_training_history(history: dict[str, list[float]], title: str):
    """Affiche les courbes d'apprentissage Keras."""
    fig, ax = plt.subplots(figsize=(6, 4))
    if "accuracy" in history:
        ax.plot(history["accuracy"], label="train_accuracy")
    if "val_accuracy" in history:
        ax.plot(history["val_accuracy"], label="val_accuracy")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    return fig


def split_correct_incorrect(predictions_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sépare les exemples correctement et incorrectement prédits."""
    if predictions_df.empty:
        return predictions_df, predictions_df
    correct_mask = predictions_df["true_label"] == predictions_df["predicted_label"]
    return predictions_df[correct_mask], predictions_df[~correct_mask]


def image_exists(path: str) -> bool:
    """Vérifie qu'un chemin d'image est exploitable depuis l'interface."""
    return Path(path).exists()
