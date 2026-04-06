from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models

from plant_disease_app.config import IMAGE_SIZE, RANDOM_STATE, TEST_SIZE


@dataclass
class DeepLearningResult:
    history: dict[str, list[float]]
    metrics: dict[str, float]
    predictions_df: pd.DataFrame
    model: tf.keras.Model


def load_dataset_for_dl(feature_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, LabelEncoder, np.ndarray]:
    """Charge les pixels RGB et les labels à partir du dataframe de features."""
    labeled_df = feature_df[feature_df["label"] != "unknown"].copy()
    if labeled_df.empty:
        raise ValueError("Aucune image étiquetée n'est disponible pour le Deep Learning.")

    images: list[np.ndarray] = []
    kept_labels: list[str] = []
    kept_paths: list[str] = []
    for _, row in labeled_df.iterrows():
        bgr = cv2.imread(str(row["image_path"]))
        if bgr is None:
            continue
        resized = cv2.resize(bgr, IMAGE_SIZE)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
        images.append(rgb)
        kept_labels.append(str(row["label"]))
        kept_paths.append(str(row["image_path"]))

    if not images:
        raise ValueError("Impossible de charger les images pour le modèle Deep Learning.")

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(kept_labels)
    X = np.asarray(images)
    paths = np.asarray(kept_paths)
    return X, y, label_encoder, paths


def build_cnn_model(num_classes: int) -> tf.keras.Model:
    """Construit un CNN compact, simple à exécuter localement."""
    model = models.Sequential(
        [
            layers.Input(shape=(IMAGE_SIZE[1], IMAGE_SIZE[0], 3)),
            layers.Conv2D(32, 3, activation="relu", padding="same"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation="relu", padding="same"),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 3, activation="relu", padding="same"),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dropout(0.3),
            layers.Dense(128, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_deep_learning_model(feature_df: pd.DataFrame, epochs: int = 5, batch_size: int = 16) -> DeepLearningResult:
    """Entraîne un modèle de transfert léger et retourne ses performances."""
    X, y, label_encoder, paths = load_dataset_for_dl(feature_df)
    if len(np.unique(y)) < 2:
        raise ValueError("Au moins deux classes étiquetées sont nécessaires pour le Deep Learning.")
    stratify = y if len(np.unique(y)) > 1 else None
    X_train, X_test, y_train, y_test, _, paths_test = train_test_split(
        X,
        y,
        paths,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )

    model = build_cnn_model(num_classes=len(label_encoder.classes_))
    validation_split = 0.2 if len(X_train) > 4 else 0.0
    callbacks = []
    if validation_split > 0:
        callbacks.append(tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True))
    history = model.fit(
        X_train,
        y_train,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=callbacks,
    )

    y_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    predictions_df = pd.DataFrame(
        {
            "image_path": paths_test,
            "true_label": label_encoder.inverse_transform(y_test),
            "predicted_label": label_encoder.inverse_transform(y_pred),
            "confidence": np.max(y_prob, axis=1),
        }
    )
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
    }
    return DeepLearningResult(
        history={key: [float(v) for v in values] for key, values in history.history.items()},
        metrics=metrics,
        predictions_df=predictions_df,
        model=model,
    )
