from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import callbacks, layers, models

from plant_disease_app.config import ARTIFACTS_DIR, IMAGE_SIZE, RANDOM_STATE, TEST_SIZE


@dataclass
class DeepLearningResult:
    model_name: str
    history: dict[str, list[float]]
    metrics: dict[str, float]
    predictions_df: pd.DataFrame
    model: tf.keras.Model
    notes: list[str]
    class_count: int
    meets_target: bool


def load_dataset_for_dl(feature_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, LabelEncoder, np.ndarray]:
    labeled_df = feature_df[feature_df["label"] != "unknown"].copy()
    if labeled_df.empty:
        raise ValueError("No labeled images are available for Deep Learning.")

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
        raise ValueError("Unable to load images for the Deep Learning models.")

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(kept_labels)
    return np.asarray(images), y, label_encoder, np.asarray(kept_paths)


def _data_augmentation() -> tf.keras.Sequential:
    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.12),
            layers.RandomContrast(0.1),
        ],
        name="augmentation",
    )


def build_scratch_cnn(num_classes: int) -> tf.keras.Model:
    augmentation = _data_augmentation()
    inputs = layers.Input(shape=(IMAGE_SIZE[1], IMAGE_SIZE[0], 3))
    x = augmentation(inputs)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs=inputs, outputs=outputs, name="cnn_scratch")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_pretrained_cnn(num_classes: int) -> tuple[tf.keras.Model, list[str]]:
    notes: list[str] = []
    keras_home = ARTIFACTS_DIR / "keras"
    models_dir = keras_home / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    os.environ["KERAS_HOME"] = str(keras_home)
    model_filename = "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_128_no_top.h5"
    local_weights_path = models_dir / model_filename

    try:
        if not local_weights_path.exists():
            downloaded = tf.keras.utils.get_file(
                fname=model_filename,
                origin=f"https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/{model_filename}",
                cache_dir=str(keras_home),
                cache_subdir="models",
            )
            local_weights_path = Path(downloaded)
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(IMAGE_SIZE[1], IMAGE_SIZE[0], 3),
            include_top=False,
            weights=str(local_weights_path),
        )
        notes.append("MobileNetV2 backbone initialized with ImageNet weights.")
    except Exception as exc:
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(IMAGE_SIZE[1], IMAGE_SIZE[0], 3),
            include_top=False,
            weights=None,
        )
        notes.append(f"ImageNet weights unavailable, using random MobileNetV2 initialization: {exc}")

    base_model.trainable = False
    augmentation = _data_augmentation()
    preprocess = tf.keras.applications.mobilenet_v2.preprocess_input

    inputs = layers.Input(shape=(IMAGE_SIZE[1], IMAGE_SIZE[0], 3))
    x = augmentation(inputs)
    x = layers.Lambda(lambda tensor: preprocess(tensor * 255.0))(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs=inputs, outputs=outputs, name="cnn_pretrained")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, notes


def _train_single_model(
    model_name: str,
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    paths_test: np.ndarray,
    label_encoder: LabelEncoder,
    notes: list[str] | None = None,
    epochs: int = 8,
    batch_size: int = 16,
    target_accuracy: float = 0.7,
) -> DeepLearningResult:
    validation_split = 0.2 if len(X_train) > 8 else 0.0
    fit_callbacks: list[tf.keras.callbacks.Callback] = [
        callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.5, min_lr=1e-5),
    ]
    if validation_split == 0.0:
        fit_callbacks = []

    history = model.fit(
        X_train,
        y_train,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=fit_callbacks,
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
    merged_notes = list(notes or [])
    merged_notes.append(f"Evaluated on {len(np.unique(y_test))} classes and {len(y_test)} validation images.")
    return DeepLearningResult(
        model_name=model_name,
        history={key: [float(v) for v in values] for key, values in history.history.items()},
        metrics=metrics,
        predictions_df=predictions_df,
        model=model,
        notes=merged_notes,
        class_count=int(len(label_encoder.classes_)),
        meets_target=metrics["accuracy"] >= target_accuracy,
    )


def train_deep_learning_models(
    feature_df: pd.DataFrame,
    epochs_scratch: int = 8,
    epochs_pretrained: int = 8,
    batch_size: int = 16,
    target_accuracy: float = 0.7,
) -> list[DeepLearningResult]:
    X, y, label_encoder, paths = load_dataset_for_dl(feature_df)
    class_count = len(np.unique(y))
    if class_count < 3:
        raise ValueError("At least three labeled classes are required for the Deep Learning comparison.")

    X_train, X_test, y_train, y_test, _, paths_test = train_test_split(
        X,
        y,
        paths,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    scratch_model = build_scratch_cnn(num_classes=len(label_encoder.classes_))
    scratch_result = _train_single_model(
        model_name="CNN from scratch",
        model=scratch_model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        paths_test=paths_test,
        label_encoder=label_encoder,
        notes=["Custom compact CNN trained from random initialization."],
        epochs=epochs_scratch,
        batch_size=batch_size,
        target_accuracy=target_accuracy,
    )

    pretrained_model, pretrained_notes = build_pretrained_cnn(num_classes=len(label_encoder.classes_))
    pretrained_result = _train_single_model(
        model_name="CNN predefined (MobileNetV2)",
        model=pretrained_model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        paths_test=paths_test,
        label_encoder=label_encoder,
        notes=pretrained_notes,
        epochs=epochs_pretrained,
        batch_size=batch_size,
        target_accuracy=target_accuracy,
    )

    base_model = pretrained_model.get_layer(index=3)
    if isinstance(base_model, tf.keras.Model):
        base_model.trainable = True
        for layer in base_model.layers[:-30]:
            layer.trainable = False
        pretrained_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        pretrained_result = _train_single_model(
            model_name="CNN predefined (MobileNetV2)",
            model=pretrained_model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            paths_test=paths_test,
            label_encoder=label_encoder,
            notes=pretrained_notes + ["Fine-tuned the last MobileNetV2 blocks after the frozen warm-up stage."],
            epochs=max(4, epochs_pretrained // 2),
            batch_size=batch_size,
            target_accuracy=target_accuracy,
        )

    return [scratch_result, pretrained_result]
