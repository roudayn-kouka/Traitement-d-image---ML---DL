from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.svm import SVC

from plant_disease_app.config import RANDOM_STATE, TEST_SIZE


@dataclass
class ClassicalModelResult:
    model_name: str
    model: Pipeline
    metrics: dict
    predictions_df: pd.DataFrame
    pr_curves: dict[str, dict[str, list[float]]]


def prepare_feature_matrix(feature_df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, LabelEncoder]:
    """Sépare les colonnes explicatives et encode les labels."""
    usable_df = feature_df[feature_df["label"] != "unknown"].copy()
    if usable_df.empty:
        raise ValueError("Aucune image étiquetée n'est disponible pour l'entraînement ML.")

    X = usable_df.drop(columns=["label", "image_path"], errors="ignore")
    y_raw = usable_df["label"].astype(str).to_numpy()
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    return X, y, label_encoder


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, target_names: list[str]) -> dict:
    """Calcule les métriques de classification standards."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=target_names,
            zero_division=0,
            output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def compute_pr_curves(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    class_names: list[str],
) -> dict[str, dict[str, list[float]]]:
    """Construit les courbes précision/rappel one-vs-rest pour chaque classe."""
    if not hasattr(model[-1], "predict_proba"):
        return {}

    y_score = model.predict_proba(X_test)
    if len(class_names) < 2:
        return {}

    if len(class_names) == 2:
        positive_class = class_names[1]
        precision, recall, _ = precision_recall_curve((y_test == 1).astype(int), y_score[:, 1])
        return {
            positive_class: {
                "precision": precision.tolist(),
                "recall": recall.tolist(),
            }
        }

    y_bin = label_binarize(y_test, classes=list(range(len(class_names))))
    curves: dict[str, dict[str, list[float]]] = {}
    for idx, class_name in enumerate(class_names):
        if y_bin[:, idx].sum() == 0:
            continue
        precision, recall, _ = precision_recall_curve(y_bin[:, idx], y_score[:, idx])
        curves[class_name] = {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
        }
    return curves


def train_classical_models(feature_df: pd.DataFrame) -> list[ClassicalModelResult]:
    """Entraîne SVM et RandomForest puis retourne leurs résultats détaillés."""
    X, y, label_encoder = prepare_feature_matrix(feature_df)
    if len(np.unique(y)) < 2:
        raise ValueError("Au moins deux classes étiquetées sont nécessaires pour l'entraînement ML.")
    stratify = y if len(np.unique(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )
    class_names = label_encoder.classes_.tolist()

    models: dict[str, Pipeline] = {
        "SVM": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)),
            ]
        ),
        "RandomForest": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)),
            ]
        ),
    }

    results: list[ClassicalModelResult] = []
    for model_name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        metrics = compute_metrics(y_test, y_pred, class_names)
        pr_curves = compute_pr_curves(pipeline, X_test, y_test, class_names)
        prediction_table = pd.DataFrame(
            {
                "image_path": feature_df.loc[X_test.index, "image_path"].values,
                "true_label": label_encoder.inverse_transform(y_test),
                "predicted_label": label_encoder.inverse_transform(y_pred),
            }
        )
        results.append(
            ClassicalModelResult(
                model_name=model_name,
                model=pipeline,
                metrics=metrics,
                predictions_df=prediction_table,
                pr_curves=pr_curves,
            )
        )
    return results
