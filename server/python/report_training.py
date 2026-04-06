from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cache_utils import dataset_signature, load_cached_report, load_or_create_dataframe, save_cached_report  # noqa: E402
from plant_disease_app.config import TEST_SIZE  # noqa: E402
from plant_disease_app.image_pipeline import dataset_to_dataframe, resolve_dataset_split_root  # noqa: E402
from plant_disease_app.ml_pipeline import train_classical_models  # noqa: E402

REPORT_CACHE_VERSION = "v2"


def dataset_summary(feature_df):
    label_counts = (
        feature_df["label"].value_counts().sort_index().to_dict() if not feature_df.empty and "label" in feature_df else {}
    )
    return {
        "sampleCount": int(len(feature_df)),
        "featureCount": int(len(feature_df.columns) - 2) if not feature_df.empty else 0,
        "labels": {str(key): int(value) for key, value in label_counts.items()},
        "trainTestSplit": {
            "trainRatio": round(float(1 - TEST_SIZE), 2),
            "testRatio": round(float(TEST_SIZE), 2),
        },
    }


def evaluate_with_val_split(train_df, val_df):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.svm import SVC

    usable_train = train_df[train_df["label"] != "unknown"].copy()
    usable_val = val_df[val_df["label"] != "unknown"].copy()
    if usable_train.empty or usable_val.empty:
        return []

    shared_labels = sorted(set(usable_train["label"]).intersection(set(usable_val["label"])))
    if len(shared_labels) < 2:
        return []

    usable_train = usable_train[usable_train["label"].isin(shared_labels)].copy()
    usable_val = usable_val[usable_val["label"].isin(shared_labels)].copy()

    X_train = usable_train.drop(columns=["label", "image_path"], errors="ignore")
    X_val = usable_val.drop(columns=["label", "image_path"], errors="ignore")
    y_train_raw = usable_train["label"].astype(str).to_numpy()
    y_val_raw = usable_val["label"].astype(str).to_numpy()

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_raw)
    y_val = label_encoder.transform(y_val_raw)

    models = {
        "SVM": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", SVC(kernel="rbf", probability=True, random_state=42)),
            ]
        ),
        "RandomForest": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", RandomForestClassifier(n_estimators=200, random_state=42)),
            ]
        ),
    }

    serialized = []
    for model_name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)
        serialized.append(
            {
                "modelName": model_name,
                "metrics": {
                    "accuracy": round(float(accuracy_score(y_val, y_pred)), 4),
                    "precision": round(float(precision_score(y_val, y_pred, average="weighted", zero_division=0)), 4),
                    "recall": round(float(recall_score(y_val, y_pred, average="weighted", zero_division=0)), 4),
                    "f1Score": round(float(f1_score(y_val, y_pred, average="weighted", zero_division=0)), 4),
                },
                "predictionExamples": [],
            }
        )
    return serialized


def serialize_classical_results(results):
    serialized = []
    for result in results:
        serialized.append(
            {
                "modelName": result.model_name,
                "metrics": {
                    "accuracy": round(float(result.metrics["accuracy"]), 4),
                    "precision": round(float(result.metrics["precision"]), 4),
                    "recall": round(float(result.metrics["recall"]), 4),
                    "f1Score": round(float(result.metrics["f1_score"]), 4),
                },
                "predictionExamples": result.predictions_df.head(5).to_dict(orient="records"),
            }
        )
    return serialized


def build_deep_learning_section(feature_df):
    try:
        from plant_disease_app.dl_pipeline import train_deep_learning_model
    except Exception as exc:
        return {
            "available": False,
            "message": f"Deep Learning unavailable in this environment: {exc}",
        }

    try:
        deep_result = train_deep_learning_model(feature_df, epochs=3, batch_size=8)
    except Exception as exc:
        return {
            "available": False,
            "message": str(exc),
        }

    return {
        "available": True,
        "modelName": "CNN",
        "metrics": {
            "accuracy": round(float(deep_result.metrics["accuracy"]), 4),
            "precision": round(float(deep_result.metrics["precision"]), 4),
            "recall": round(float(deep_result.metrics["recall"]), 4),
            "f1Score": round(float(deep_result.metrics["f1_score"]), 4),
        },
        "history": {key: [round(float(v), 4) for v in values] for key, values in deep_result.history.items()},
        "predictionExamples": deep_result.predictions_df.head(5).to_dict(orient="records"),
    }


def interpret_results(classical_results, deep_learning_result):
    comparisons = []

    if classical_results:
        best_classical = max(classical_results, key=lambda item: item["metrics"]["accuracy"])
        comparisons.append(
            f"Best classical model: {best_classical['modelName']} with accuracy {best_classical['metrics']['accuracy']}."
        )

    if deep_learning_result.get("available") and classical_results:
        best_classical = max(classical_results, key=lambda item: item["metrics"]["accuracy"])
        deep_accuracy = deep_learning_result["metrics"]["accuracy"]
        delta = round(deep_accuracy - best_classical["metrics"]["accuracy"], 4)
        comparisons.append(f"CNN accuracy delta versus best classical model: {delta}.")
    elif not deep_learning_result.get("available"):
        comparisons.append(deep_learning_result["message"])

    comparisons.append("Accuracy measures global correctness, precision penalizes false positives, and recall penalizes misses.")
    return comparisons


def build_report(dataset_root: Path) -> dict:
    train_root = resolve_dataset_split_root(dataset_root, split="train")
    val_root = resolve_dataset_split_root(dataset_root, split="val")
    signature = dataset_signature(train_root, val_root)
    cache_key = f"training_report_{REPORT_CACHE_VERSION}_{signature}"
    cached_report = load_cached_report(cache_key)
    if cached_report is not None:
        return cached_report

    feature_df = load_or_create_dataframe(
        f"train_features_{signature}",
        lambda: dataset_to_dataframe(dataset_root),
        {"dataset_root": str(train_root), "split": "train"},
    )
    val_df = None
    if val_root != train_root and val_root.exists():
        val_df = load_or_create_dataframe(
            f"val_features_{signature}",
            lambda: dataset_to_dataframe(val_root),
            {"dataset_root": str(val_root), "split": "val"},
        )

    if feature_df.empty:
        report = {
            "dataset": dataset_summary(feature_df),
            "message": "Dataset is empty. Add labeled class folders under data/dataset before training.",
            "classicalModels": [],
            "deepLearning": {
                "available": False,
                "message": "Dataset is empty.",
            },
            "interpretation": [],
        }
        save_cached_report(cache_key, report)
        return report

    if feature_df["label"].nunique() < 2:
        report = {
            "dataset": dataset_summary(feature_df),
            "message": "At least two labeled classes are required for ML and CNN training.",
            "classicalModels": [],
            "deepLearning": {
                "available": False,
                "message": "Not enough classes for CNN training.",
            },
            "interpretation": [],
        }
        save_cached_report(cache_key, report)
        return report

    if val_df is not None and not val_df.empty:
        classical_results = evaluate_with_val_split(feature_df, val_df)
    else:
        classical_results = serialize_classical_results(train_classical_models(feature_df))
    deep_learning_result = build_deep_learning_section(feature_df)

    report = {
        "dataset": dataset_summary(feature_df),
        "message": "Training and evaluation completed.",
        "classicalModels": classical_results,
        "deepLearning": deep_learning_result,
        "interpretation": interpret_results(classical_results, deep_learning_result),
    }
    save_cached_report(cache_key, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    args = parser.parse_args()

    try:
        payload = build_report(Path(args.dataset_root))
    except Exception as exc:
        print(json.dumps({"error": str(exc)}))
        return 1

    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
