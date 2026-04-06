from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cache_utils import dataset_signature, load_or_create_dataframe, train_or_load_random_forest  # noqa: E402
from plant_disease_app.image_pipeline import (  # noqa: E402
    dataset_to_dataframe,
    extract_features_from_rgb,
    load_image,
    preprocess_image,
    segment_leaf,
)


def build_feature_vector(feature_dict: dict[str, float]) -> tuple[list[str], list[float]]:
    keys = sorted(feature_dict.keys())
    return keys, [float(feature_dict[key]) for key in keys]


def compute_segmentation_stats(segmentation: dict[str, np.ndarray], rgb: np.ndarray) -> dict[str, float]:
    mask = segmentation["mask"] > 0
    total_pixels = int(mask.size)
    leaf_pixels = int(mask.sum())

    if leaf_pixels == 0:
      leaf_pixels = total_pixels
      mask = np.ones_like(mask, dtype=bool)

    hsv = segmentation.get("hsv")
    if hsv is None:
        import cv2

        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    selected_hsv = hsv[mask]

    lesion_mask = (
        ((selected_hsv[:, 0] <= 25) | (selected_hsv[:, 0] >= 160))
        & (selected_hsv[:, 1] >= 60)
        & (selected_hsv[:, 2] <= 200)
    )
    yellow_mask = (
        (selected_hsv[:, 0] >= 18)
        & (selected_hsv[:, 0] <= 40)
        & (selected_hsv[:, 1] >= 50)
        & (selected_hsv[:, 2] >= 80)
    )
    green_mask = (
        (selected_hsv[:, 0] >= 35)
        & (selected_hsv[:, 0] <= 95)
        & (selected_hsv[:, 1] >= 40)
        & (selected_hsv[:, 2] >= 40)
    )

    return {
        "leaf_coverage": leaf_pixels / max(total_pixels, 1),
        "lesion_ratio": float(lesion_mask.mean()) if selected_hsv.size else 0.0,
        "yellow_ratio": float(yellow_mask.mean()) if selected_hsv.size else 0.0,
        "green_ratio": float(green_mask.mean()) if selected_hsv.size else 0.0,
    }


def heuristic_prediction(features: dict[str, float], stats: dict[str, float]) -> dict:
    lesion_ratio = stats["lesion_ratio"]
    yellow_ratio = stats["yellow_ratio"]
    green_ratio = stats["green_ratio"]
    circularity = float(features.get("shape_circularity", 0.0))

    if lesion_ratio >= 0.22:
        return {
            "predicted_label": "blight",
            "confidence": min(0.92, 0.62 + lesion_ratio),
            "health_status": "critical",
            "notes": [
                "Strong lesion coverage detected from segmented leaf pixels.",
                "This result comes from image-derived heuristics because no labeled dataset was available.",
            ],
            "mode": "heuristic",
        }

    if yellow_ratio >= 0.18 or (green_ratio < 0.45 and lesion_ratio >= 0.1):
        return {
            "predicted_label": "rust",
            "confidence": min(0.88, 0.58 + yellow_ratio + lesion_ratio / 2),
            "health_status": "warning",
            "notes": [
                "Yellow or brown stress areas were detected on the leaf.",
                "This result comes from image-derived heuristics because no labeled dataset was available.",
            ],
            "mode": "heuristic",
        }

    confidence = 0.55 + min(0.35, green_ratio / 2 + circularity / 5)
    return {
        "predicted_label": "healthy",
        "confidence": min(0.94, confidence),
        "health_status": "healthy",
        "notes": [
            "Leaf segmentation is dominated by green pixels and limited lesion-like regions.",
            "This result comes from image-derived heuristics because no labeled dataset was available.",
        ],
        "mode": "heuristic",
    }


def ml_prediction(features: dict[str, float], dataset_root: Path) -> dict | None:
    cache_key = f"train_features_{dataset_signature(dataset_root)}"
    feature_df = load_or_create_dataframe(
        cache_key,
        lambda: dataset_to_dataframe(dataset_root),
        {"dataset_root": str(dataset_root)},
    )
    if feature_df.empty:
        return None

    model_payload = train_or_load_random_forest(f"rf_model_{dataset_signature(dataset_root)}", feature_df)
    if model_payload is None:
        return None

    ordered_columns = model_payload["columns"]
    sample = np.asarray([[float(features[column]) for column in ordered_columns]])
    probabilities = model_payload["model"].predict_proba(sample)[0]
    best_index = int(np.argmax(probabilities))
    predicted_label = str(model_payload["label_encoder"].inverse_transform([best_index])[0])
    confidence = float(probabilities[best_index])

    if predicted_label == "healthy":
        health_status = "healthy"
    elif confidence >= 0.75:
        health_status = "critical"
    else:
        health_status = "warning"

    return {
        "predicted_label": predicted_label,
        "confidence": confidence,
        "health_status": health_status,
        "notes": [
            f"Prediction generated from {model_payload['training_samples']} labeled dataset images.",
            "Model used: RandomForest on extracted color, texture, and shape features.",
        ],
        "mode": "ml",
        "training_samples": int(model_payload["training_samples"]),
        "known_labels": model_payload["known_labels"],
    }


def run_inference(image_path: Path, dataset_root: Path) -> dict:
    image_bgr = load_image(image_path)
    processed = preprocess_image(image_bgr)
    segmentation = segment_leaf(processed["rgb"])
    segmentation["hsv"] = processed["hsv"]
    features = extract_features_from_rgb(processed["rgb"])
    stats = compute_segmentation_stats(segmentation, processed["rgb"])

    prediction = ml_prediction(features, dataset_root)
    if prediction is None:
        prediction = heuristic_prediction(features, stats)

    _, feature_values = build_feature_vector(features)

    return {
        "diagnosis": prediction["predicted_label"],
        "confidence": round(float(prediction["confidence"]), 4),
        "health_status": prediction["health_status"],
        "notes": prediction["notes"],
        "mode": prediction["mode"],
        "stats": {
            **{key: round(float(value), 4) for key, value in stats.items()},
            "feature_count": len(feature_values),
        },
        "training_samples": int(prediction.get("training_samples", 0)),
        "known_labels": prediction.get("known_labels", []),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--dataset-root", required=True)
    args = parser.parse_args()

    try:
        payload = run_inference(Path(args.image), Path(args.dataset_root))
    except Exception as exc:
        print(json.dumps({"error": str(exc)}))
        return 1

    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
