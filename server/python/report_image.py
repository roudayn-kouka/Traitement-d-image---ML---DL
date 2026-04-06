from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plant_disease_app.image_pipeline import extract_features_from_rgb, load_image, preprocess_image, segment_leaf  # noqa: E402


def png_data_url(image: np.ndarray) -> str:
    if image.ndim == 2:
      output = image
    else:
      output = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    success, encoded = cv2.imencode(".png", output)
    if not success:
        raise ValueError("Failed to encode image preview.")
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def normalize_grayscale(image: np.ndarray) -> np.ndarray:
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def compute_histogram_bundle(rgb: np.ndarray, hsv: np.ndarray) -> dict:
    histogram_bundle: dict[str, dict[str, list[float]]] = {}
    channel_names = {
        "rgb": ["R", "G", "B"],
        "hsv": ["H", "S", "V"],
    }

    for color_space, image in {"rgb": rgb, "hsv": hsv}.items():
        histogram_bundle[color_space] = {}
        for index, channel_name in enumerate(channel_names[color_space]):
            hist = cv2.calcHist([image], [index], None, [16], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            histogram_bundle[color_space][channel_name] = [round(float(value), 4) for value in hist]
    return histogram_bundle


def compute_segmentation_bundle(rgb: np.ndarray, gray: np.ndarray, hsv_mask: np.ndarray) -> dict:
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = normalize_grayscale(cv2.magnitude(sobel_x, sobel_y))
    canny = cv2.Canny(gray, 80, 180)
    _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    pixels = rgb.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 12, 1.0)
    _, labels, centers = cv2.kmeans(pixels, 3, None, criteria, 6, cv2.KMEANS_PP_CENTERS)
    clustered = centers[labels.flatten()].reshape(rgb.shape).astype(np.uint8)

    leaf_ratio = float(np.count_nonzero(hsv_mask)) / float(hsv_mask.size)
    threshold_ratio = float(np.count_nonzero(threshold)) / float(threshold.size)

    return {
        "images": {
            "sobel": png_data_url(sobel),
            "canny": png_data_url(canny),
            "threshold": png_data_url(threshold),
            "hsvMask": png_data_url(hsv_mask),
            "clustered": png_data_url(clustered),
        },
        "subjectiveComparison": [
            f"HSV isolates about {round(leaf_ratio * 100, 1)}% of the frame as potential leaf area.",
            f"Otsu threshold keeps about {round(threshold_ratio * 100, 1)}% of the frame as foreground.",
            "Canny highlights fine leaf borders while Sobel gives a smoother gradient-based contour map.",
            "K-means offers a coarse color grouping that is useful to compare with HSV-based isolation.",
        ],
    }


def feature_groups(features: dict[str, float]) -> dict:
    rgb_hist = {key: value for key, value in features.items() if key.startswith("rgb_hist")}
    hsv_hist = {key: value for key, value in features.items() if key.startswith("hsv_hist")}
    texture = {key: round(float(value), 4) for key, value in features.items() if key.startswith("glcm_")}
    shape = {key: round(float(value), 4) for key, value in features.items() if key.startswith("shape_")}

    return {
        "color": {
            "rgbHistogramFeatureCount": len(rgb_hist),
            "hsvHistogramFeatureCount": len(hsv_hist),
        },
        "texture": texture,
        "shape": shape,
        "vectorReadyForMl": {
            "totalFeatures": len(features),
            "isNumeric": all(isinstance(value, (int, float)) for value in features.values()),
            "ready": True,
        },
    }


def build_report(image_path: Path) -> dict:
    image_bgr = load_image(image_path)
    processed = preprocess_image(image_bgr)
    segmentation = segment_leaf(processed["rgb"])
    features = extract_features_from_rgb(processed["rgb"])
    histograms = compute_histogram_bundle(processed["rgb"], processed["hsv"])
    segmentation_bundle = compute_segmentation_bundle(processed["rgb"], processed["gray"], segmentation["mask"])

    return {
        "preprocessing": {
            "steps": [
                "Resize to 128x128",
                "Convert BGR to RGB",
                "Convert RGB to HSV",
                "Convert RGB to grayscale",
                "Gaussian blur filtering",
            ],
            "images": {
                "rgb": png_data_url(processed["rgb"]),
                "gray": png_data_url(processed["gray"]),
                "blurred": png_data_url(processed["blurred"]),
                "contourOverlay": png_data_url(segmentation["contour_overlay"]),
            },
            "histograms": histograms,
            "impactExamples": [
                "Blur reduces small noise before segmentation and contour extraction.",
                "HSV isolates color ranges more effectively than raw RGB for green leaf masking.",
                "Grayscale simplifies edge operators such as Sobel and Canny.",
            ],
        },
        "segmentation": segmentation_bundle,
        "features": feature_groups(features),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    try:
        payload = build_report(Path(args.image))
    except Exception as exc:
        print(json.dumps({"error": str(exc)}))
        return 1

    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
