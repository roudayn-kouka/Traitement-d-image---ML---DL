import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops

from plant_disease_app.config import IMAGE_SIZE, SUPPORTED_EXTENSIONS


def load_image(path: str | Path) -> np.ndarray:
    """Charge une image BGR depuis le disque et valide sa presence."""
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Impossible de charger l'image : {path}")
    return image


def preprocess_image(image_bgr: np.ndarray) -> dict[str, np.ndarray]:
    """Prepare plusieurs representations utiles au pipeline."""
    resized = cv2.resize(image_bgr, IMAGE_SIZE)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(rgb, (5, 5), 0)
    return {
        "resized_bgr": resized,
        "rgb": rgb,
        "hsv": hsv,
        "gray": gray,
        "blurred": blurred,
    }


def segment_leaf(rgb_image: np.ndarray) -> dict[str, np.ndarray]:
    """Segmente la feuille via un seuil HSV puis extrait les contours Canny."""
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    lower_green = np.array([20, 30, 20], dtype=np.uint8)
    upper_green = np.array([95, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_green, upper_green)
    kernel = np.ones((5, 5), np.uint8)
    cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)
    segmented = cv2.bitwise_and(rgb_image, rgb_image, mask=cleaned_mask)
    contours = cv2.Canny(segmented, threshold1=50, threshold2=150)
    contour_overlay = rgb_image.copy()
    contour_list, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(contour_overlay, contour_list, -1, (255, 0, 0), 2)
    return {
        "mask": cleaned_mask,
        "segmented": segmented,
        "contours": contours,
        "contour_overlay": contour_overlay,
        "contour_list": contour_list,
    }


def color_histograms(rgb_image: np.ndarray, hsv_image: np.ndarray, bins: int = 16) -> dict[str, float]:
    """Calcule des histogrammes normalises pour les espaces RGB et HSV."""
    features: dict[str, float] = {}
    for color_space_name, image in {"rgb": rgb_image, "hsv": hsv_image}.items():
        for channel_index in range(3):
            hist = cv2.calcHist([image], [channel_index], None, [bins], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            for idx, value in enumerate(hist):
                features[f"{color_space_name}_hist_c{channel_index}_{idx}"] = float(value)
    return features


def glcm_features(gray_image: np.ndarray) -> dict[str, float]:
    """Extrait les metriques de texture GLCM."""
    quantized = (gray_image / 32).astype(np.uint8)
    glcm = graycomatrix(quantized, distances=[1], angles=[0], levels=8, symmetric=True, normed=True)
    return {
        "glcm_contrast": float(graycoprops(glcm, "contrast")[0, 0]),
        "glcm_homogeneity": float(graycoprops(glcm, "homogeneity")[0, 0]),
        "glcm_energy": float(graycoprops(glcm, "energy")[0, 0]),
    }


def shape_features(mask: np.ndarray, contour_list: list[np.ndarray]) -> dict[str, float]:
    """Mesure des caracteristiques geometriques de la feuille."""
    area = float(np.count_nonzero(mask))
    perimeter = 0.0
    circularity = 0.0
    if contour_list:
        largest_contour = max(contour_list, key=cv2.contourArea)
        perimeter = float(cv2.arcLength(largest_contour, True))
        contour_area = float(cv2.contourArea(largest_contour))
        if perimeter > 0:
            circularity = float((4 * np.pi * contour_area) / (perimeter ** 2))
    return {
        "shape_area": area,
        "shape_perimeter": perimeter,
        "shape_circularity": circularity,
    }


def extract_features_from_rgb(rgb_image: np.ndarray) -> dict[str, float]:
    """Orchestre extraction couleur, texture et forme pour une image RGB."""
    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    segmentation = segment_leaf(rgb_image)

    features: dict[str, float] = {}
    features.update(color_histograms(rgb_image, hsv_image))
    features.update(glcm_features(gray_image))
    features.update(shape_features(segmentation["mask"], segmentation["contour_list"]))
    return features


def build_feature_row(image_path: str | Path, label: str | None = None) -> dict[str, float | str]:
    """Construit une ligne exploitable par pandas/scikit-learn."""
    image = load_image(image_path)
    processed = preprocess_image(image)
    features = extract_features_from_rgb(processed["rgb"])
    features["image_path"] = str(image_path)
    features["label"] = label if label else "unknown"
    return features


def resolve_dataset_split_root(dataset_root: str | Path, split: str = "train") -> Path:
    """Resout le repertoire d'un split de dataset pour plusieurs layouts courants."""
    dataset_root = Path(dataset_root)
    if (dataset_root / split).is_dir():
        return dataset_root / split

    direct_subdirs = [path for path in sorted(dataset_root.iterdir()) if path.is_dir()] if dataset_root.exists() else []
    if len(direct_subdirs) == 1 and (direct_subdirs[0] / split).is_dir():
        return direct_subdirs[0] / split

    return dataset_root


def dataset_to_dataframe(dataset_root: str | Path) -> pd.DataFrame:
    """Parcourt un dataset structure par sous-dossiers de classes."""
    dataset_root = resolve_dataset_split_root(dataset_root, split="train")
    rows: list[dict[str, float | str]] = []
    if not dataset_root.exists():
        return pd.DataFrame()

    for class_dir in sorted(dataset_root.iterdir()):
        if not class_dir.is_dir():
            continue
        label = class_dir.name
        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            rows.append(build_feature_row(image_path, label))
    return pd.DataFrame(rows)


def features_to_json(features: dict[str, float]) -> str:
    """Serialise proprement les features vers la base de donnees."""
    return json.dumps(features)


def features_from_json(payload: str) -> dict[str, float]:
    """Deserialise les features depuis la base de donnees."""
    return json.loads(payload)
