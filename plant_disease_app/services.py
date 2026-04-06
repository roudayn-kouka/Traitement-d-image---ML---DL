from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd

from plant_disease_app.config import SUPPORTED_EXTENSIONS, UPLOAD_DIR, ensure_directories
from plant_disease_app.database import Base, engine, get_session
from plant_disease_app.image_pipeline import (
    build_feature_row,
    extract_features_from_rgb,
    features_from_json,
    features_to_json,
    load_image,
    preprocess_image,
    segment_leaf,
)
from plant_disease_app.models import FeatureRecord, ImageRecord, PredictionRecord


def init_database() -> None:
    """Initialise la base et les répertoires associés."""
    ensure_directories()
    Base.metadata.create_all(bind=engine)


def save_uploaded_file(uploaded_file, category: str | None = None) -> ImageRecord:
    """Sauvegarde un fichier uploadé sur disque puis crée son enregistrement en base."""
    extension = Path(uploaded_file.name).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Extension non supportée : {extension}")

    unique_name = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{uuid4().hex}{extension}"
    stored_path = UPLOAD_DIR / unique_name
    uploaded_file.seek(0)
    with stored_path.open("wb") as destination:
        shutil.copyfileobj(uploaded_file, destination)

    with get_session() as session:
        record = ImageRecord(
            original_name=uploaded_file.name,
            stored_path=str(stored_path),
            category=category or None,
            mime_type=getattr(uploaded_file, "type", None),
        )
        session.add(record)
        session.flush()
        session.refresh(record)
        session.expunge(record)
        return record


def process_and_store_features(image_id: int) -> dict:
    """Prétraite une image, extrait ses features puis les persiste."""
    with get_session() as session:
        record = session.get(ImageRecord, image_id)
        if record is None:
            raise ValueError(f"Image introuvable pour l'identifiant {image_id}")

        image_bgr = load_image(record.stored_path)
        processed = preprocess_image(image_bgr)
        segmentation = segment_leaf(processed["rgb"])
        features = extract_features_from_rgb(processed["rgb"])

        existing_feature = record.features
        payload = features_to_json(features)
        if existing_feature:
            existing_feature.feature_json = payload
        else:
            session.add(FeatureRecord(image_id=record.id, feature_json=payload))

        session.flush()
        session.refresh(record)
        session.expunge(record)
        return {
            "record": record,
            "processed": processed,
            "segmentation": segmentation,
            "features": features,
        }


def list_images_with_history() -> pd.DataFrame:
    """Retourne l'historique consolidé images/features/prédictions."""
    with get_session() as session:
        records = session.query(ImageRecord).order_by(ImageRecord.uploaded_at.desc()).all()
        rows: list[dict] = []
        for record in records:
            prediction_summary = ", ".join(
                f"{prediction.model_name}: {prediction.predicted_label}"
                for prediction in record.predictions
            )
            rows.append(
                {
                    "id": record.id,
                    "nom": record.original_name,
                    "categorie": record.category or "",
                    "date_upload": record.uploaded_at,
                    "chemin": record.stored_path,
                    "features_extraites": bool(record.features),
                    "predictions": prediction_summary,
                }
            )
        return pd.DataFrame(rows)


def build_feature_dataframe(dataset_root: str | None = None) -> pd.DataFrame:
    """Assemble features venant de la base et d'un dataset local optionnel."""
    rows: list[dict] = []
    with get_session() as session:
        records = session.query(ImageRecord).all()
        for record in records:
            if record.features:
                feature_dict = features_from_json(record.features.feature_json)
            else:
                feature_dict = build_feature_row(record.stored_path, record.category)
                feature_dict.pop("image_path", None)
                feature_dict.pop("label", None)

            feature_dict["image_path"] = record.stored_path
            feature_dict["label"] = record.category or "unknown"
            rows.append(feature_dict)

    db_df = pd.DataFrame(rows)
    if dataset_root:
        from plant_disease_app.image_pipeline import dataset_to_dataframe

        dataset_df = dataset_to_dataframe(dataset_root)
        if not dataset_df.empty:
            return pd.concat([dataset_df, db_df], ignore_index=True)
    return db_df


def save_prediction(image_id: int, model_name: str, predicted_label: str, confidence: float | None, metrics: dict | None) -> None:
    """Ajoute une prédiction au journal des résultats."""
    with get_session() as session:
        session.add(
            PredictionRecord(
                image_id=image_id,
                model_name=model_name,
                predicted_label=predicted_label,
                confidence=confidence,
                metrics_json=json.dumps(metrics) if metrics else None,
            )
        )
