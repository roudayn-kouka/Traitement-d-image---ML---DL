from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from plant_disease_app.database import Base


class ImageRecord(Base):
    """Stocke les images uploadées et leur contexte métier."""

    __tablename__ = "images"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    original_name: Mapped[str] = mapped_column(String(255), nullable=False)
    stored_path: Mapped[str] = mapped_column(String(500), nullable=False, unique=True)
    uploaded_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    category: Mapped[str | None] = mapped_column(String(128), nullable=True)
    mime_type: Mapped[str | None] = mapped_column(String(128), nullable=True)

    features: Mapped["FeatureRecord | None"] = relationship(
        "FeatureRecord",
        back_populates="image",
        uselist=False,
        cascade="all, delete-orphan",
    )
    predictions: Mapped[list["PredictionRecord"]] = relationship(
        "PredictionRecord",
        back_populates="image",
        cascade="all, delete-orphan",
    )


class FeatureRecord(Base):
    """Persiste les caractéristiques extraites sous forme JSON sérialisée."""

    __tablename__ = "features"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    image_id: Mapped[int] = mapped_column(ForeignKey("images.id"), nullable=False, unique=True)
    feature_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    image: Mapped[ImageRecord] = relationship("ImageRecord", back_populates="features")


class PredictionRecord(Base):
    """Historise les prédictions générées par chaque modèle."""

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    image_id: Mapped[int] = mapped_column(ForeignKey("images.id"), nullable=False, index=True)
    model_name: Mapped[str] = mapped_column(String(64), nullable=False)
    predicted_label: Mapped[str] = mapped_column(String(128), nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    metrics_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    image: Mapped[ImageRecord] = relationship("ImageRecord", back_populates="predictions")
