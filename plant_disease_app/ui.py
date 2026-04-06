from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from plant_disease_app.dl_pipeline import train_deep_learning_model
from plant_disease_app.ml_pipeline import train_classical_models
from plant_disease_app.services import (
    build_feature_dataframe,
    init_database,
    list_images_with_history,
    process_and_store_features,
    save_prediction,
    save_uploaded_file,
)
from plant_disease_app.visualization import (
    image_exists,
    plot_precision_recall_curves,
    plot_training_history,
    split_correct_incorrect,
)


def render_image_preview(uploaded_files) -> None:
    """Affiche un aperçu des images avant enregistrement/traitement."""
    if not uploaded_files:
        return
    st.subheader("Aperçu avant traitement")
    columns = st.columns(max(1, min(3, len(uploaded_files))))
    for idx, uploaded in enumerate(uploaded_files):
        uploaded.seek(0)
        columns[idx % len(columns)].image(Image.open(uploaded), caption=uploaded.name, use_container_width=True)


def render_processing_result(result: dict) -> None:
    """Présente les étapes de prétraitement et segmentation."""
    processed = result["processed"]
    segmentation = result["segmentation"]

    st.subheader(f"Prétraitement et segmentation : {result['record'].original_name}")
    col1, col2 = st.columns(2)
    col1.image(processed["rgb"], caption="Image redimensionnée (RGB)", use_container_width=True)
    col2.image(processed["blurred"], caption="Après filtrage gaussien", use_container_width=True)

    col3, col4 = st.columns(2)
    col3.image(segmentation["mask"], caption="Masque HSV", use_container_width=True)
    col4.image(segmentation["contour_overlay"], caption="Contours détectés", use_container_width=True)

    with st.expander("Features extraites"):
        st.dataframe(pd.DataFrame([result["features"]]), use_container_width=True)


def render_training_results(classical_results, deep_result) -> None:
    """Affiche les métriques et comparaisons des modèles."""
    st.header("Comparaison des modèles")

    comparison_rows: list[dict] = []
    for result in classical_results:
        comparison_rows.append(
            {
                "model": result.model_name,
                "accuracy": result.metrics["accuracy"],
                "precision": result.metrics["precision"],
                "recall": result.metrics["recall"],
                "f1_score": result.metrics["f1_score"],
            }
        )
    comparison_rows.append({"model": "CNN", **deep_result.metrics})
    st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True)

    for result in classical_results:
        st.subheader(f"Métriques ML classique : {result.model_name}")
        st.json(
            {
                "accuracy": result.metrics["accuracy"],
                "precision": result.metrics["precision"],
                "recall": result.metrics["recall"],
                "f1_score": result.metrics["f1_score"],
            }
        )
        if result.pr_curves:
            st.pyplot(plot_precision_recall_curves(result.pr_curves, f"Precision / Recall - {result.model_name}"))
        correct_df, incorrect_df = split_correct_incorrect(result.predictions_df)
        st.write(f"Prédictions correctes : {len(correct_df)}")
        st.write(f"Prédictions incorrectes : {len(incorrect_df)}")
        if not incorrect_df.empty:
            st.dataframe(incorrect_df.head(10), use_container_width=True)
            preview_cols = st.columns(min(3, len(incorrect_df.head(3))))
            for idx, (_, row) in enumerate(incorrect_df.head(3).iterrows()):
                if image_exists(str(row["image_path"])):
                    preview_cols[idx].image(
                        str(Path(row["image_path"])),
                        caption=f"Vrai: {row['true_label']} | Prédit: {row['predicted_label']}",
                        use_container_width=True,
                    )

    st.subheader("Deep Learning : CNN")
    st.json(deep_result.metrics)
    st.pyplot(plot_training_history(deep_result.history, "Historique d'entraînement CNN"))
    correct_df, incorrect_df = split_correct_incorrect(deep_result.predictions_df)
    st.write(f"Prédictions correctes : {len(correct_df)}")
    st.write(f"Prédictions incorrectes : {len(incorrect_df)}")
    if not incorrect_df.empty:
        st.dataframe(incorrect_df.head(10), use_container_width=True)
        preview_cols = st.columns(min(3, len(incorrect_df.head(3))))
        for idx, (_, row) in enumerate(incorrect_df.head(3).iterrows()):
            if image_exists(str(row["image_path"])):
                preview_cols[idx].image(
                    str(Path(row["image_path"])),
                    caption=f"Vrai: {row['true_label']} | Prédit: {row['predicted_label']}",
                    use_container_width=True,
                )


def persist_inference_examples(classical_results, deep_result, image_index_lookup: dict[str, int]) -> None:
    """Enregistre les prédictions des modèles dans l'historique."""
    for result in classical_results:
        for _, row in result.predictions_df.iterrows():
            image_id = image_index_lookup.get(str(row["image_path"]))
            if image_id:
                save_prediction(
                    image_id=image_id,
                    model_name=result.model_name,
                    predicted_label=str(row["predicted_label"]),
                    confidence=None,
                    metrics=result.metrics,
                )

    for _, row in deep_result.predictions_df.iterrows():
        image_id = image_index_lookup.get(str(row["image_path"]))
        if image_id:
            save_prediction(
                image_id=image_id,
                model_name="CNN",
                predicted_label=str(row["predicted_label"]),
                confidence=float(row["confidence"]),
                metrics=deep_result.metrics,
            )


def render_history() -> None:
    """Affiche l'historique des traitements et résultats."""
    st.header("Historique")
    history_df = list_images_with_history()
    if history_df.empty:
        st.info("Aucune image n'a encore été traitée.")
        return

    st.dataframe(history_df, use_container_width=True)
    preview_rows = history_df.head(6)
    if not preview_rows.empty:
        cols = st.columns(min(3, len(preview_rows)))
        for idx, (_, row) in enumerate(preview_rows.iterrows()):
            if image_exists(row["chemin"]):
                cols[idx % len(cols)].image(
                    str(Path(row["chemin"])),
                    caption=f"{row['nom']} | {row['predictions']}",
                    use_container_width=True,
                )


def main() -> None:
    """Point d'entrée principal de l'interface Streamlit."""
    st.set_page_config(page_title="Plant Disease Pipeline", layout="wide")
    init_database()

    st.title("Pipeline complet de détection de maladies sur feuilles")
    st.write(
        "Upload d'images, stockage en base, prétraitement, segmentation, "
        "extraction de caractéristiques, classification ML et Deep Learning."
    )

    with st.sidebar:
        st.header("Paramètres")
        dataset_root = st.text_input("Chemin du dataset supervisé", value="data/dataset")
        default_category = st.text_input("Catégorie associée aux uploads (optionnel)", value="")
        dl_epochs = st.slider("Epochs CNN", min_value=3, max_value=15, value=5)
        dl_batch_size = st.select_slider("Batch size CNN", options=[8, 16, 32], value=16)

    uploaded_files = st.file_uploader(
        "Téléchargez une ou plusieurs images de feuilles",
        type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=True,
    )
    render_image_preview(uploaded_files)

    if "current_upload_ids" not in st.session_state:
        st.session_state.current_upload_ids = {}
    if "processed_records" not in st.session_state:
        st.session_state.processed_records = []
    if "training_payload" not in st.session_state:
        st.session_state.training_payload = None

    if st.button("Stocker et traiter les images", type="primary"):
        if not uploaded_files:
            st.warning("Ajoutez au moins une image avant de lancer le traitement.")
        else:
            st.session_state.processed_records = []
            st.session_state.current_upload_ids = {}
            for uploaded_file in uploaded_files:
                record = save_uploaded_file(uploaded_file, category=default_category.strip() or None)
                st.session_state.current_upload_ids[record.stored_path] = record.id
                result = process_and_store_features(record.id)
                st.session_state.processed_records.append(result)
            st.success(f"{len(st.session_state.processed_records)} image(s) sauvegardée(s) et traitée(s).")

    for result in st.session_state.processed_records:
        render_processing_result(result)

    st.header("Extraction de caractéristiques")
    feature_df = build_feature_dataframe(dataset_root)
    if feature_df.empty:
        st.info("Aucune feature disponible pour le moment.")
    else:
        st.dataframe(feature_df.head(20), use_container_width=True)

    if st.button("Entraîner et comparer les modèles"):
        try:
            feature_df = build_feature_dataframe(dataset_root)
            classical_results = train_classical_models(feature_df)
            deep_result = train_deep_learning_model(
                feature_df,
                epochs=dl_epochs,
                batch_size=dl_batch_size,
            )
            persist_inference_examples(
                classical_results,
                deep_result,
                st.session_state.current_upload_ids,
            )
            st.session_state.training_payload = (classical_results, deep_result)
        except Exception as exc:
            st.error(f"Impossible d'entraîner les modèles : {exc}")

    if st.session_state.training_payload:
        classical_results, deep_result = st.session_state.training_payload
        render_training_results(classical_results, deep_result)

    render_history()
