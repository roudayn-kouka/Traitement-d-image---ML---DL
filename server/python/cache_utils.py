from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "server" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _iter_dataset_files(root: Path):
    if not root.exists():
        return
    for path in sorted(root.rglob("*")):
        if path.is_file():
            yield path


def dataset_signature(*roots: Path) -> str:
    hasher = hashlib.sha256()
    for root in roots:
        hasher.update(str(root.resolve()).encode("utf-8"))
        if not root.exists():
            hasher.update(b"missing")
            continue
        for path in _iter_dataset_files(root):
            stat = path.stat()
            hasher.update(str(path.relative_to(root)).encode("utf-8"))
            hasher.update(str(stat.st_size).encode("utf-8"))
            hasher.update(str(int(stat.st_mtime)).encode("utf-8"))
    return hasher.hexdigest()[:24]


def cache_paths(cache_key: str) -> dict[str, Path]:
    prefix = CACHE_DIR / cache_key
    return {
      "meta": prefix.with_suffix(".meta.json"),
      "dataframe": prefix.with_suffix(".pkl"),
      "model": prefix.with_suffix(".joblib"),
      "report": prefix.with_suffix(".report.json"),
    }


def load_cached_dataframe(cache_key: str):
    paths = cache_paths(cache_key)
    if paths["dataframe"].exists():
        try:
            return pd.read_pickle(paths["dataframe"])
        except Exception:
            paths["dataframe"].unlink(missing_ok=True)
            paths["meta"].unlink(missing_ok=True)
    return None


def save_cached_dataframe(cache_key: str, dataframe: pd.DataFrame, meta: dict) -> None:
    paths = cache_paths(cache_key)
    dataframe.to_pickle(paths["dataframe"])
    paths["meta"].write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_or_create_dataframe(cache_key: str, build_fn, meta: dict) -> pd.DataFrame:
    cached = load_cached_dataframe(cache_key)
    if cached is not None:
        return cached
    dataframe = build_fn()
    save_cached_dataframe(cache_key, dataframe, meta)
    return dataframe


def load_cached_model(cache_key: str):
    paths = cache_paths(cache_key)
    if paths["model"].exists():
        try:
            return joblib.load(paths["model"])
        except Exception:
            paths["model"].unlink(missing_ok=True)
    return None


def save_cached_model(cache_key: str, payload: dict) -> None:
    paths = cache_paths(cache_key)
    joblib.dump(payload, paths["model"])


def train_or_load_random_forest(cache_key: str, feature_df: pd.DataFrame):
    cached = load_cached_model(cache_key)
    if cached is not None:
        return cached

    labeled_df = feature_df[feature_df["label"] != "unknown"].copy()
    if labeled_df.empty or labeled_df["label"].nunique() < 2:
        return None

    X = labeled_df.drop(columns=["label", "image_path"], errors="ignore")
    y_raw = labeled_df["label"].astype(str).to_numpy()

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)

    payload = {
        "model": model,
        "label_encoder": label_encoder,
        "columns": list(X.columns),
        "training_samples": int(len(labeled_df)),
        "known_labels": sorted({str(label) for label in y_raw}),
    }
    save_cached_model(cache_key, payload)
    return payload


def load_cached_report(cache_key: str):
    path = cache_paths(cache_key)["report"]
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def save_cached_report(cache_key: str, report: dict) -> None:
    path = cache_paths(cache_key)["report"]
    path.write_text(json.dumps(report), encoding="utf-8")
