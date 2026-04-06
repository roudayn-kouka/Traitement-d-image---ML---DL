from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = DATA_DIR / "artifacts"
DB_PATH = DATA_DIR / "app.db"
DATABASE_URL = f"sqlite:///{DB_PATH.as_posix()}"
IMAGE_SIZE = (128, 128)
RANDOM_STATE = 42
TEST_SIZE = 0.2
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def ensure_directories() -> None:
    """Crée les répertoires utiles au runtime de l'application."""
    for directory in (DATA_DIR, UPLOAD_DIR, PROCESSED_DIR, ARTIFACTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
