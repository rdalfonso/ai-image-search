# ============================================================================
# Configuration
# ============================================================================
from pathlib import Path

class Config:
    """Application configuration"""
    MODEL_NAME = "google/gemma-3n-e4b"
    LM_STUDIO_URL = "http://localhost:1234/v1"
    IMAGES_DIR = Path("./images")
    RENAMED_DIR = Path("./images_renamed")
    CHROMA_DB_PATH = "./chroma_db"
    COLLECTION_NAME = "images-description-collection"
    DISTANCE_THRESHOLD = 1.2
    MAX_RESULTS = 5