"""Load config from .env (or key.env). Single place for project, location, key, index path, dimension."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env first, then key.env (so .env overrides). Either file can hold the key and project settings.
_root = Path(__file__).resolve().parent.parent
load_dotenv(_root / ".env", override=False)
load_dotenv(_root / "key.env", override=False)

PROJECT_ID: str = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("PROJECT_ID") or ""
VERTEX_LOCATION: str = os.environ.get("VERTEX_LOCATION", "us-central1")
EMBEDDING_DIMENSION: int = 512

# Paths
DATA_DIR: Path = _root / "data"
INDEX_EMBEDDINGS_PATH: Path = DATA_DIR / "embeddings.npy"
INDEX_PATHS_PATH: Path = DATA_DIR / "paths.json"
IMAGES_ROOT: Path = _root / "test_dir"
