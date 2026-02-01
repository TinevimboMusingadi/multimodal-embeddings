"""Vertex AI multimodal embedding client. Init once; embed image (path/bytes) and text."""

from pathlib import Path
from typing import List, Union

import vertexai
from vertexai.vision_models import Image, MultiModalEmbeddingModel

from src.config import EMBEDDING_DIMENSION, PROJECT_ID, VERTEX_LOCATION

_model: MultiModalEmbeddingModel | None = None


def _ensure_init() -> None:
    if PROJECT_ID is None or PROJECT_ID == "":
        raise ValueError("PROJECT_ID or GOOGLE_CLOUD_PROJECT must be set (e.g. in key.env)")
    vertexai.init(project=PROJECT_ID, location=VERTEX_LOCATION)


def get_model() -> MultiModalEmbeddingModel:
    """Lazy-init and return the multimodal embedding model."""
    global _model
    _ensure_init()
    if _model is None:
        _model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")
    return _model


def embed_image(image_input: Union[str, Path, bytes], dimension: int = EMBEDDING_DIMENSION) -> List[float]:
    """
    Embed a single image. image_input can be a file path (str or Path) or raw bytes.
    Returns the image embedding vector.
    """
    model = get_model()
    if isinstance(image_input, bytes):
        # SDK may accept image_bytes; otherwise we write to temp file and load.
        # Vertex Image can be constructed from bytes in some SDK versions.
        import tempfile
        # SDK Image.load_from_file needs a path; write bytes to temp file.
        # Use temp file for bytes to keep compatibility.
        suffix = ".jpg"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(image_input)
            path = f.name
        try:
            image = Image.load_from_file(path)
            emb = model.get_embeddings(image=image, dimension=dimension)
            return list(emb.image_embedding) if emb.image_embedding else []
        finally:
            Path(path).unlink(missing_ok=True)
    else:
        path = Path(image_input)
        if not path.is_file():
            raise FileNotFoundError(f"Image file not found: {path}")
        image = Image.load_from_file(str(path))
        emb = model.get_embeddings(image=image, dimension=dimension)
        return list(emb.image_embedding) if emb.image_embedding else []


def embed_text(text: str, dimension: int = EMBEDDING_DIMENSION) -> List[float]:
    """Embed a text query. Returns the text embedding vector."""
    model = get_model()
    emb = model.get_embeddings(text=text, dimension=dimension)
    return list(emb.text_embedding) if emb.text_embedding else []
