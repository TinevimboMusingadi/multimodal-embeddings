"""Vertex AI multimodal embedding client. Init once; embed image (path/bytes) and text."""

import base64
import logging
import os
import tempfile
from pathlib import Path
from typing import List, Union

from src.config import EMBEDDING_DIMENSION, GOOGLE_API_KEY, PROJECT_ID, VERTEX_LOCATION

logger = logging.getLogger(__name__)

# When API key is set we use REST; otherwise the Vertex SDK (ADC).
_use_rest = bool(GOOGLE_API_KEY)
_model_sdk = None

if not _use_rest:
    import vertexai
    from vertexai.vision_models import Image, MultiModalEmbeddingModel


def _ensure_init() -> None:
    if not PROJECT_ID:
        raise ValueError("PROJECT_ID or GOOGLE_CLOUD_PROJECT must be set in .env or key.env")
    if GOOGLE_API_KEY:
        os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    if not _use_rest:
        import vertexai as _v
        _v.init(project=PROJECT_ID, location=VERTEX_LOCATION)


def _embed_image_rest(image_b64: str, mime_type: str, dimension: int) -> List[float]:
    """Call Vertex predict REST API with API key for image embedding."""
    import requests
    url = (
        f"https://{VERTEX_LOCATION}-aiplatform.googleapis.com/v1"
        f"/projects/{PROJECT_ID}/locations/{VERTEX_LOCATION}"
        f"/publishers/google/models/multimodalembedding@001:predict"
        f"?key={GOOGLE_API_KEY}"
    )
    body = {
        "instances": [{"image": {"bytesBase64Encoded": image_b64, "mimeType": mime_type}}],
        "parameters": {"dimension": dimension},
    }
    payload_kb = len(image_b64) * 3 // 4 // 1024  # approx decoded size in KB
    logger.debug("Embedding image: %s, payload ~%d KB, timeout=180s", mime_type, payload_kb)
    try:
        r = requests.post(url, json=body, timeout=180)
        r.raise_for_status()
    except requests.exceptions.Timeout as e:
        logger.error("Image embed failed: request timed out after 180s (large image or slow network). Payload ~%d KB", payload_kb, exc_info=True)
        raise
    except requests.exceptions.RequestException as e:
        resp = getattr(e, "response", None)
        status = getattr(resp, "status_code", None) if resp else None
        body_preview = (resp.text[:500] if resp and getattr(resp, "text", None) else "no body") if resp else "no response"
        logger.error("Image embed failed: %s (status=%s). Response: %s", e, status, body_preview, exc_info=True)
        raise
    data = r.json()
    preds = data.get("predictions", [])
    if not preds:
        logger.warning("Image embed returned no predictions (empty response)")
        return []
    emb = preds[0].get("imageEmbedding") or preds[0].get("image_embedding")
    return list(emb) if emb else []


def _embed_text_rest(text: str, dimension: int) -> List[float]:
    """Call Vertex predict REST API with API key for text embedding."""
    import requests
    url = (
        f"https://{VERTEX_LOCATION}-aiplatform.googleapis.com/v1"
        f"/projects/{PROJECT_ID}/locations/{VERTEX_LOCATION}"
        f"/publishers/google/models/multimodalembedding@001:predict"
        f"?key={GOOGLE_API_KEY}"
    )
    body = {
        "instances": [{"text": text}],
        "parameters": {"dimension": dimension},
    }
    try:
        r = requests.post(url, json=body, timeout=180)
        r.raise_for_status()
    except requests.exceptions.Timeout as e:
        logger.error("Text embed failed: request timed out after 180s", exc_info=True)
        raise
    except requests.exceptions.RequestException as e:
        resp = getattr(e, "response", None)
        status = getattr(resp, "status_code", None) if resp else None
        body_preview = (resp.text[:500] if resp and getattr(resp, "text", None) else "no body") if resp else "no response"
        logger.error("Text embed failed: %s (status=%s). Response: %s", e, status, body_preview, exc_info=True)
        raise
    data = r.json()
    preds = data.get("predictions", [])
    if not preds:
        logger.warning("Text embed returned no predictions (empty response)")
        return []
    emb = preds[0].get("textEmbedding") or preds[0].get("text_embedding")
    return list(emb) if emb else []


def get_model():
    """Lazy-init and return the model (SDK path) or None (REST path)."""
    global _model_sdk
    _ensure_init()
    if _use_rest:
        return None
    if _model_sdk is None:
        _model_sdk = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")
    return _model_sdk


def embed_image(image_input: Union[str, Path, bytes], dimension: int = EMBEDDING_DIMENSION) -> List[float]:
    """
    Embed a single image. image_input can be a file path (str or Path) or raw bytes.
    Returns the image embedding vector.
    """
    if _use_rest:
        if isinstance(image_input, bytes):
            b64 = base64.standard_b64encode(image_input).decode("ascii")
            mime = "image/jpeg"
        else:
            path = Path(image_input)
            if not path.is_file():
                raise FileNotFoundError(f"Image file not found: {path}")
            raw = path.read_bytes()
            b64 = base64.standard_b64encode(raw).decode("ascii")
            mime = "image/jpeg" if path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
        return _embed_image_rest(b64, mime, dimension)

    model = get_model()
    if isinstance(image_input, bytes):
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
    if _use_rest:
        return _embed_text_rest(text, dimension)
    model = get_model()
    emb = model.get_embeddings(text=text, dimension=dimension)
    return list(emb.text_embedding) if emb.text_embedding else []
