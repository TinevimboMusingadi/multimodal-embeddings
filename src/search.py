"""
Load index from disk; search by text or image using cosine similarity.
"""

import json
from pathlib import Path
from typing import Union

import numpy as np

from src.config import EMBEDDING_DIMENSION, INDEX_EMBEDDINGS_PATH, INDEX_PATHS_PATH
from src.embedding import embed_image, embed_text


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between vector a and each row of b. Returns 1d array."""
    a = np.asarray(a, dtype=np.float32).reshape(1, -1)
    b = np.asarray(b, dtype=np.float32)
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return (an @ bn.T).ravel()


def load_index(
    embeddings_path: Path | None = None,
    paths_path: Path | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Load embeddings matrix and paths list from disk. Raises FileNotFoundError if missing."""
    embeddings_path = embeddings_path or INDEX_EMBEDDINGS_PATH
    paths_path = paths_path or INDEX_PATHS_PATH
    if not embeddings_path.is_file() or not paths_path.is_file():
        raise FileNotFoundError(
            f"Index not found. Run indexer first: {embeddings_path}, {paths_path}"
        )
    matrix = np.load(embeddings_path)
    with open(paths_path, encoding="utf-8") as f:
        paths = json.load(f)
    return matrix, paths


def search_by_text(
    query: str,
    top_k: int = 10,
    dimension: int = EMBEDDING_DIMENSION,
    embeddings_path: Path | None = None,
    paths_path: Path | None = None,
) -> list[dict]:
    """
    Embed query text, compute cosine similarity vs index, return top_k results.
    Each result: {"path": str, "score": float}.
    """
    matrix, paths = load_index(embeddings_path, paths_path)
    if matrix.size == 0 or not paths:
        return []
    q = embed_text(query, dimension=dimension)
    q = np.array(q, dtype=np.float32)
    scores = _cosine_similarity(q, matrix)
    order = np.argsort(scores)[::-1][:top_k]
    return [{"path": paths[i], "score": float(scores[i])} for i in order]


def search_by_image(
    image_input: Union[str, Path, bytes],
    top_k: int = 10,
    dimension: int = EMBEDDING_DIMENSION,
    embeddings_path: Path | None = None,
    paths_path: Path | None = None,
) -> list[dict]:
    """
    Embed query image, compute cosine similarity vs index, return top_k results.
    image_input: file path or raw bytes (e.g. uploaded file).
    Each result: {"path": str, "score": float}.
    """
    matrix, paths = load_index(embeddings_path, paths_path)
    if matrix.size == 0 or not paths:
        return []
    q = embed_image(image_input, dimension=dimension)
    q = np.array(q, dtype=np.float32)
    scores = _cosine_similarity(q, matrix)
    order = np.argsort(scores)[::-1][:top_k]
    return [{"path": paths[i], "score": float(scores[i])} for i in order]
