"""
Index a directory of images: scan, embed each with Vertex, save to data/embeddings.npy + paths.json.
"""

import json
import logging
import time
from pathlib import Path

import numpy as np

from src.config import (
    DATA_DIR,
    EMBEDDING_DIMENSION,
    IMAGES_ROOT,
    INDEX_EMBEDDINGS_PATH,
    INDEX_PATHS_PATH,
)
from src.embedding import embed_image, get_model

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


def discover_images(directory: Path) -> list[Path]:
    """Return list of image file paths under directory (non-recursive by default)."""
    if not directory.is_dir():
        return []
    paths = []
    for p in directory.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            paths.append(p)
    return sorted(paths)


def index_directory(
    directory: Path | None = None,
    embeddings_path: Path | None = None,
    paths_path: Path | None = None,
    dimension: int = EMBEDDING_DIMENSION,
    throttle_seconds: float = 0.5,
) -> tuple[list[str], np.ndarray]:
    """
    Scan directory for images, embed each, save to embeddings_path and paths_path.
    Returns (list of relative path strings, embedding matrix).
    """
    directory = directory or IMAGES_ROOT
    embeddings_path = embeddings_path or INDEX_EMBEDDINGS_PATH
    paths_path = paths_path or INDEX_PATHS_PATH

    paths_list = discover_images(directory)
    if not paths_list:
        logger.warning("No images found in %s", directory)
        return [], np.array([])

    # Ensure model is loaded once
    get_model()

    path_strings: list[str] = []
    embeddings_list: list[list[float]] = []

    total = len(paths_list)
    logger.info("Starting index: %d images (one-by-one, throttle=%.1fs)", total, throttle_seconds)
    # Process one image at a time (no batching) to avoid timeouts and rate limits.
    for i, path in enumerate(paths_list):
        n = i + 1
        logger.info("Embedding %d/%d: %s", n, total, path.name)
        try:
            emb = embed_image(path, dimension=dimension)
            if not emb:
                logger.warning("Empty embedding for %s (API returned no vector)", path)
                continue
            # Store path relative to directory for portable index
            rel = path.relative_to(directory)
            path_strings.append(str(rel).replace("\\", "/"))
            embeddings_list.append(emb)
        except Exception as e:
            reason = type(e).__name__
            if hasattr(e, "response") and e.response is not None:
                reason = f"{reason} (status={getattr(e.response, 'status_code', '?')})"
            logger.exception("Skipping %s: %s — %s", path.name, reason, e)
        # Throttle between requests (one-by-one)
        if throttle_seconds > 0 and i < len(paths_list) - 1:
            time.sleep(throttle_seconds)

    skipped = total - len(path_strings)
    if skipped:
        logger.warning("Done: %d indexed, %d skipped (see above for why each failed)", len(path_strings), skipped)
    matrix = np.array(embeddings_list, dtype=np.float32) if embeddings_list else np.array([])

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if path_strings:
        np.save(embeddings_path, matrix)
        with open(paths_path, "w", encoding="utf-8") as f:
            json.dump(path_strings, f, indent=2)
        logger.info("Indexed %d images -> %s, %s", len(path_strings), embeddings_path, paths_path)
    else:
        logger.warning("No images indexed; nothing written to %s", embeddings_path)

    return path_strings, matrix


def main() -> None:
    import argparse
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Index images in a directory with Vertex embeddings")
    parser.add_argument("dir", nargs="?", default=None, help="Directory to index (default: test_dir)")
    parser.add_argument("--throttle", type=float, default=1.0, help="Seconds between API calls (one image at a time)")
    args = parser.parse_args()

    directory = Path(args.dir) if args.dir else IMAGES_ROOT
    index_directory(directory=directory, throttle_seconds=args.throttle)


if __name__ == "__main__":
    main()
