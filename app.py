"""
Flask app: / (search UI), /api/search (POST text or image), /images/<path> (serve indexed images safely).
"""

import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory

from src.config import IMAGES_ROOT
from src.search import search_by_image, search_by_text

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB for image upload


def _safe_image_path(relative_path: str) -> Path | None:
    """Resolve relative_path under IMAGES_ROOT. Return None if path escapes root."""
    root = IMAGES_ROOT.resolve()
    parts = Path(relative_path).parts
    if ".." in parts or relative_path.startswith("/") or relative_path.startswith("\\"):
        return None
    full = (root / relative_path).resolve()
    try:
        if os.path.commonpath([str(root), str(full)]) != str(root):
            return None
    except ValueError:
        return None
    if not full.is_file():
        return None
    return full


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/search", methods=["POST"])
def api_search():
    text = request.form.get("text", "").strip()
    image = request.files.get("image")

    if image and image.filename:
        try:
            data = image.read()
            if not data:
                return jsonify({"error": "Empty image file"}), 400
            results = search_by_image(data, top_k=20)
            return jsonify({"results": results})
        except FileNotFoundError as e:
            return jsonify({"error": "Index not found. Run the indexer first.", "detail": str(e)}), 503
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    if text:
        try:
            results = search_by_text(text, top_k=20)
            return jsonify({"results": results})
        except FileNotFoundError as e:
            return jsonify({"error": "Index not found. Run the indexer first.", "detail": str(e)}), 503
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "Provide 'text' or 'image' in the request"}), 400


@app.route("/images/<path:subpath>")
def serve_image(subpath: str):
    safe = _safe_image_path(subpath)
    if safe is None:
        return "Not found", 404
    directory = safe.parent
    filename = safe.name
    return send_from_directory(directory, filename, as_attachment=False)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
