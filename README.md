# Local Image Search

A local image search app that indexes a directory of images using **Google Vertex AI** multimodal embeddings, then lets you search by **text** or **image** (text-to-image and image-to-image). Images are stored locally; embeddings are computed via the Vertex API.

## Setup

### 1. Virtual environment

```bash
python -m venv .venv
```

- **Windows (PowerShell):** `.venv\Scripts\Activate.ps1`
- **Windows (cmd):** `.venv\Scripts\activate.bat`
- **macOS/Linux:** `source .venv/bin/activate`

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Environment (.env or key.env)

Copy the example and put your key and project settings in `.env` (or `key.env`):

```bash
copy .env.example .env
```

Edit `.env`:

- **GOOGLE_CLOUD_PROJECT** (or **PROJECT_ID**): Your GCP project ID (required for Vertex AI).
- **GOOGLE_API_KEY** (or **API_KEY**): Your Google Cloud API key (e.g. from Vertex express mode). The code loads this from `.env` or `key.env` and uses it when the client supports it.
- **GOOGLE_APPLICATION_CREDENTIALS**: Path to your service account JSON key file (optional if using API key).
- **VERTEX_LOCATION**: Vertex AI region (optional; default `us-central1`).

The app loads `.env` first, then `key.env` if present. Do not commit `.env` or `key.env`; they are in `.gitignore`.

### 4. Index your images

Put images in `test_dir` (or another directory), then run the indexer:

```bash
python -m src.indexer
```

Or specify a directory:

```bash
python -m src.indexer path/to/your/images
```

Optional: throttle API calls (seconds between requests):

```bash
python -m src.indexer --throttle 1.0
```

This writes `data/embeddings.npy` and `data/paths.json`. The index is used by the search UI.

### 5. Run the web app

```bash
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000). Use **Text search** to type a query (e.g. “sunset”, “person”) or **Image search** to upload an image and find similar images from the index.

## Usage

- **Text search:** Enter a short description; results are images whose embeddings are closest to the text embedding.
- **Image search:** Upload an image; results are images whose embeddings are closest to the uploaded image’s embedding.

Results show thumbnails and similarity scores. Click a result to open the full image.

## Project layout

| Path | Purpose |
|------|--------|
| `key.env` | GCP project and credentials (create from `key.env.example`) |
| `src/config.py` | Loads `key.env`; project, location, paths, dimension |
| `src/embedding.py` | Vertex init and helpers to embed image/text |
| `src/indexer.py` | Scan directory, embed images, write index to `data/` |
| `src/search.py` | Load index; search by text or image (cosine similarity) |
| `app.py` | Flask routes: `/`, `/api/search`, `/images/<path>` |
| `templates/index.html` | Single-page search UI (text + image tabs) |
| `static/style.css` | Styling for the search UI |
| `data/` | Index files (`embeddings.npy`, `paths.json`) — created by indexer |

## Requirements

- Python 3.10+
- GCP project with Vertex AI API enabled and quota for `multimodalembedding@001`
- Supported image formats (for indexing): JPG, JPEG, PNG, BMP, GIF
