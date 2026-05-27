# Avatar Motion Web - Local LivePortrait

This project now runs fully on your local machine:

- Node.js serves the web UI at `http://localhost:3000`.
- `/api/generate` runs `python/motion_avatar_pipeline.py` directly.
- LivePortrait must be available on the same machine with GPU/CUDA configured.
- No Render, Colab, or ngrok URL is required.

## Setup

Install Node dependencies:

```bash
npm install
```

Install the Python dependencies for this wrapper:

```bash
pip install -r requirements.txt
```

Install LivePortrait and its own dependencies separately, then set its path in `.env`:

```text
LIVEPORTRAIT_REPO=C:\path\to\LivePortrait
PYTHON_BIN=python
PORT=3000
```

If `LIVEPORTRAIT_REPO` is not set, the app looks for `LivePortrait` inside this project folder.

## Run

```bash
npm start
```

Open:

```text
http://localhost:3000
```

## Health Check

```text
http://localhost:3000/api/health
```

The response shows whether the local LivePortrait repo was found.

## Optional FastAPI Mode

The Node server is the main local app. If you still want to run the Python API directly:

```bash
npm run api
```

Then call `POST http://localhost:8000/generate`.
