# Fake News Detector (Full-Stack + Docker)

Production-ready conversion of the notebook-based fake news detector into a full-stack application.

## Project Structure

```text
fake-news-detector/
  backend/
    app.py
    model.py
    requirements.txt
    Dockerfile
    models/
      prediction_model.pkl
  frontend/
    src/
      App.js
      components/
        NewsChecker.js
      api.js
    package.json
    Dockerfile
  docker-compose.yml
  README.md
```

## Backend API

- Framework: FastAPI
- Endpoint: `POST /predict` (also available as `POST /api/predict` for Vercel/API gateway routing)
- Input:

```json
{
  "text": "Breaking news article text..."
}
```

- Output:

```json
{
  "prediction": "Fake",
  "confidence": 0.91
}
```

The notebook-based model artifact is loaded once during server startup via app lifespan hooks.

## Required Model Artifacts

Place these files in `backend/models/`:
- `prediction_model.pkl`

You can generate them with the extracted training script:

```bash
cd backend
pip install -r requirements.txt
python train_model.py
```

By default, `train_model.py` expects `Fake.csv` and `True.csv` in the backend directory.

The training script exports a single pickle artifact (`prediction_model.pkl`) that encapsulates notebook-style preprocessing, TF-IDF, and ensemble classification.

## Run with Docker

From `fake-news-detector/`:

```bash
docker compose up --build
```

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- Health check: http://localhost:8000/health

The frontend calls `/api/predict`, and Nginx proxies that request to the backend service (`backend:8000`) using Docker container networking.

## Local Development (without Docker)

### Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm start
```

If running frontend locally in dev mode, set `REACT_APP_API_BASE=http://localhost:8000`.

## Vercel Deployment

This repo includes `vercel.json` in `fake-news-detector/` so the project can be deployed as:
- Static frontend build from `frontend/`
- Python API function from `backend/app.py`

### Routes
- `POST /api/predict` → backend ML inference (`FakeNewsModel.predict`)
- `GET /api/health` → backend health endpoint

The frontend already calls `/api/predict`, so deployed requests use the trained model output returned by the backend API.
