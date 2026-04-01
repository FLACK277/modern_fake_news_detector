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
pip install -r requirements-train.txt
python train_model.py
```

By default, `train_model.py` expects `Fake.csv` and `True.csv` in the backend directory.

`requirements.txt` is optimized for deployment/runtime. Use `requirements-train.txt` when training artifacts locally.

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

### Before deploy

1. Ensure `backend/models/prediction_model.pkl` exists in the repository.
2. Verify backend dependencies in `backend/requirements.txt` are inference-only runtime deps.
3. Keep frontend API calls relative (`/api/predict`) so Vercel routing works without extra env vars.

### Deploy with Vercel CLI

1. Install and login (one-time):

```bash
npm i -g vercel
vercel login
```

2. Deploy preview from this folder:

```bash
cd fake-news-detector
vercel
```

3. Use these CLI answers on first deploy:
- Set up and deploy: `Y`
- Link to existing project: choose based on whether project already exists
- In which directory is your code located: `./`
- Override settings: `N` (keep `vercel.json`)

4. Deploy production:

```bash
vercel --prod
```

`vercel.json` already routes `/api/*` to `backend/app.py` and serves the React static build for all other routes.

### Deploy from Vercel Dashboard

1. Import this repository in Vercel.
2. Set Root Directory to `fake-news-detector`.
3. Keep framework/build settings as detected from `vercel.json`.
4. Trigger deploy.

### Verify after deploy

- Open `/api/health` and confirm a healthy response.
- Send a POST request to `/api/predict` with:

```json
{
  "text": "Some news content to classify"
}
```

### Routes
- `POST /api/predict` → backend ML inference (`FakeNewsModel.predict`)
- `GET /api/health` → backend health endpoint

The frontend already calls `/api/predict`, so deployed requests use the trained model output returned by the backend API.
