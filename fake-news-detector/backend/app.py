from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from model import FakeNewsModel


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=10000)


class PredictResponse(BaseModel):
    prediction: str
    confidence: float


ml_model = FakeNewsModel(model_dir="models")


@asynccontextmanager
async def lifespan(_: FastAPI):
    ml_model.load()
    yield


app = FastAPI(
    title="Fake News Detection API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
@app.post("/api/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    try:
        result = ml_model.predict(payload.text)
        return PredictResponse(prediction=result.prediction, confidence=result.confidence)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
