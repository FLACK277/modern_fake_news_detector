from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

@dataclass
class PredictionResult:
    prediction: str
    confidence: float


class FakeNewsModel:
    """Loads a TF-IDF vectorizer and a trained classifier once at startup."""

    def __init__(self, model_dir: str = "models") -> None:
        # Resolve model directory relative to this file so serverless runtimes
        # can find artifacts regardless of current working directory.
        self.model_dir = (Path(__file__).resolve().parent / model_dir).resolve()
        self.model_pipeline = None
        self.vectorizer = None
        self.classifier = None

    def load(self) -> None:
        pipeline_path = self.model_dir / "prediction_model.pkl"
        vectorizer_path = self.model_dir / "tfidf_vectorizer.pkl"
        model_path = self.model_dir / "model.pkl"

        if pipeline_path.exists():
            self.model_pipeline = joblib.load(pipeline_path)
            return

        if vectorizer_path.exists() and model_path.exists():
            self.vectorizer = joblib.load(vectorizer_path)
            self.classifier = joblib.load(model_path)
            return

        raise FileNotFoundError(
            "Model artifacts are missing. Expected either "
            f"{pipeline_path} or both {vectorizer_path} and {model_path}."
        )

    def _ensure_loaded(self) -> None:
        if self.model_pipeline is None and (self.vectorizer is None or self.classifier is None):
            raise RuntimeError("Model is not loaded. Call load() during app startup.")

    def predict(self, text: str) -> PredictionResult:
        self._ensure_loaded()

        if self.model_pipeline is not None:
            probabilities = self.model_pipeline.predict_proba([text])[0]
            classes = self.model_pipeline.classes_
        else:
            features = self.vectorizer.transform([text])
            probabilities = self.classifier.predict_proba(features)[0]
            classes = self.classifier.classes_

        predicted_index = int(np.argmax(probabilities))
        raw_label = classes[predicted_index]
        confidence = float(probabilities[predicted_index])

        label = self._normalize_label(raw_label)
        return PredictionResult(prediction=label, confidence=confidence)

    @staticmethod
    def _normalize_label(raw_label: object) -> str:
        """Maps model output labels to API contract: Fake/Real."""
        if isinstance(raw_label, str):
            normalized = raw_label.strip().lower()
            if normalized in {"fake", "0"}:
                return "Fake"
            if normalized in {"real", "true", "1"}:
                return "Real"

        if isinstance(raw_label, (int, np.integer)):
            return "Real" if int(raw_label) == 1 else "Fake"

        return str(raw_label)
