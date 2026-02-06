"""
Hybrid Intelligence Model Wrapper
Combines traditional ML ensemble with transformer architecture
"""

import pickle
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


class HybridVeracityDetector:
    """
    Dual-model architecture combining classical ML and deep learning
    Implements weighted ensemble strategy for robust predictions
    """
    
    # Model weight distribution
    TRADITIONAL_ML_WEIGHT = 0.40
    TRANSFORMER_WEIGHT = 0.60
    
    # Classification labels
    LABEL_FAKE = "FAKE"
    LABEL_REAL = "REAL"
    
    def __init__(self, base_path: str = "models/"):
        """Initialize with model artifact paths"""
        self.artifact_directory = Path(base_path)
        
        # Model components (lazy loaded)
        self.sklearn_ensemble = None
        self.vectorization_engine = None
        self.neural_tokenizer = None
        self.neural_classifier = None
        self.computation_device = self._detect_computation_device()
        
    def _detect_computation_device(self) -> torch.device:
        """Automatically select GPU if available, otherwise CPU"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    
    def initialize_model_artifacts(self):
        """
        Load all model components from disk
        Two-stage loading: traditional ML + transformer
        """
        # Phase 1: Load classical ML models
        ensemble_path = self.artifact_directory / "ml_ensemble.pkl"
        vectorizer_path = self.artifact_directory / "tfidf_vectorizer.pkl"
        
        with open(ensemble_path, 'rb') as ensemble_file:
            self.sklearn_ensemble = pickle.load(ensemble_file)
        
        with open(vectorizer_path, 'rb') as vectorizer_file:
            self.vectorization_engine = pickle.load(vectorizer_file)
        
        # Phase 2: Load transformer components
        transformer_directory = self.artifact_directory / "transformer_model"
        base_model_identifier = "prajjwal1/bert-tiny"
        
        # Check if local model exists, otherwise use base
        if transformer_directory.exists():
            model_source = str(transformer_directory)
        else:
            model_source = base_model_identifier
        
        self.neural_tokenizer = AutoTokenizer.from_pretrained(model_source)
        self.neural_classifier = AutoModelForSequenceClassification.from_pretrained(model_source)
        self.neural_classifier.to(self.computation_device)
        self.neural_classifier.eval()
    
    def _compute_traditional_ml_score(self, text_content: str) -> Dict[str, float]:
        """
        Execute traditional ML pipeline: vectorization -> prediction
        Returns probability distribution
        """
        # Transform text to TF-IDF features
        feature_vector = self.vectorization_engine.transform([text_content])
        
        # Get probability estimates
        probability_matrix = self.sklearn_ensemble.predict_proba(feature_vector)
        
        # Map to label dictionary
        score_distribution = {
            self.LABEL_REAL: float(probability_matrix[0][0]),
            self.LABEL_FAKE: float(probability_matrix[0][1])
        }
        
        return score_distribution
    
    def _compute_transformer_score(self, text_content: str) -> Dict[str, float]:
        """
        Execute transformer pipeline: tokenization -> inference
        Returns softmax probabilities
        """
        # Tokenize with truncation
        encoded_input = self.neural_tokenizer(
            text_content,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        
        # Move tensors to computation device
        encoded_input = {key: val.to(self.computation_device) for key, val in encoded_input.items()}
        
        # Forward pass without gradient computation
        with torch.no_grad():
            model_output = self.neural_classifier(**encoded_input)
            logits = model_output.logits
        
        # Apply softmax to get probabilities
        softmax_probs = torch.softmax(logits, dim=1)
        prob_array = softmax_probs.cpu().numpy()[0]
        
        # Map to label dictionary
        score_distribution = {
            self.LABEL_REAL: float(prob_array[0]),
            self.LABEL_FAKE: float(prob_array[1])
        }
        
        return score_distribution
    
    def _combine_model_outputs(
        self, 
        ml_scores: Dict[str, float], 
        transformer_scores: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Weighted averaging of model predictions
        Uses predefined weight distribution
        """
        combined_scores = {}
        
        for label_name in [self.LABEL_REAL, self.LABEL_FAKE]:
            weighted_score = (
                ml_scores[label_name] * self.TRADITIONAL_ML_WEIGHT +
                transformer_scores[label_name] * self.TRANSFORMER_WEIGHT
            )
            combined_scores[label_name] = weighted_score
        
        return combined_scores
    
    def predict(self, article_text: str) -> Dict[str, Any]:
        """
        Main prediction interface
        Orchestrates dual-model inference and aggregation
        """
        # Start timing
        start_timestamp = time.perf_counter()
        
        # Execute both models in parallel concept
        ml_probability_dist = self._compute_traditional_ml_score(article_text)
        transformer_probability_dist = self._compute_transformer_score(article_text)
        
        # Merge predictions with weights
        final_probability_dist = self._combine_model_outputs(
            ml_probability_dist, 
            transformer_probability_dist
        )
        
        # Determine winning class
        predicted_label = max(final_probability_dist, key=final_probability_dist.get)
        confidence_score = final_probability_dist[predicted_label]
        
        # Calculate elapsed time
        end_timestamp = time.perf_counter()
        elapsed_milliseconds = (end_timestamp - start_timestamp) * 1000
        
        # Package results
        result_payload = {
            "prediction": predicted_label,
            "confidence": confidence_score,
            "probabilities": final_probability_dist,
            "ensemble_scores": {
                "ml_ensemble": ml_probability_dist[predicted_label],
                "transformer": transformer_probability_dist[predicted_label]
            },
            "processing_time_ms": elapsed_milliseconds
        }
        
        return result_payload


# Singleton instance manager
_detector_instance: Optional[HybridVeracityDetector] = None


def get_detector_instance() -> HybridVeracityDetector:
    """
    Retrieves or creates the global detector instance
    Ensures single model loading across application lifecycle
    """
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = HybridVeracityDetector()
    return _detector_instance
