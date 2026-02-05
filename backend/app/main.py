"""
FastAPI Service Entry Point
News Veracity Detection REST API
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Dict, Any
import logging

from .schemas import NewsArticleInput, VeracityAssessmentOutput
from .utils import prepare_text_for_analysis, validate_input_text
from .model import get_detector_instance

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def service_lifecycle(app_instance: FastAPI):
    """
    Lifespan event manager
    Handles model initialization on startup and cleanup on shutdown
    """
    logger.info("🚀 Initializing News Veracity Detection Service...")
    
    # Startup: Load ML models into memory
    try:
        detector = get_detector_instance()
        detector.initialize_model_artifacts()
        logger.info("✅ Model artifacts loaded successfully")
    except Exception as initialization_error:
        logger.error(f"❌ Model initialization failed: {initialization_error}")
        raise
    
    yield  # Service runs here
    
    # Shutdown: Cleanup resources
    logger.info("🛑 Shutting down News Veracity Detection Service...")


# Instantiate FastAPI application with lifespan management
api_service = FastAPI(
    title="Fake News Detection API",
    description="Hybrid ML system for news article veracity assessment",
    version="1.0.0",
    lifespan=service_lifecycle
)


# Configure CORS for frontend integration
permitted_origins = [
    "http://localhost:5173",
    "http://localhost:3000",
]

api_service.add_middleware(
    CORSMiddleware,
    allow_origins=permitted_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@api_service.get("/", response_model=Dict[str, Any])
async def root_endpoint():
    """
    Service information endpoint
    Returns metadata about the API service
    """
    return {
        "service_name": "Fake News Detection API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "health": "/health",
            "prediction": "/predict"
        },
        "description": "Hybrid ML system combining ensemble methods and transformers"
    }


@api_service.get("/health", response_model=Dict[str, str])
async def health_check_endpoint():
    """
    Health monitoring endpoint
    Verifies service readiness
    """
    detector = get_detector_instance()
    
    # Verify critical components are loaded
    models_ready = all([
        detector.sklearn_ensemble is not None,
        detector.vectorization_engine is not None,
        detector.neural_classifier is not None,
        detector.neural_tokenizer is not None
    ])
    
    if not models_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model components not fully initialized"
        )
    
    return {
        "status": "healthy",
        "models_loaded": "true",
        "message": "All systems operational"
    }


@api_service.post("/predict", response_model=VeracityAssessmentOutput)
async def prediction_endpoint(article_input: NewsArticleInput):
    """
    Core prediction endpoint
    Accepts article content and returns veracity assessment
    """
    logger.info("📝 Received prediction request")
    
    try:
        # Step 1: Sanitize input text
        sanitized_text = prepare_text_for_analysis(article_input.text)
        
        # Step 2: Validate content quality
        is_valid, validation_message = validate_input_text(sanitized_text)
        if not is_valid:
            logger.warning(f"⚠️ Validation failed: {validation_message}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Input validation error: {validation_message}"
            )
        
        # Step 3: Combine title and text if title provided
        if article_input.title:
            sanitized_title = prepare_text_for_analysis(article_input.title)
            full_content = f"{sanitized_title}. {sanitized_text}"
        else:
            full_content = sanitized_text
        
        # Step 4: Execute prediction
        detector = get_detector_instance()
        prediction_results = detector.predict(full_content)
        
        logger.info(f"✅ Prediction completed: {prediction_results['prediction']} "
                   f"(confidence: {prediction_results['confidence']:.2f})")
        
        # Step 5: Return structured response
        return VeracityAssessmentOutput(**prediction_results)
    
    except HTTPException:
        raise
    except Exception as processing_error:
        logger.error(f"❌ Prediction error: {processing_error}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction processing failed: {str(processing_error)}"
        )


# Export for uvicorn
app = api_service
