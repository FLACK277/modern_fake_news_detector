"""
Data contracts for the News Veracity Analysis System
Defines input/output structures with validation logic
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict


class NewsArticleInput(BaseModel):
    """
    Incoming article data for veracity assessment
    """
    text: str = Field(
        ..., 
        min_length=10, 
        max_length=10000,
        description="Main article content for analysis"
    )
    title: Optional[str] = Field(
        None,
        max_length=500,
        description="Optional headline or title"
    )
    
    @field_validator('text')
    @classmethod
    def sanitize_text_input(cls, content: str) -> str:
        """Ensure text content is properly formatted"""
        stripped_content = content.strip()
        if not stripped_content:
            raise ValueError('Content cannot be empty or whitespace only')
        return stripped_content
    
    @field_validator('title')
    @classmethod
    def sanitize_title_input(cls, header: Optional[str]) -> Optional[str]:
        """Clean up title if provided"""
        if header:
            cleaned_header = header.strip()
            return cleaned_header if cleaned_header else None
        return header


class VeracityAssessmentOutput(BaseModel):
    """
    Analysis results containing prediction and confidence metrics
    """
    prediction: str = Field(..., description="Classification outcome: REAL or FAKE")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence level (0-1)")
    probabilities: Dict[str, float] = Field(..., description="Class probability distribution")
    ensemble_scores: Dict[str, float] = Field(..., description="Individual model scores")
    processing_time_ms: float = Field(..., gt=0, description="Inference duration in milliseconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "prediction": "FAKE",
                "confidence": 0.87,
                "probabilities": {"REAL": 0.13, "FAKE": 0.87},
                "ensemble_scores": {"ml_ensemble": 0.82, "transformer": 0.90},
                "processing_time_ms": 142.5
            }
        }
