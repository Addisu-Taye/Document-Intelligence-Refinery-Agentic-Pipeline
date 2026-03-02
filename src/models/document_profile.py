# src/models/document_profile.py
"""
Document Profile Schema - Output of Triage Agent

This schema classifies documents before extraction begins, governing which
extraction strategy (Fast Text / Layout-Aware / Vision-Augmented) will be used.
"""

from pydantic import BaseModel, Field
from typing import Literal, Optional
from datetime import datetime


class DocumentProfile(BaseModel):
    """
    Classification profile for a document, produced by the Triage Agent.
    
    This profile determines extraction strategy selection and cost estimation.
    All fields are derived from empirical analysis (pdfplumber, layout heuristics).
    """
    
    # === Identity ===
    doc_id: str = Field(..., description="Unique document identifier (filename hash)")
    filename: str = Field(..., description="Original filename")
    profile_created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # === Origin Type (Critical for Strategy Selection) ===
    origin_type: Literal["native_digital", "scanned_image", "mixed", "form_fillable"] = Field(
        ...,
        description="Document origin: native_digital=character stream exists, scanned_image=pure image"
    )
    origin_confidence: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Confidence in origin_type classification (0.0-1.0)"
    )
    
    # === Layout Complexity ===
    layout_complexity: Literal["single_column", "multi_column", "table_heavy", "figure_heavy", "mixed"] = Field(
        ...,
        description="Layout structure complexity"
    )
    layout_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in layout_complexity classification"
    )
    
    # === Content Characteristics ===
    language: str = Field(default="en", description="ISO 639-1 language code")
    language_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    page_count: int = Field(..., gt=0, description="Total pages in document")
    
    # === Domain Classification ===
    domain_hint: Literal["financial", "legal", "technical", "medical", "general"] = Field(
        default="general",
        description="Domain hint for prompt strategy selection"
    )
    domain_keywords_found: list[str] = Field(
        default_factory=list,
        description="Keywords that triggered domain classification"
    )
    
    # === Empirical Metrics (from pdfplumber analysis) ===
    metrics: dict = Field(
        default_factory=dict,
        description="""
        Empirical measurements used for classification:
        - median_char_density: chars per point² (median across sampled pages)
        - median_image_ratio: image area / page area (median)
        - pages_with_text: count of pages with char_count > threshold
        - table_count: total tables detected
        - char_density_stddev: variance indicates mixed content
        """
    )
    
    # === Strategy Recommendation ===
    recommended_strategy: Literal["fast_text", "layout_aware", "vision_augmented"] = Field(
        ...,
        description="Recommended extraction strategy based on profile"
    )
    estimated_extraction_cost: Literal["fast_text_sufficient", "needs_layout_model", "needs_vision_model"] = Field(
        ...,
        description="Cost tier estimate"
    )
    
    # === Escalation Flags ===
    requires_escalation: bool = Field(
        default=False,
        description="True if document characteristics suggest Strategy A may fail"
    )
    escalation_reason: Optional[str] = Field(
        default=None,
        description="Reason for potential escalation (e.g., 'low char density on page 1-3')"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "doc_id": "cbe_report_2024_abc123",
                "filename": "CBE ANNUAL REPORT 2023-24.pdf",
                "origin_type": "mixed",
                "origin_confidence": 0.87,
                "layout_complexity": "table_heavy",
                "layout_confidence": 0.92,
                "language": "en",
                "page_count": 161,
                "domain_hint": "financial",
                "domain_keywords_found": ["balance sheet", "net profit", "ETB", "NBE directive"],
                "metrics": {
                    "median_char_density": 0.00053,
                    "median_image_ratio": 0.666,
                    "pages_with_text": 158,
                    "table_count": 47
                },
                "recommended_strategy": "layout_aware",
                "estimated_extraction_cost": "needs_layout_model",
                "requires_escalation": True,
                "escalation_reason": "Cover pages (1-2) are image-based, content starts page 3"
            }
        }